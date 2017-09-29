#include "Geometry/geometryutils.hpp"
#include "Utils/utility.hpp"

#include <QApplication>

#include <GL/freeglut_std.h>

#include <opencv2/opencv.hpp>

#include "common.h"

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include <vector>
#include <cassert>
#include <cmath>
#include <cfloat>

#ifdef __llvm__
double omp_get_wtime() { return 1; }
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 1; }
#else
#include <omp.h>
#endif

using namespace std;

#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/costfunctions.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>
#include <MultilinearReconstruction/OffscreenMeshVisualizer.h>
#include <MultilinearReconstruction/statsutils.h>

#include "defs.h"
#include "utils.h"

// http://www.altdevblogaday.com/2012/05/03/generating-uniformly-distributed-points-on-sphere/
void random_direction(float* result)
{
    float z = 2.0f * rand() / static_cast<float>(RAND_MAX) - 1.0f;
    float t = 2.0f * rand() / static_cast<float>(RAND_MAX) * 3.14f;
    float r = sqrt(1.0f - z * z);
    result[0] = r * cos(t);
    result[1] = r * sin(t);
    result[2] = z;

    //cout << result[0] << ' ' << result[1] << ' ' << result[2] << endl;
}

void raytrace(const char* meshobj, const char* resultpng,
    int nsamples = 128, int tex_size_in = 2048)
{
    // Intel says to do this, so we're doing it.
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    // Load the mesh.
    BasicMesh mesh;
    mesh.LoadOBJMesh(meshobj);
    mesh.ComputeNormals();
    mesh.BuildHalfEdgeMesh();
    mesh.Subdivide();
    mesh.ComputeNormals();

    // Create the embree device and scene.
    RTCDevice device = rtcNewDevice(NULL);
    assert(device && "Unable to create embree device.");
    RTCScene scene = rtcDeviceNewScene(device, RTC_SCENE_STATIC | RTC_SCENE_HIGH_QUALITY,
        RTC_INTERSECT1);
    assert(scene);

    // Populate the embree mesh.
    uint32_t gid = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC,
      mesh.NumFaces(), mesh.NumVertices());
    float* vertices = (float*) rtcMapBuffer(scene, gid, RTC_VERTEX_BUFFER);
    for (size_t i = 0; i < mesh.NumVertices(); i++) {
        *vertices++ = mesh.vertex(i)[0];
        *vertices++ = mesh.vertex(i)[1];
        *vertices++ = mesh.vertex(i)[2];
        vertices++;
    }
    rtcUnmapBuffer(scene, gid, RTC_VERTEX_BUFFER);

    uint32_t* triangles = (uint32_t*) rtcMapBuffer(scene, gid, RTC_INDEX_BUFFER);
    for (size_t i = 0; i < mesh.NumFaces(); i++) {
        *triangles++ = static_cast<uint32_t>(mesh.face(i)[0]);
        *triangles++ = static_cast<uint32_t>(mesh.face(i)[1]);
        *triangles++ = static_cast<uint32_t>(mesh.face(i)[2]);
    }
    rtcUnmapBuffer(scene, gid, RTC_INDEX_BUFFER);
    rtcCommit(scene);

    // Load the triangle indices map and barycentric coordinates map
    const int tex_size = tex_size_in;
    const string albedo_index_map_filename("/home/phg/Data/Multilinear/albedo_index.png");
    const string albedo_pixel_map_filename("/home/phg/Data/Multilinear/albedo_pixel.png");
    const string valid_faces_indices_filename("/home/phg/Data/Multilinear/face_region_indices.txt");

    QImage albedo_index_map = GetIndexMap(albedo_index_map_filename, mesh, true, tex_size);

    vector<vector<PixelInfo>> albedo_pixel_map;
    QImage pixel_map_image;
    tie(pixel_map_image, albedo_pixel_map) = GetPixelCoordinatesMap(albedo_pixel_map_filename,
                                                                    albedo_index_map,
                                                                    mesh,
                                                                    false,
                                                                    tex_size);

#if 0
    auto valid_faces_indices_quad = LoadIndices(valid_faces_indices_filename);
    // @HACK each quad face is triangulated, so the indices change from i to [2*i, 2*i+1]
    vector<int> valid_faces_indices;
    for(auto fidx : valid_faces_indices_quad) {
      valid_faces_indices.push_back(fidx*2);
      valid_faces_indices.push_back(fidx*2+1);
    }

    vector<bool> valid_faces_flag(mesh.NumFaces(), false);
    for(auto fidx : valid_faces_indices) valid_faces_flag[fidx] = true;
#else
    vector<bool> valid_faces_flag(mesh.NumFaces(), true);
#endif

    // Iterate over each pixel in the light map, row by row.
    printf("Rendering ambient occlusion (%d threads)...\n",
        omp_get_max_threads());
    double begintime = omp_get_wtime();
    vector<unsigned char> results(tex_size*tex_size, 0);
    vector<unsigned char> normals(tex_size*tex_size*3, 0);
    vector<unsigned char> positions(tex_size*tex_size*3, 0);

    const uint32_t npixels = tex_size*tex_size;
    const float E = 0.00001f;

    srand(time(NULL));

    vector<vector<float>> dirs(nsamples, vector<float>(3, 0));
#ifdef EVEN_SAMPLING
    const int hstep = sqrt(nsamples*2);
    const int vstep = hstep / 2;
    for(int vi = 0, di = 0; vi < vstep; ++vi) {
      double phi = vi / static_cast<float>(vstep - 1) * 3.1415926;
      for(int hi = 0; hi < hstep; ++hi, ++di) {
        double theta = hi / static_cast<float>(hstep) * 3.1415926 * 2.0;
        dirs[di][0] = cos(theta)*cos(phi);
        dirs[di][1] = sin(theta)*cos(phi);
        dirs[di][2] = sin(phi);
      }
    }
#else
    for(int i=0;i<nsamples;++i) {
      random_direction(&(dirs[i][0]));
    }
#endif

#pragma omp parallel
  {
    RTCRay ray;
    ray.primID = RTC_INVALID_GEOMETRY_ID;
    ray.instID = RTC_INVALID_GEOMETRY_ID;
    ray.mask = 0xFFFFFFFF;
    ray.time = 0.f;

#pragma omp for
    for (uint32_t i = 0; i < npixels; i++) {
      const int r = i / tex_size;
      const int c = i % tex_size;

      // Get the pixel info
      const auto& info_i = albedo_pixel_map[r][c];

      const int fidx = info_i.fidx;
      if(fidx < 0 || !valid_faces_flag[fidx]) continue;

      const glm::vec3& bcoords = info_i.bcoords;

      const Vector3i& face_i = mesh.face(fidx);

      // interpolate the normal
      Vector3d norm_vec = bcoords.x * mesh.vertex_normal(face_i[0])
                        + bcoords.y * mesh.vertex_normal(face_i[1])
                        + bcoords.z * mesh.vertex_normal(face_i[2]);

      // interpolate the position vector
      Vector3d pos_vec = bcoords.x * mesh.vertex(face_i[0])
                       + bcoords.y * mesh.vertex(face_i[1])
                       + bcoords.z * mesh.vertex(face_i[2]);

      norm_vec.normalize();
      pos_vec = pos_vec + 7.5e-3 * norm_vec;

      ray.org[0] = pos_vec[0];
      ray.org[1] = pos_vec[1];
      ray.org[2] = pos_vec[2];

      int nhits = 0;

      // Shoot rays through the differential hemisphere.
      for (int nsamp = 0; nsamp < nsamples; nsamp++) {
          ray.dir[0] = dirs[nsamp][0];
          ray.dir[1] = dirs[nsamp][1];
          ray.dir[2] = dirs[nsamp][2];

          float dotp = norm_vec[0] * ray.dir[0] +
                       norm_vec[1] * ray.dir[1] +
                       norm_vec[2] * ray.dir[2];
          if (dotp < 0) {
              ray.dir[0] = -ray.dir[0];
              ray.dir[1] = -ray.dir[1];
              ray.dir[2] = -ray.dir[2];
          }
          ray.tnear = E;
          ray.tfar = FLT_MAX;
          ray.geomID = RTC_INVALID_GEOMETRY_ID;
          rtcOccluded(scene, ray);
          if (ray.geomID == 0) {
              nhits++;
          }
      }
      float ao = (1.0f - (float) nhits / nsamples);
      results[i] = std::min(255., 255.*ao);

      normals[i*3+0] = (norm_vec[0] + 1.0) * 0.5 * 255.0;
      normals[i*3+1] = (norm_vec[1] + 1.0) * 0.5 * 255.0;
      normals[i*3+2] = (norm_vec[2] + 1.0) * 0.5 * 255.0;

      const float pos_scale = 0.75;
      positions[i*3+0] = (pos_scale * pos_vec[0] + 1.0) * 0.5 * 255.0;
      positions[i*3+1] = (pos_scale * pos_vec[1] + 1.0) * 0.5 * 255.0;
      positions[i*3+2] = (pos_scale * pos_vec[2] + 1.0) * 0.5 * 255.0;
    }
  }

  // Print a one-line performance report.
  double duration = omp_get_wtime() - begintime;
  printf("%f seconds\n", duration);

  // Write the image.
  printf("Writing %s...\n", resultpng);
  QImage resultimg(results.data(), tex_size, tex_size, QImage::Format_Grayscale8);
  resultimg.save("result.png");

  QImage normalimg(normals.data(), tex_size, tex_size, QImage::Format_RGB888);
  normalimg.save("normal.png");

  QImage positionimg(positions.data(), tex_size, tex_size, QImage::Format_RGB888);
  positionimg.save("position.png");

  // Free all embree data.
  rtcDeleteGeometry(scene, gid);
  rtcDeleteScene(scene);
  rtcDeleteDevice(device);
}

int main(int argc, char* argv[]) {
  QApplication a(argc, argv);
  glutInit(&argc, argv);

  raytrace(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]));
  return 0;
}
