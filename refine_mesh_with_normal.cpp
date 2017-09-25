#include "Geometry/geometryutils.hpp"
#include "Utils/utility.hpp"

#include <QApplication>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOffscreenSurface>
#include <QFile>

#include <GL/freeglut_std.h>

#include <opencv2/opencv.hpp>

#include "common.h"

#include "ceres/ceres.h"

#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/costfunctions.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>
#include <MultilinearReconstruction/OffscreenMeshVisualizer.h>
#include <MultilinearReconstruction/statsutils.h>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/timer/timer.hpp>

namespace fs = boost::filesystem;

#include "json/src/json.hpp"
using json = nlohmann::json;

#include "cost_functions.h"
#include "defs.h"
#include "utils.h"

struct NormalConstraint {
  int fidx;
  Eigen::Vector3d bcoords;
  Eigen::Vector3d n;
};

int main(int argc, char** argv) {
  QApplication a(argc, argv);
  glutInit(&argc, argv);

  //google::InitGoogleLogging(argv[0]);

  // load the settings file
  PhGUtils::message("Loading global settings ...");
  json global_settings = json::parse(ifstream("/home/phg/Codes/FaceShapeFromShading/settings.txt"));
  PhGUtils::message("done.");
  cout << setw(2) << global_settings << endl;

  const string model_filename("/home/phg/Data/Multilinear/blendshape_core.tensor");
  const string id_prior_filename("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor");
  const string exp_prior_filename("/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");
  const string template_mesh_filename("/home/phg/Data/Multilinear/template.obj");
  const string contour_points_filename("/home/phg/Data/Multilinear/contourpoints.txt");
  const string landmarks_filename("/home/phg/Data/Multilinear/landmarks_73.txt");
  const string albedo_index_map_filename("/home/phg/Data/Multilinear/albedo_index.png");
  const string albedo_pixel_map_filename("/home/phg/Data/Multilinear/albedo_pixel.png");
  const string mean_albedo_filename("/home/phg/Data/Texture/mean_texture.png");
  const string core_face_region_filename("/home/phg/Data/Multilinear/albedos/core_face.png");

  const string valid_faces_indices_filename("/home/phg/Data/Multilinear/face_region_indices.txt");
  const string face_boundary_indices_filename("/home/phg/Data/Multilinear/face_boundary_indices.txt");
  const string hair_region_filename("/home/phg/Data/Multilinear/hair_region_indices.txt");

  BasicMesh mesh(template_mesh_filename);
  auto landmarks = LoadIndices(landmarks_filename);
  auto contour_indices = LoadContourIndices(contour_points_filename);

  auto valid_faces_indices_quad = LoadIndices(valid_faces_indices_filename);
  // @HACK each quad face is triangulated, so the indices change from i to [2*i, 2*i+1]
  vector<int> valid_faces_indices;
  for(auto fidx : valid_faces_indices_quad) {
    valid_faces_indices.push_back(fidx*2);
    valid_faces_indices.push_back(fidx*2+1);
  }

  auto faces_boundary_indices_quad = LoadIndices(face_boundary_indices_filename);
  // @HACK each quad face is triangulated, so the indices change from i to [2*i, 2*i+1]
  unordered_set<int> face_boundary_indices;
  for(auto fidx : faces_boundary_indices_quad) {
    face_boundary_indices.insert(fidx*2);
    face_boundary_indices.insert(fidx*2+1);
  }

  auto hair_region_indices_quad = LoadIndices(hair_region_filename);
  // @HACK each quad face is triangulated, so the indices change from i to [2*i, 2*i+1]
  unordered_set<int> hair_region_indices;
  for(auto fidx : hair_region_indices_quad) {
    hair_region_indices.insert(fidx*2);
    hair_region_indices.insert(fidx*2+1);
  }

  MultilinearModel model(model_filename);

  // Start the main process
  {
    fs::path image_filename = argv[1];
    fs::path pts_filename = argv[2];
    fs::path res_filename = argv[3];
    fs::path mask_filename = argv[4];
    fs::path optimized_normal_filename = argv[5];
    fs::path results_path = argv[6];

    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string(), false);
    auto recon_results = LoadReconstructionResult(res_filename.string());

    model.ApplyWeights(recon_results.params_model.Wid, recon_results.params_model.Wexp);
    mesh.UpdateVertices(model.GetTM());
    mesh.ComputeNormals();

    {
      const int max_subdivisions = 1;
      for(int i=0;i<max_subdivisions;++i) {
        mesh.BuildHalfEdgeMesh();
        cout << "Subdivision #" << i << endl;
        mesh.Subdivide();
        cout << "#faces = " << mesh.NumFaces() << endl;
      }

      // HACK: each valid face i becomes [4i, 4i+1, 4i+2, 4i+3] after the each
      // subdivision. See BasicMesh::Subdivide for details
      for(int i=0;i<max_subdivisions;++i) {
        vector<int> valid_faces_indices_new;
        for(auto fidx : valid_faces_indices) {
          int fidx_base = fidx*4;
          valid_faces_indices_new.push_back(fidx_base);
          valid_faces_indices_new.push_back(fidx_base+1);
          valid_faces_indices_new.push_back(fidx_base+2);
          valid_faces_indices_new.push_back(fidx_base+3);
        }
        valid_faces_indices = valid_faces_indices_new;
      }

      mesh.ComputeNormals();
    }

    // render the mesh to FBO with culling to get the visible triangles
    OffscreenMeshVisualizer visualizer(image_points_pair.first.width(), image_points_pair.first.height());
    visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
    visualizer.SetRenderMode(OffscreenMeshVisualizer::Normal);
    visualizer.BindMesh(mesh);
    visualizer.SetCameraParameters(recon_results.params_cam);
    visualizer.SetMeshRotationTranslation(recon_results.params_model.R, recon_results.params_model.T);
    visualizer.SetFacesToRender(valid_faces_indices);

    pair<QImage, vector<float>> img_and_depth = visualizer.RenderWithDepth();
    QImage img = img_and_depth.first;
    const vector<float>& depth = img_and_depth.second;

    // get camera parameters for computing actual z values
    const double aspect_ratio =
      recon_results.params_cam.image_size.x / recon_results.params_cam.image_size.y;

    const double far = recon_results.params_cam.far;
    // near is the focal length
    const double near = recon_results.params_cam.focal_length;
    const double top = near * tan(0.5 * recon_results.params_cam.fovy);
    const double right = top * aspect_ratio;
    glm::dmat4 Mproj = glm::dmat4(near/right, 0, 0, 0,
                                  0, near/top, 0, 0,
                                  0, 0, -(far+near)/(far-near), -1,
                                  0, 0, -2.0 * far * near / (far - near), 0.0);

    glm::ivec4 viewport(0, 0, image_points_pair.first.width(), image_points_pair.first.height());

    glm::dmat4 Rmat = glm::eulerAngleYXZ(recon_results.params_model.R[0],
                                         recon_results.params_model.R[1],
                                         recon_results.params_model.R[2]);

    glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                     glm::dvec3(recon_results.params_model.T[0],
                                                recon_results.params_model.T[1],
                                                recon_results.params_model.T[2]));
    glm::dmat4 Mview = Tmat * Rmat;

    // copy to normal maps and depth maps
    cv::Mat normal_maps_ref = cv::Mat(img.height(), img.width(), CV_64FC3);
    cv::Mat depth_maps_ref = cv::Mat(img.height(), img.width(), CV_64F);
    cv::Mat depth_maps = cv::Mat(img.height(), img.width(), CV_64FC3);
    cv::Mat depth_maps_no_rot = cv::Mat(img.height(), img.width(), CV_64FC3);
    cv::Mat zmaps = cv::Mat(img.height(), img.width(), CV_32F);
    QImage depth_img = img;
    vector<glm::dvec3> point_cloud;
    vector<glm::dvec4> point_cloud_with_id;
    vector<double> output_depth_map; output_depth_map.reserve(img.height()*img.width());
    //#pragma omp parallel for
    for(int y=0;y<img.height();++y) {
      for(int x=0;x<img.width();++x) {
        auto pix = img.pixel(x, y);
        // 0~255 range
        double nx = qRed(pix) / 255.0 * 2.0 - 1.0;
        double ny = qGreen(pix) / 255.0 * 2.0 - 1.0;
        double nz = max(0.0, qBlue(pix) / 255.0 * 2.0 - 1.0);

        double theta, phi;
        tie(theta, phi) = normal2sphericalcoords<double>(nx, ny, nz);
        tie(nx, ny, nz) = sphericalcoords2normal<double>(theta, phi);

        normal_maps_ref.at<cv::Vec3d>(y, x) = cv::Vec3d(nx, ny, nz);

        // get the screen z-value
        double dvalue = depth[(img.height()-1-y)*img.width()+x];
        if(dvalue < 1) {
          // unproject this point to obtain the actual z value
          glm::dvec3 XYZ = glm::unProject(glm::dvec3(x, img.height()-1-y, dvalue), Mview, Mproj, viewport);
          glm::dvec4 Rxyz = Rmat * glm::dvec4(XYZ.x, XYZ.y, XYZ.z, 1);
          point_cloud.push_back(glm::dvec3(Rxyz.x, Rxyz.y, Rxyz.z));
          point_cloud_with_id.push_back(glm::dvec4(Rxyz.x, Rxyz.y, Rxyz.z, y*img.width()+x));
          depth_maps_ref.at<double>(y, x) = Rxyz.z;
          depth_maps.at<cv::Vec3d>(y, x) = cv::Vec3d(Rxyz.x, Rxyz.y, Rxyz.z);
          depth_maps_no_rot.at<cv::Vec3d>(y, x) = cv::Vec3d(XYZ.x, XYZ.y, XYZ.z);
          output_depth_map.push_back(Rxyz.x); output_depth_map.push_back(Rxyz.y); output_depth_map.push_back(Rxyz.z);
          zmaps.at<float>(y, x) = Rxyz.z;
          depth_img.setPixel(x, y, qRgb(dvalue*255, 0, (1-dvalue)*255));
        } else {
          depth_img.setPixel(x, y, qRgb(255, 255, 255));
          depth_maps_ref.at<double>(y, x) = -1e6;
          depth_maps.at<cv::Vec3d>(y, x) = cv::Vec3d(0, 0, -1e6);
          output_depth_map.push_back(0); output_depth_map.push_back(0); output_depth_map.push_back(-1e6);
          zmaps.at<float>(y, x) = -1e6;
        }
      }
    }

    img.save( (results_path / fs::path("normal.png")).string().c_str() );
    depth_img.save( (results_path / fs::path("depth.png")).string().c_str() );


    // for each image bundle, render the mesh to FBO with culling to get the visible triangles
    visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
    visualizer.SetRenderMode(OffscreenMeshVisualizer::Mesh);
    visualizer.BindMesh(mesh);
    visualizer.SetCameraParameters(recon_results.params_cam);
    visualizer.SetMeshRotationTranslation(recon_results.params_model.R, recon_results.params_model.T);
    visualizer.SetIndexEncoded(true);
    visualizer.SetEnableLighting(false);
    QImage mesh_img = visualizer.Render();
    mesh_img.save( (results_path / fs::path("mesh.png")).string().c_str() );

    // find the visible triangles from the index map
    auto triangles_indices_pair = FindTrianglesIndices(mesh_img);
    set<int> triangles = triangles_indices_pair.first;
    vector<int> triangle_indices_map = triangles_indices_pair.second;
    cerr << "Num triangles visible: " << triangles.size() << endl;

    // for each visible pixel, compute its bary-centric coordinates
    cv::Mat mask_image = cv::imread(mask_filename.string(), CV_LOAD_IMAGE_GRAYSCALE);
    QImage optimized_normal_image = QImage(optimized_normal_filename.string().c_str());
    QImage bcoords_image = img;
    QImage normal_constraints_image = img;
    cv::Mat bcoords = cv::Mat(img.height(), img.width(), CV_64FC3);
    vector<NormalConstraint> normal_constraints;
    for(int y=0,pidx=0;y<img.height();++y) {
      for(int x=0;x<img.width();++x,++pidx) {
        auto pix = mask_image.at<unsigned char>(y, x);
        const int fidx = triangle_indices_map[pidx];

        if(pix > 0 && fidx >= 0) {
          cout << pix << " " << fidx << endl;
          auto f = mesh.face(fidx);
          auto v0 = mesh.vertex(f[0]), v1 = mesh.vertex(f[1]), v2 = mesh.vertex(f[2]);

          auto p = depth_maps_no_rot.at<cv::Vec3d>(y, x);
          PhGUtils::Point3f bcoords;
          // Compute barycentric coordinates
          PhGUtils::computeBarycentricCoordinates(PhGUtils::Point3d(p[0], p[1], p[2]),
                                                  PhGUtils::Point3d(v0[0], v0[1], v0[2]),
                                                  PhGUtils::Point3d(v1[0], v1[1], v1[2]),
                                                  PhGUtils::Point3d(v2[0], v2[1], v2[2]),
                                                  bcoords);

          cout << bcoords.x << ", " << bcoords.y << ", " << bcoords.z << endl;
          bcoords.x = clamp<float>(bcoords.x, 0, 1);
          bcoords.y = clamp<float>(bcoords.y, 0, 1);
          bcoords.z = clamp<float>(bcoords.z, 0, 1);

          Eigen::Vector3d bcoords_vec(bcoords.x, bcoords.y, bcoords.z);
          bcoords_vec = bcoords_vec / bcoords_vec.norm();

          auto npix = optimized_normal_image.pixel(x, y);
          double nx, ny, nz;
          tie(nx, ny, nz) = std::make_tuple(qRed(npix)/255.0*2-1.0,
                                            qGreen(npix)/255.0*2-1.0,
                                            qBlue(npix)/255.0*2-1.0);
          Eigen::Vector3d normal_vec(nx, ny, nz);

          normal_constraints.push_back( NormalConstraint{fidx, bcoords_vec, normal_vec} );

          bcoords_image.setPixel(x, y, qRgb(bcoords.x*255.0, bcoords.y*255.0, bcoords.z*255.0));
          normal_constraints_image.setPixel(x, y, npix);
        } else {
          bcoords.at<cv::Vec3d>(y, x) = cv::Vec3d(0, 0, 0);
        }
      }
    }
    bcoords_image.save( (results_path / fs::path("bcoords.png")).string().c_str() );
    normal_constraints_image.save( (results_path / fs::path("normal_constraints.png")).string().c_str() );

    {
      ofstream fout( (results_path / fs::path("constraints.txt") ).string() );
      for(auto c : normal_constraints) {
        fout << c.fidx << " ";
        fout << c.bcoords[0] << ' '
             << c.bcoords[1] << ' '
             << c.bcoords[2] << ' ';
        fout << c.n[0] << ' ' << c.n[1] << ' ' << c.n[2] << endl;
      }
    }
  }

  return 0;
}
