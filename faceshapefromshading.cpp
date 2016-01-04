#ifndef FACE_SHAPE_FROM_SHADING_H
#define FACE_SHAPE_FROM_SHADING_H

#include "Geometry/geometryutils.hpp"
#include "Utils/utility.hpp"

#include <QApplication>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOffscreenSurface>
#include <QFile>

#include <GL/freeglut_std.h>
#include "glm/glm.hpp"
#include "gli/gli.hpp"

#include <opencv2/opencv.hpp>

#include "common.h"
#include "OffscreenMeshVisualizer.h"
#include "utils.h"

#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/timer/timer.hpp>
#include <MultilinearReconstruction/costfunctions.h>

namespace fs = boost::filesystem;

struct PixelInfo {
  PixelInfo() : fidx(-1) {}
  PixelInfo(int fidx, glm::vec3 bcoords) : fidx(fidx), bcoords(bcoords) {}

  int fidx;           // trinagle index
  glm::vec3 bcoords;  // bary centric coordinates
};

struct ImageBundle {
  ImageBundle() {}
  ImageBundle(const QImage& image, const vector<Constraint2D>& points, const ReconstructionResult& params)
    : image(image), points(points), params(params) {}
  QImage image;
  vector<Constraint2D> points;
  ReconstructionResult params;
};

int main(int argc, char **argv) {
  QApplication a(argc, argv);
  glutInit(&argc, argv);

  //google::InitGoogleLogging(argv[0]);

  const string model_filename("/home/phg/Data/Multilinear/blendshape_core.tensor");
  const string id_prior_filename("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor");
  const string exp_prior_filename("/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");
  const string template_mesh_filename("/home/phg/Data/Multilinear/template.obj");
  const string contour_points_filename("/home/phg/Data/Multilinear/contourpoints.txt");
  const string landmarks_filename("/home/phg/Data/Multilinear/landmarks_73.txt");
  const string albedo_index_map_filename("/home/phg/Data/Multilinear/albedo_index.png");
  const string albedo_pixel_map_filename("/home/phg/Data/Multilinear/albedo_pixel.png");

  BasicMesh mesh(template_mesh_filename);
  auto landmarks = LoadIndices(landmarks_filename);
  auto contour_indices = LoadContourIndices(contour_points_filename);

  const int tex_size = 2048;

  // Generate index map for albedo
  bool generate_index_map = true;
  QImage albedo_index_map;
  if(QFile::exists(albedo_index_map_filename.c_str()) && (!generate_index_map)) {
    PhGUtils::message("loading index map for albedo.");
    albedo_index_map = QImage(albedo_index_map_filename.c_str());
    albedo_index_map.save("albedo_index.png");
  } else {
    OffscreenMeshVisualizer visualizer(tex_size, tex_size);
    visualizer.BindMesh(mesh);
    visualizer.SetRenderMode(OffscreenMeshVisualizer::Texture);
    visualizer.SetMVPMode(OffscreenMeshVisualizer::OrthoNormal);
    QImage img = visualizer.Render();
    img.save("albedo_index.png");
    albedo_index_map = img;
  }

  // Compute the barycentric coordinates for each pixel
  vector<vector<PixelInfo>> albedo_pixel_map(tex_size, vector<PixelInfo>(tex_size));

  // Generate pixel map for albedo
  bool gen_pixel_map = false;
  QImage pixel_map_image;
  if(QFile::exists(albedo_pixel_map_filename.c_str()) && (!gen_pixel_map)) {
    pixel_map_image = QImage(albedo_pixel_map_filename.c_str());

    PhGUtils::message("generating pixel map for albedo ...");
    boost::timer::auto_cpu_timer t("pixel map for albedo generation time = %w seconds.\n");

    for(int i=0;i<tex_size;++i) {
      for(int j=0;j<tex_size;++j) {
        QRgb pix = albedo_index_map.pixel(j, i);
        unsigned char r = static_cast<unsigned char>(qRed(pix));
        unsigned char g = static_cast<unsigned char>(qGreen(pix));
        unsigned char b = static_cast<unsigned char>(qBlue(pix));
        if(r == 0 && g == 0 && b == 0) continue;
        int fidx;
        decode_index(r, g, b, fidx);

        QRgb bcoords_pix = pixel_map_image.pixel(j, i);

        float x = static_cast<float>(qRed(bcoords_pix)) / 255.0f;
        float y = static_cast<float>(qGreen(bcoords_pix)) / 255.0f;
        float z = static_cast<float>(qBlue(bcoords_pix)) / 255.0f;
        albedo_pixel_map[i][j] = PixelInfo(fidx, glm::vec3(x, y, z));
      }
    }
    PhGUtils::message("done.");
  } else {
    /// @FIXME antialiasing issue because of round-off error
    pixel_map_image = QImage(tex_size, tex_size, QImage::Format_ARGB32);
    pixel_map_image.fill(0);
    PhGUtils::message("generating pixel map for albedo ...");
    boost::timer::auto_cpu_timer t("pixel map for albedo generation time = %w seconds.\n");

    for(int i=0;i<tex_size;++i) {
      for(int j=0;j<tex_size;++j) {
        double y = 1.0 - (i + 0.5) / static_cast<double>(tex_size);
        double x = (j + 0.5) / static_cast<double>(tex_size);

        QRgb pix = albedo_index_map.pixel(j, i);
        unsigned char r = static_cast<unsigned char>(qRed(pix));
        unsigned char g = static_cast<unsigned char>(qGreen(pix));
        unsigned char b = static_cast<unsigned char>(qBlue(pix));
        if(r == 0 && g == 0 && b == 0) continue;
        int fidx;
        decode_index(r, g, b, fidx);

        auto f = mesh.face_texture(fidx);
        auto t0 = mesh.texture_coords(f[0]), t1 = mesh.texture_coords(f[1]), t2 = mesh.texture_coords(f[2]);

        using PhGUtils::Point3f;
        using PhGUtils::Point2d;
        Point3f bcoords;
        // Compute barycentric coordinates
        PhGUtils::computeBarycentricCoordinates(Point2d(x, y),
                                                Point2d(t0[0], t0[1]), Point2d(t1[0], t1[1]), Point2d(t2[0], t2[1]),
                                                bcoords);
        //cerr << bcoords << endl;
        albedo_pixel_map[i][j] = PixelInfo(fidx, glm::vec3(bcoords.x, bcoords.y, bcoords.z));

        pixel_map_image.setPixel(j, i, qRgb(bcoords.x*255, bcoords.y*255, bcoords.z*255));
      }
      pixel_map_image.save("albedo_pixel.jpg");
    }
    PhGUtils::message("done.");
  }

  const string settings_filename(argv[1]);

  // Parse the setting file and load image related resources
  fs::path settings_filepath(settings_filename);

  vector<pair<string, string>> image_points_filenames = ParseSettingsFile(settings_filename);
  vector<ImageBundle> image_bundles;
  for(auto& p : image_points_filenames) {
    fs::path image_filename = settings_filepath.parent_path() / fs::path(p.first);
    fs::path pts_filename = settings_filepath.parent_path() / fs::path(p.second);
    fs::path res_filename = settings_filepath.parent_path() / fs::path(p.first + ".res");
    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string());
    auto recon_results = LoadReconstructionResult(res_filename.string());
    image_bundles.push_back(ImageBundle(image_points_pair.first, image_points_pair.second, recon_results));
  }

  MultilinearModel model(model_filename);
  vector<vector<glm::dvec3>> mean_texture(tex_size, vector<glm::dvec3>(tex_size, glm::dvec3(0, 0, 0)));
  vector<vector<double>> mean_texture_weight(tex_size, vector<double>(tex_size, 0));

  // Collect texture information from each input (image, mesh) pair to obtain mean texture
  QImage mean_texture_image;
  {
    for(auto& bundle : image_bundles) {
      // get the geometry of the mesh, update normal
      model.ApplyWeights(bundle.params.params_model.Wid, bundle.params.params_model.Wexp);
      mesh.UpdateVertices(model.GetTM());
      mesh.ComputeNormals();

      // for each image bundle, render the mesh to FBO with culling to get the visible triangles
      OffscreenMeshVisualizer visualizer(bundle.image.width(), bundle.image.height());
      visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
      visualizer.SetRenderMode(OffscreenMeshVisualizer::Mesh);
      visualizer.BindMesh(mesh);
      visualizer.SetCameraParameters(bundle.params.params_cam);
      visualizer.SetMeshRotationTranslation(bundle.params.params_model.R, bundle.params.params_model.T);
      QImage img = visualizer.Render();

      img.save("mesh.png");

      // find the visible triangles from the index map
      set<int> triangles = FindTrianglesIndices(img);
      cerr << triangles.size() << endl;

      // get the projection parameters
      glm::dmat4 Rmat = glm::eulerAngleYXZ(bundle.params.params_model.R[0], bundle.params.params_model.R[1],
                                           bundle.params.params_model.R[2]);
      glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                       glm::dvec3(bundle.params.params_model.T[0],
                                                  bundle.params.params_model.T[1],
                                                  bundle.params.params_model.T[2]));
      glm::dmat4 Mview = Tmat * Rmat;

      // for each visible triangle, compute the coordinates of its 3 corners
      QImage img_vertices = img;
      vector<vector<glm::dvec3>> triangles_projected;
      for(auto tidx : triangles) {
        auto face_i = mesh.face(tidx);
        auto v0_mesh = mesh.vertex(face_i[0]);
        auto v1_mesh = mesh.vertex(face_i[1]);
        auto v2_mesh = mesh.vertex(face_i[2]);
        glm::dvec3 v0_tri = ProjectPoint(glm::dvec3(v0_mesh[0], v0_mesh[1], v0_mesh[2]), Mview, bundle.params.params_cam);
        glm::dvec3 v1_tri = ProjectPoint(glm::dvec3(v1_mesh[0], v1_mesh[1], v1_mesh[2]), Mview, bundle.params.params_cam);
        glm::dvec3 v2_tri = ProjectPoint(glm::dvec3(v2_mesh[0], v2_mesh[1], v2_mesh[2]), Mview, bundle.params.params_cam);
        triangles_projected.push_back(vector<glm::dvec3>{v0_tri, v1_tri, v2_tri});


        img_vertices.setPixel(v0_tri.x, img.height()-1-v0_tri.y, qRgb(255, 255, 255));
        img_vertices.setPixel(v1_tri.x, img.height()-1-v1_tri.y, qRgb(255, 255, 255));
        img_vertices.setPixel(v2_tri.x, img.height()-1-v2_tri.y, qRgb(255, 255, 255));
      }
      img_vertices.save("mesh_with_vertices.png");

      // for each pixel in the texture map, use backward projection to obtain pixel value in the input image
      // accumulate the texels in average texel map
      for(int i=0;i<tex_size;++i) {
        for(int j=0;j<tex_size;++j) {
          PixelInfo pix_ij = albedo_pixel_map[i][j];

          // skip if the triangle is not visible
          if(triangles.find(pix_ij.fidx) == triangles.end()) continue;

          auto face_i = mesh.face(pix_ij.fidx);

          auto v0_mesh = mesh.vertex(face_i[0]);
          auto v1_mesh = mesh.vertex(face_i[1]);
          auto v2_mesh = mesh.vertex(face_i[2]);

          auto v = v0_mesh * pix_ij.bcoords.x + v1_mesh * pix_ij.bcoords.y + v2_mesh * pix_ij.bcoords.z;

          glm::dvec3 v_img = ProjectPoint(glm::dvec3(v[0], v[1], v[2]), Mview, bundle.params.params_cam);

          // take the pixel from the input image through bilinear sampling
          glm::dvec3 texel = bilinear_sample(bundle.image, v_img.x, bundle.image.height()-1-v_img.y);

          if(texel.r == 0 && texel.g == 0 && texel.b == 0) continue;

          mean_texture[i][j] += texel;
          mean_texture_weight[i][j] += 1.0;
        }
      }
    }

    // [Optional]: render the mesh with texture to verify the texel values
    mean_texture_image = QImage(tex_size, tex_size, QImage::Format_ARGB32);
    mean_texture_image.fill(0);
    for(int i=0;i<tex_size;++i) {
      for (int j = 0; j < tex_size; ++j) {
        double weight_ij = mean_texture_weight[i][j];
        if(weight_ij == 0) continue;
        else {
          glm::dvec3 texel = mean_texture[i][j] / weight_ij;
          mean_texture[i][j] = texel;
          mean_texture_image.setPixel(j, i, qRgb(texel.r, texel.g, texel.b));
        }
      }
    }
    mean_texture_image.save("mean_texture.png");
  }

  {
    // [Shape from shading] initialization
    const int num_images = image_bundles.size();

    vector<VectorXd> ligting_coeffs(num_images);
    vector<cv::Mat> normal_maps(num_images);
    vector<cv::Mat> albedos(num_images);

    // generate reference normal map
    for(int i=0;i<num_images;++i) {
      auto& bundle = image_bundles[i];
      // get the geometry of the mesh, update normal
      model.ApplyWeights(bundle.params.params_model.Wid, bundle.params.params_model.Wexp);
      mesh.UpdateVertices(model.GetTM());
      mesh.ComputeNormals();

      // for each image bundle, render the mesh to FBO with culling to get the visible triangles
      OffscreenMeshVisualizer visualizer(bundle.image.width(), bundle.image.height());
      visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
      visualizer.SetRenderMode(OffscreenMeshVisualizer::Normal);
      visualizer.BindMesh(mesh);
      visualizer.SetCameraParameters(bundle.params.params_cam);
      visualizer.SetMeshRotationTranslation(bundle.params.params_model.R, bundle.params.params_model.T);
      QImage img = visualizer.Render();

      // copy to normal maps
      normal_maps[i] = cv::Mat(img.height(), img.width(), CV_64FC3);
      for(int y=0;y<img.height();++y) {
        for(int x=0;x<img.width();++x) {
          auto pix = img.pixel(x, y);
          // 0~255 range
          normal_maps[i].at<cv::Vec3d>(y, x) = cv::Vec3d(qRed(pix), qGreen(pix), qBlue(pix));
        }
      }

      //img.save(string("normal" + std::to_string(i) + ".png").c_str());

      cv::imwrite("normal" + std::to_string(i) + ".png", normal_maps[i]);

      // convert back to [-1, 1] range
      normal_maps[i] = (normal_maps[i] / 255.0) * 2.0 - 1.0;
    }

    // initialize albedos by rendering the mesh with texture
    for(int i=0;i<num_images;++i) {
      // copy to mean texture to albedos
      auto& bundle = image_bundles[i];

      // get the geometry of the mesh, update normal
      model.ApplyWeights(bundle.params.params_model.Wid, bundle.params.params_model.Wexp);
      mesh.UpdateVertices(model.GetTM());
      mesh.ComputeNormals();

      // for each image bundle, render the mesh to FBO with culling to get the visible triangles
      OffscreenMeshVisualizer visualizer(bundle.image.width(), bundle.image.height());
      visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
      visualizer.SetRenderMode(OffscreenMeshVisualizer::TexturedMesh);
      visualizer.BindMesh(mesh);
      visualizer.BindTexture(mean_texture_image);
      visualizer.SetCameraParameters(bundle.params.params_cam);
      visualizer.SetMeshRotationTranslation(bundle.params.params_model.R, bundle.params.params_model.T);
      QImage img = visualizer.Render();

      QImage albedo_image = visualizer.Render(true);

      albedos[i] = cv::Mat(bundle.image.height(), bundle.image.width(), CV_64FC3);
      for(int y=0;y<albedo_image.height();++y) {
        for(int x=0;x<albedo_image.width();++x) {

          QRgb pix = albedo_image.pixel(x, y);
          unsigned char r = static_cast<unsigned char>(qRed(pix));
          unsigned char g = static_cast<unsigned char>(qGreen(pix));
          unsigned char b = static_cast<unsigned char>(qBlue(pix));

          // 0~255 range
          albedos[i].at<cv::Vec3d>(y, x) = cv::Vec3d(r, g, b);
        }
      }

      cv::imwrite("albedo" + std::to_string(i) + ".png", albedos[i]);

      // convert to [0, 1] range
      albedos[i] /= 255.0;
    }

    // [Shape from shading] main loop

    // fix albedo and normal map, estimate lighting coefficients
    for(int i=0;i<num_images;++i) {
      auto& bundle = image_bundles[i];
      // collect constraints
      vector<glm::dvec3> normals_i;
      vector<glm::dvec3> albedo_i;
      vector<glm::dvec3> pixels_i;
      for(int y=0;y<normal_maps[i].rows;++y) {
        for(int x=0;x<normal_maps[i].cols;++x) {
          cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(y, x);
          if(pix[0] == 0 && pix[1] == 0 && pix[2] == 0) continue;
          else {
            // restore r, g, b order -> (x, y, z) for normal
            normals_i.push_back(glm::dvec3(pix[0], pix[1], pix[2]));

            cv::Vec3d pix_albedo = albedos[i].at<cv::Vec3d>(y, x);
            albedo_i.push_back(glm::dvec3(pix_albedo[0], pix_albedo[1], pix_albedo[2]));

            auto pix_i = bundle.image.pixel(x, y);
            pixels_i.push_back(glm::dvec3(qRed(pix_i) / 255.0, qGreen(pix_i) / 255.0, qBlue(pix_i) / 255.0));
          }
        }
      }

      // solve it
      const int num_constraints = normals_i.size();
      const int num_dof = 4;  // use first order approximation
      MatrixXd A(num_constraints * 3, num_dof);
      VectorXd b(num_constraints * 3);
      for(int i=0, offset=0;i<num_constraints;++i, offset+=3) {
        A(offset, 0) = 1.0; A(offset, 1) = normals_i[i].x; A(offset, 1) = normals_i[i].y; A(offset, 1) = normals_i[i].z;
        A.row(offset+1) = A.row(offset);
        A.row(offset+2) = A.row(offset);
        A.row(offset) *= albedo_i[i].r;
        A.row(offset+1) *= albedo_i[i].g;
        A.row(offset+2) *= albedo_i[i].b;

        b(offset) = pixels_i[i].r;
        b(offset+1) = pixels_i[i].g;
        b(offset+2) = pixels_i[i].b;
      }

      VectorXd l_i = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
      cout << l_i.transpose() << endl;

      // [Optional] output result of estimated lighting
      cv::Mat image_with_lighting = normal_maps[i].clone();
      for(int y=0;y<normal_maps[i].rows;++y) {
        for(int x=0;x<normal_maps[i].cols;++x) {
          cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(y, x);
          if(pix[0] == 0 && pix[1] == 0 && pix[2] == 0) continue;
          else {
            VectorXd Y(4);
            Y(0) = 1.0; Y(1) = pix[0]; Y(2) = pix[1]; Y(3) = pix[2];

            double LdotY = l_i.transpose() * Y;
            cv::Vec3d rho = albedos[i].at<cv::Vec3d>(y, x) * 255.0 * LdotY;

            image_with_lighting.at<cv::Vec3d>(y, x) = cv::Vec3d(rho[0], rho[1], rho[2]);
          }
        }
      }
      cv::imwrite("lighting" + std::to_string(i) + ".png", image_with_lighting);
    }

    // fix albedo and lighting, estimate depth

    // fix depth and lighting, estimate albedo
  }

  return 0;
}

#endif  // FACE_SHAPE_FROM_SHADING
