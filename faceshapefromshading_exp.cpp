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
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>

namespace fs = boost::filesystem;
namespace po = boost::program_options;

#include "json/src/json.hpp"
using json = nlohmann::json;

#include "cost_functions.h"
#include "defs.h"
#include "utils.h"

po::variables_map ParseCommandlineOptions(int argc, char** argv) {
  po::options_description desc("Options");
  desc.add_options()
    ("help", "Print help messages")
    ("settings_file", po::value<string>()->required(), "Settings file.")
    ("blendshapes_path", po::value<string>()->required(), "Input blendshapes path.")
    ("init_recon_path", po::value<string>()->required(), "Initial reconstructions path.")
    ("iter", po::value<int>()->required(), "The iteration number.");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(vm.count("help")) {
      cout << desc << endl;
      exit(1);
    }
    return vm;
  } catch(po::error& e) {
    cerr << "Error: " << e.what() << endl;
    cerr << desc << endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  po::variables_map vm = ParseCommandlineOptions(argc, argv);

  QApplication a(argc, argv);
  glutInit(&argc, argv);

  //google::InitGoogleLogging(argv[0]);

  // load the settings file
  PhGUtils::message("Loading global settings ...");
  json global_settings = json::parse(ifstream("/home/phg/Codes/FaceShapeFromShading/settings.txt"));
  PhGUtils::message("done.");
  cout << setw(2) << global_settings << endl;

  // Multilinear model related files
  const string model_filename("/home/phg/Data/Multilinear/blendshape_core.tensor");
  const string id_prior_filename("/home/phg/Data/Multilinear/blendshape_u_0_aug.tensor");
  const string exp_prior_filename("/home/phg/Data/Multilinear/blendshape_u_1_aug.tensor");

  const string template_mesh_filename("/home/phg/Data/Multilinear/template.obj");
  const string contour_points_filename("/home/phg/Data/Multilinear/contourpoints.txt");
  const string landmarks_filename("/home/phg/Data/Multilinear/landmarks_73.txt");

  // The following files are related to the base template, i.e. the one before subdivision.
  // TODO Need to create a set of index map and pixel map for different level of subdivisions.
  // Maybe in the form of face indices mapping
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

  const int tex_size = 2048;

  // HACK: subdivie the template mesh so it has the same topology as the input
  // blendshapes
  // Subdivide the mesh twice
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

  // Generate index map for albedo
  const bool gen_albedo_index_map = true;
  QImage albedo_index_map = GetIndexMap(albedo_index_map_filename,
                                        mesh,
                                        gen_albedo_index_map,
                                        tex_size);

  // Compute the barycentric coordinates for each pixel
  const bool gen_albedo_pixel_map = true;
  vector<vector<PixelInfo>> albedo_pixel_map;
  QImage pixel_map_image;
  tie(pixel_map_image, albedo_pixel_map) = GetPixelCoordinatesMap(albedo_pixel_map_filename,
                                                                  albedo_index_map,
                                                                  mesh,
                                                                  gen_albedo_pixel_map,
                                                                  tex_size);

  const string settings_filename = vm["settings_file"].as<string>();
  int iteration_index = vm["iter"].as<int>();
  const string recon_path = vm["init_recon_path"].as<string>();
  const string blendshapes_path = vm["blendshapes_path"].as<string>();

  // Parse the setting file and load image related resources
  fs::path settings_filepath(settings_filename);

  // Create SFS results directory
  fs::path image_files_path = settings_filepath.parent_path();
  fs::path results_path = image_files_path / fs::path("iteration_" + to_string(iteration_index)) / fs::path("SFS");
  fs::create_directories(results_path);

  // Load the settings file
  cout << "Reading settings file " << settings_filename << endl;
  vector<pair<string, string>> image_points_filenames = ParseSettingsFile(settings_filename);
  cout << image_points_filenames.size() << " input images." << endl;

  // Load the image bundles: image, points and its reconstruction result
  vector<ImageBundle> image_bundles;
  for(auto& p : image_points_filenames) {
    fs::path image_filename = settings_filepath.parent_path() / fs::path(p.first);
    fs::path pts_filename = settings_filepath.parent_path() / fs::path(p.second);
    fs::path res_filename = fs::path(recon_path) / fs::path(p.first + ".res");
    cout << "[" << image_filename << ", " << pts_filename << "]" << endl;

    auto image_points_pair = LoadImageAndPoints(image_filename.string(), pts_filename.string(), false);
    auto recon_results = LoadReconstructionResult(res_filename.string());
    image_bundles.push_back(ImageBundle(image_points_pair.first, image_points_pair.second, recon_results));
  }
  cout << "Image bundles loaded." << endl;

  // Load all the input blendshapes
  const int num_blendshapes = 46;
  vector<BasicMesh> blendshapes(num_blendshapes+1);
  for(int i=0;i<=num_blendshapes;++i) {
    blendshapes[i].LoadOBJMesh( blendshapes_path + "/" + "B_" + to_string(i) + ".obj" );
    blendshapes[i].ComputeNormals();
  }

  MultilinearModel model(model_filename);

  vector<vector<glm::dvec3>> mean_texture(tex_size, vector<glm::dvec3>(tex_size, glm::dvec3(0, 0, 0)));
  cv::Mat mean_texture_mat(tex_size, tex_size, CV_64FC3);
  vector<vector<double>> mean_texture_weight(tex_size, vector<double>(tex_size, 0));

  // Collect texture information from each input (image, mesh) pair to obtain mean texture
  QImage mean_texture_image;
  vector<vector<int>> face_indices_maps;
  json mean_texture_options = R"(
    {
      "generate_mean_texture": true,
      "refine_method": "hsv",
      "hsv_threshold": 0.1,
      "use_blendshapes": true
    }
  )"_json;
  mean_texture_options["core_face_region_filename"] = core_face_region_filename;

  tie(mean_texture_image, face_indices_maps) = GenerateMeanTexture(
    image_bundles,
    model,  // it is not used when use_blendshapes = true
    blendshapes,
    mesh,
    tex_size,
    albedo_pixel_map,
    mean_texture,
    mean_texture_weight,
    mean_texture_mat,
    mean_albedo_filename,
    results_path,
    mean_texture_options.dump()
  );


  // [Shape from shading]
  {
    // [Shape from shading] initialization
    const int num_images = image_bundles.size();

    vector<VectorXd> lighting_coeffs(num_images, VectorXd::Zero(9));
    for(auto &lco : lighting_coeffs) lco[0] = 1.0;

    vector<cv::Mat> normal_maps_ref(num_images);
    vector<cv::Mat> normal_maps_ref_LoG(num_images);
    vector<cv::Mat> normal_maps(num_images);
    vector<cv::Mat> depth_maps_ref(num_images);
    vector<cv::Mat> depth_maps_ref_LoG(num_images);
    vector<cv::Mat> depth_maps(num_images);
    vector<cv::Mat> zmaps(num_images);
    vector<vector<int>> valie_pixels_map(num_images);
    vector<cv::Mat> albedos_ref(num_images);
    vector<cv::Mat> albedos_ref_LoG(num_images);
    vector<cv::Mat> albedos(num_images);

    vector<vector<glm::ivec2>> valid_depth_pixels(num_images);

    // generate reference normal map and depth map
    for(int i=0;i<num_images;++i) {
      auto& bundle = image_bundles[i];
      // get the geometry of the mesh, update normal
      /*
      model.ApplyWeights(bundle.params.params_model.Wid, bundle.params.params_model.Wexp);
      mesh.UpdateVertices(model.GetTM());
      mesh.ComputeNormals();
      */
      ApplyWeights(mesh, blendshapes, bundle.params.params_model.Wexp_FACS);

      // for each image bundle, render the mesh to FBO with culling to get the visible triangles
      OffscreenMeshVisualizer visualizer(bundle.image.width(), bundle.image.height());
      visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
      visualizer.SetRenderMode(OffscreenMeshVisualizer::Normal);
      visualizer.BindMesh(mesh);
      visualizer.SetCameraParameters(bundle.params.params_cam);
      visualizer.SetMeshRotationTranslation(bundle.params.params_model.R, bundle.params.params_model.T);
      visualizer.SetFacesToRender(valid_faces_indices);

      pair<QImage, vector<float>> img_and_depth = visualizer.RenderWithDepth();
      QImage img = img_and_depth.first;
      const vector<float>& depth = img_and_depth.second;

      // get camera parameters for computing actual z values
      const double aspect_ratio =
        bundle.params.params_cam.image_size.x / bundle.params.params_cam.image_size.y;

      const double far = bundle.params.params_cam.far;
      // near is the focal length
      const double near = bundle.params.params_cam.focal_length;
      const double top = near * tan(0.5 * bundle.params.params_cam.fovy);
      const double right = top * aspect_ratio;
      glm::dmat4 Mproj = glm::dmat4(near/right, 0, 0, 0,
                                    0, near/top, 0, 0,
                                    0, 0, -(far+near)/(far-near), -1,
                                    0, 0, -2.0 * far * near / (far - near), 0.0);

      glm::ivec4 viewport(0, 0, bundle.image.width(), bundle.image.height());

      glm::dmat4 Rmat = glm::eulerAngleYXZ(bundle.params.params_model.R[0],
                                           bundle.params.params_model.R[1],
                                           bundle.params.params_model.R[2]);

      glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                       glm::dvec3(bundle.params.params_model.T[0],
                                                  bundle.params.params_model.T[1],
                                                  bundle.params.params_model.T[2]));
      glm::dmat4 Mview = Tmat * Rmat;

      // copy to normal maps and depth maps
      normal_maps_ref[i] = cv::Mat(img.height(), img.width(), CV_64FC3);
      depth_maps_ref[i] = cv::Mat(img.height(), img.width(), CV_64F);
      depth_maps[i] = cv::Mat(img.height(), img.width(), CV_64FC3);
      zmaps[i] = cv::Mat(img.height(), img.width(), CV_32F);
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

          normal_maps_ref[i].at<cv::Vec3d>(y, x) = cv::Vec3d(nx, ny, nz);

          // get the screen z-value
          double dvalue = depth[(img.height()-1-y)*img.width()+x];
          if(dvalue < 1) {
            // unproject this point to obtain the actual z value
            glm::dvec3 XYZ = glm::unProject(glm::dvec3(x, img.height()-1-y, dvalue), Mview, Mproj, viewport);
            glm::dvec4 Rxyz = Rmat * glm::dvec4(XYZ.x, XYZ.y, XYZ.z, 1);
            point_cloud.push_back(glm::dvec3(Rxyz.x, Rxyz.y, Rxyz.z));
            point_cloud_with_id.push_back(glm::dvec4(Rxyz.x, Rxyz.y, Rxyz.z, y*img.width()+x));
            depth_maps_ref[i].at<double>(y, x) = Rxyz.z;
            depth_maps[i].at<cv::Vec3d>(y, x) = cv::Vec3d(Rxyz.x, Rxyz.y, Rxyz.z);
            output_depth_map.push_back(Rxyz.x); output_depth_map.push_back(Rxyz.y); output_depth_map.push_back(Rxyz.z);
            zmaps[i].at<float>(y, x) = Rxyz.z;
            depth_img.setPixel(x, y, qRgb(dvalue*255, 0, (1-dvalue)*255));
            valie_pixels_map[i].push_back(y * img.width() + x);
          } else {
            depth_img.setPixel(x, y, qRgb(255, 255, 255));
            depth_maps_ref[i].at<double>(y, x) = -1e6;
            depth_maps[i].at<cv::Vec3d>(y, x) = cv::Vec3d(0, 0, -1e6);
            output_depth_map.push_back(0); output_depth_map.push_back(0); output_depth_map.push_back(-1e6);
            zmaps[i].at<float>(y, x) = -1e6;
          }
        }
      }

      img.save( (results_path / fs::path("normal" + std::to_string(i) + ".png")).string().c_str() );
      depth_img.save( (results_path / fs::path("depth" + std::to_string(i) + ".png")).string().c_str() );

      // Write out the entire depth map
      {
        ofstream fout( (results_path / fs::path("depth_map" + std::to_string(i) + ".bin")).string(), ios::binary );
        int depth_map_size[] = {img.height(), img.width()};
        fout.write(reinterpret_cast<char*>(depth_map_size), sizeof(int)*2);
        fout.write(reinterpret_cast<char*>(output_depth_map.data()), sizeof(double)*img.height()*img.width()*3);
        fout.close();
      }

      // Write out the depth map as a per-pixel mesh
      {
        ofstream fout((results_path / fs::path("depth_mesh" + std::to_string(i) + ".obj")).string());

        vector<int> depth_node_map(img.width()*img.height(), 0);
        for(int j=0;j<point_cloud_with_id.size();++j) {
          auto& p = point_cloud_with_id[j];
          fout << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
          depth_node_map[static_cast<int>(p.w)] = j + 1;
        }

        for(int j=0;j<point_cloud_with_id.size();++j) {
          int idx = point_cloud_with_id[j].w;
          int r = idx / img.width(), c = idx % img.width();
          int lidx = idx - 1;
          int ridx = idx + 1;
          int uidx = idx - img.width();
          int didx = idx + img.width();
          if(ridx < img.width()*img.height() && didx < img.width()*img.height()) {
            if(depth_node_map[ridx] > 0 && depth_node_map[didx] > 0) {
              fout << "f " << depth_node_map[idx] << " " << depth_node_map[didx] << " " << depth_node_map[ridx] << '\n';
            }
          }
          if(lidx >= 0 && uidx >= 0) {
            if(depth_node_map[lidx] > 0 && depth_node_map[uidx] > 0) {
              fout << "f " << depth_node_map[idx] << " " << depth_node_map[uidx] << " " << depth_node_map[lidx] << '\n';
            }
          }
        }
        fout.close();
      }

      // Write out the initial point cloud
      {
        ofstream fout( (results_path / fs::path("point_cloud" + std::to_string(i) + ".txt")).string() );
        for(auto p : point_cloud) {
          fout << p.x << ' ' << p.y << ' ' << p.z << endl;
        }
        fout.close();
      }
    }
    // make a copy, use it as initial value
    normal_maps = normal_maps_ref;

    // initialize albedos by rendering the mesh with texture
    for(int i=0;i<num_images;++i) {
      // copy to mean texture to albedos
      auto& bundle = image_bundles[i];

      // get the geometry of the mesh, update normal
      /*
      model.ApplyWeights(bundle.params.params_model.Wid, bundle.params.params_model.Wexp);
      mesh.UpdateVertices(model.GetTM());
      mesh.ComputeNormals();
      */
      ApplyWeights(mesh, blendshapes, bundle.params.params_model.Wexp_FACS);

      // for each image bundle, render the mesh to FBO with culling to get the visible triangles
      OffscreenMeshVisualizer visualizer(bundle.image.width(), bundle.image.height());
      visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
      visualizer.SetRenderMode(OffscreenMeshVisualizer::TexturedMesh);
      visualizer.BindMesh(mesh);
      visualizer.BindTexture(mean_texture_image);
      visualizer.SetCameraParameters(bundle.params.params_cam);
      visualizer.SetMeshRotationTranslation(bundle.params.params_model.R, bundle.params.params_model.T);
      visualizer.SetFacesToRender(valid_faces_indices);

      QImage albedo_image = visualizer.Render(true);

      albedos_ref[i] = cv::Mat(bundle.image.height(), bundle.image.width(), CV_64FC3);
      //#pragma omp parallel for
      for(int y=0;y<albedo_image.height();++y) {
        for(int x=0;x<albedo_image.width();++x) {

          QRgb pix = albedo_image.pixel(x, y);
          unsigned char r = static_cast<unsigned char>(qRed(pix));
          unsigned char g = static_cast<unsigned char>(qGreen(pix));
          unsigned char b = static_cast<unsigned char>(qBlue(pix));

          // convert from BGR to RGB
          albedo_image.setPixel(x, y, qRgb(b, g, r));
        }
      }

      albedo_image.save( (results_path / fs::path("albedo" + std::to_string(i) + ".png")).string().c_str() );

      // color transfer from bundle.image to albedo_image, so the initial albedo
      // is a better match
      albedo_image = TransferColor(albedo_image, bundle.image, valie_pixels_map[i], valie_pixels_map[i]);

      albedo_image.save( (results_path / fs::path("albedo_transferred_" + std::to_string(i) + ".png")).string().c_str() );

      //#pragma omp parallel for
      for(int y=0;y<albedo_image.height();++y) {
        for(int x=0;x<albedo_image.width();++x) {
          QRgb pix = albedo_image.pixel(x, y);
          unsigned char r = static_cast<unsigned char>(qRed(pix));
          unsigned char g = static_cast<unsigned char>(qGreen(pix));
          unsigned char b = static_cast<unsigned char>(qBlue(pix));
          // 0~255 range
          albedos_ref[i].at<cv::Vec3d>(y, x) = cv::Vec3d(r, g, b);
        }
      }
      // convert to [0, 1] range
      albedos_ref[i] /= 255.0;
    }

    // HACK In preparation only mode, this program generates initial normal map,
    // albedo, depth map and point clouds. The actual SFS is done in a separate
    // program.
    if (bool(global_settings["preparation_only"])) {
      return 0;
    }

    // transfer the color from the input image to the reference albedo as initial albedo
    albedos = albedos_ref;

    for(int i=0;i<num_images;++i) {

      auto &bundle = image_bundles[i];

      // ====================================================================
      // construct LoG matrix for this image
      // ====================================================================
      const int num_rows = bundle.image.height(), num_cols = bundle.image.width();
      using Tripletd = Eigen::Triplet<double>;
      using SparseMatrixd = Eigen::SparseMatrix<double, Eigen::RowMajor>;

      const int kLoG = 2;
      const double sigmaLoG = 1.0;
      MatrixXd LoG = ComputeLoGKernel(kLoG, sigmaLoG);

      vector<Tripletd> LoG_coeffs;
      vector<vector<pair<int, double>>> LoG_coeffs_perpixel(num_rows*num_cols);
      SparseMatrixd M_LoG(num_rows*num_cols, num_rows*num_cols);
      {
        boost::timer::auto_cpu_timer timer("[Shape from shading] M_LoG computation time = %w seconds.\n");
        // collect the coefficients for each pixel
        for (int r = 0; r < num_rows; ++r) {
          for (int c = 0; c < num_cols; ++c) {
            int pidx = r * num_cols + c;
            for (int kr = -kLoG; kr <= kLoG; ++kr) {
              int ri = r + kr;
              if (ri < 0 || ri >= num_rows) continue;
              for (int kc = -kLoG; kc <= kLoG; ++kc) {
                int ci = c + kc;
                if (ci < 0 || ci >= num_cols) continue;

                int qidx = ri * num_cols + ci;

                // add this element to the matrix
                LoG_coeffs.push_back(Tripletd(pidx, qidx, LoG(kr+kLoG, kc+kLoG)));

                LoG_coeffs_perpixel[pidx].push_back(make_pair(qidx, LoG(kr+kLoG, kc+kLoG)));
              }
            }

          }
        }
        M_LoG.setFromTriplets(LoG_coeffs.begin(), LoG_coeffs.end());
      }

      // ====================================================================
      // compute LoG filtered reference albedo and normal map
      // ====================================================================
      cv::Mat LoG_kernel(kLoG*2+1, kLoG*2+1, CV_64F);
      for(int kr=0;kr<kLoG*2+1;++kr) {
        for(int kc=0;kc<kLoG*2+1;++kc) {
          LoG_kernel.at<double>(kr, kc) = LoG(kr, kc);
        }
      }

      cv::filter2D(albedos_ref[i], albedos_ref_LoG[i], -1, LoG_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
      cv::imwrite( (results_path / fs::path("albedo_LoG" + std::to_string(i) + ".png")).string(), (albedos_ref_LoG[i] + 0.5) * 255.0);

      // store it in num_pixels-by-3 matrix
      MatrixXd albedo_ref_LoG_i(num_rows*num_cols, 3);
      for(int r=0, pidx=0;r<num_rows;++r) {
        for(int c=0;c<num_cols;++c,++pidx) {
          cv::Vec3d pix = albedos_ref_LoG[i].at<cv::Vec3d>(r, c);
          albedo_ref_LoG_i(pidx, 0) = pix[0];
          albedo_ref_LoG_i(pidx, 1) = pix[1];
          albedo_ref_LoG_i(pidx, 2) = pix[2];
        }
      }

      cv::filter2D(normal_maps_ref[i], normal_maps_ref_LoG[i], -1, LoG_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
      cv::imwrite( (results_path / fs::path("normal_LoG" + std::to_string(i) + ".png")).string(), (normal_maps_ref_LoG[i] + 1.0) * 0.5 * 255.0);

      // store it in num_pixels-by-3 matrix
      MatrixXd normal_map_ref_LoG_i(num_rows*num_cols, 3);
      for(int r=0, pidx=0;r<num_rows;++r) {
        for(int c=0;c<num_cols;++c,++pidx) {
          cv::Vec3d pix = normal_maps_ref_LoG[i].at<cv::Vec3d>(r, c);
          normal_map_ref_LoG_i(pidx, 0) = pix[0];
          normal_map_ref_LoG_i(pidx, 1) = pix[1];
          normal_map_ref_LoG_i(pidx, 2) = pix[2];
        }
      }

      cv::filter2D(depth_maps_ref[i], depth_maps_ref_LoG[i], -1, LoG_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
      VectorXd depth_map_ref_LoG_i(num_rows*num_cols);
      for(int r=0, pidx=0;r<num_rows;++r) {
        for(int c=0;c<num_cols;++c,++pidx) {
          normal_map_ref_LoG_i(pidx) = depth_maps_ref_LoG[i].at<double>(r, c);
        }
      }

      cv::Mat dzmapdx, dzmapdy;
      cv::Sobel(zmaps[i], dzmapdx, -1, 1, 0);
      cv::Sobel(zmaps[i], dzmapdy, -1, 0, 1);
      cv::Mat dz_gradient(num_rows, num_cols, CV_32F);
      for(int i=0;i<num_rows;++i) {
        for(int j=0;j<num_cols;++j) {
          float dzdx = dzmapdx.at<float>(i, j);
          float dzdy = dzmapdy.at<float>(i, j);

          dz_gradient.at<float>(i, j) = sqrt(dzdx * dzdx + dzdy * dzdy);
        }
      }
      cv::threshold(dz_gradient, dz_gradient, 0.1, 255, cv::THRESH_BINARY);
      cv::dilate(dz_gradient, dz_gradient, cv::Mat());
      cv::imwrite( (results_path / fs::path("zmap_gradient" + std::to_string(i) + ".png")).string().c_str(), dz_gradient );

      vector<bool> is_boundary(num_rows * num_cols, false);

      cout << "Shape from shading ..." << endl;
      const int max_iters = global_settings["max_iters"];
      int iters = 0;

      double second_order_weights = 0;
      const double second_order_scale = 0.0;

      // [Shape from shading] main loop
      while(iters++ < max_iters){
        cout << "iteration " << iters << endl;
        // [Shape from shading] step 1: fix albedo and normal map, estimate lighting coefficients
        {
          second_order_weights = min((iters - 1) / static_cast<double>(max_iters - 1), 1.0) * second_order_scale;

          // ====================================================================
          // collect valid pixels
          // ====================================================================
          vector<glm::ivec2> pixel_indices_i;

          for (int y = 0; y < normal_maps[i].rows; ++y) {
            for (int x = 0; x < normal_maps[i].cols; ++x) {
              float zval = zmaps[i].at<float>(y, x);
              int pidx = y * num_cols + x;
              bool is_good_pixel = true;
              is_good_pixel &= (zval > -1e5);
              is_good_pixel &= (hair_region_indices.count(face_indices_maps[i][pidx]) == 0);
              is_good_pixel &= (face_boundary_indices.count(face_indices_maps[i][pidx]) == 0);

              auto pix = bundle.image.pixel(x, y);
              const int SATURATED_THRESHOLD = global_settings["lighting"]["saturated_pixels_threshold"];
              if(qRed(pix) + qGreen(pix) + qBlue(pix) > SATURATED_THRESHOLD * 3) {
                is_good_pixel = false;
              }
              const int DARK_PIXEL_THRESHOLD = global_settings["lighting"]["dark_pixels_threshold"];
              if(qRed(pix) + qGreen(pix) + qBlue(pix) < DARK_PIXEL_THRESHOLD * 3) {
                is_good_pixel = false;
              }

              if(is_good_pixel) pixel_indices_i.push_back(glm::ivec2(y, x));
            }
          }

          // ====================================================================
          // filter pixels
          // ====================================================================
          vector<double> albedo_distances_i(num_cols*num_rows);
          vector<double> albedo_distances_i_vec;
          for(int j = 0; j < pixel_indices_i.size(); ++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            cv::Vec3d pix_ref = albedos_ref[i].at<cv::Vec3d>(r, c);
            cv::Vec3d pix = albedos[i].at<cv::Vec3d>(r, c);
            cv::Vec3d pix_diff = pix_ref - pix_diff;
            double d_j = pix_diff[0] * pix_diff[0] + pix_diff[1] * pix_diff[1] + pix_diff[2] * pix_diff[2];
            albedo_distances_i[r*num_cols+c] = d_j;
            albedo_distances_i_vec.push_back(d_j);
          }

          std::sort(pixel_indices_i.begin(), pixel_indices_i.end(), [&](glm::ivec2 p1, glm::ivec2 p2) {
            return albedo_distances_i[p1.x*num_cols+p1.y] < albedo_distances_i[p2.x*num_cols+p2.y];
          });

          vector<double> albedo_distances_i_sorted = albedo_distances_i_vec;
          std::sort(albedo_distances_i_sorted.begin(), albedo_distances_i_sorted.end());

          const int nbins = global_settings["lighting"]["albedo_distance_bins"];
          vector<int> counter(nbins, 0);
          double max_albedo_distance = albedo_distances_i_sorted.back(), min_albedo_distance = albedo_distances_i_sorted.front();
          double diff_albedo_distance = max(max_albedo_distance - min_albedo_distance, 1e-16);
          cout << min_albedo_distance << ", " << max_albedo_distance << ", " << diff_albedo_distance << endl;
          for(auto d_j : albedo_distances_i_sorted) {
            int binidx = min(static_cast<int>((d_j - min_albedo_distance) / diff_albedo_distance * nbins), nbins-1);
            ++counter[binidx];
          }
          for(int j=1;j<nbins;++j) {
            counter[j] += counter[j-1];
          }
          const double lighting_pixels_ratio_lower = global_settings["lighting"]["lighting_pixels_ratio_lower"];
          const double lighting_pixels_ratio_upper = global_settings["lighting"]["lighting_pixels_ratio_upper"];
          double lighting_pixels_ratio = iters / (double)max_iters * lighting_pixels_ratio_upper + (1.0 - iters / (double) max_iters) * lighting_pixels_ratio_lower;
          const int cutoff_count = *std::lower_bound(counter.begin(), counter.end(), static_cast<int>(lighting_pixels_ratio*albedo_distances_i_sorted.size()));
          cout << "num constraints [before]: " << pixel_indices_i.size() << endl;
          pixel_indices_i.erase(pixel_indices_i.begin()+cutoff_count, pixel_indices_i.end());
          cout << "num constraints [after]: " << pixel_indices_i.size() << endl;

          QImage lighting_pixel_image(num_cols, num_rows, QImage::Format_ARGB32);
          lighting_pixel_image.fill(0);
          for(int j=0;j<pixel_indices_i.size();++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            lighting_pixel_image.setPixel(c, r, qRgb(255, 255, 255));
          }
          lighting_pixel_image.save( (results_path / fs::path("lighting_pixels" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );

          // ====================================================================
          // collect constraints from valid pixels
          // ====================================================================
          const int num_constraints = pixel_indices_i.size();

          MatrixXd normals_i(num_constraints, 3);
          MatrixXd albedos_i(num_constraints, 3);
          MatrixXd pixels_i(num_constraints, 3);

          for (int j = 0; j < num_constraints; ++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;

            cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(r, c);
            normals_i(j, 0) = pix[0];
            normals_i(j, 1) = pix[1];
            normals_i(j, 2) = pix[2];

            cv::Vec3d pix_albedo = albedos[i].at<cv::Vec3d>(r, c);
            albedos_i(j, 0) = pix_albedo[0];
            albedos_i(j, 1) = pix_albedo[1];
            albedos_i(j, 2) = pix_albedo[2];

            auto pix_i = bundle.image.pixel(c, r);
            pixels_i(j, 0) = qRed(pix_i) / 255.0;
            pixels_i(j, 1) = qGreen(pix_i) / 255.0;
            pixels_i(j, 2) = qBlue(pix_i) / 255.0;
          }

          // ====================================================================
          // assemble matrices
          // ====================================================================
          const int num_dof = global_settings["lighting"]["num_dof"];

          MatrixXd Y(num_constraints, num_dof);
          MatrixXd A;
          VectorXd b;
          VectorXd l_i;
          bool use_Lab_color = false;

          if(use_Lab_color) {
            A = MatrixXd(num_constraints, num_dof);
            b = VectorXd(num_constraints);

            for(int j=0;j<num_constraints;++j) {
              int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;

              cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(r, c);
              double nx, ny, nz;
              nx = pix[0], ny = pix[1], nz = pix[2];

              cv::Vec3d pix_albedo = albedos[i].at<cv::Vec3d>(r, c);
              double ar = pix_albedo[0], ag = pix_albedo[1], ab = pix_albedo[2];

              auto pix_i = bundle.image.pixel(c, r);
              double Ir = qRed(pix_i) / 255.0;
              double Ig = qGreen(pix_i) / 255.0;
              double Ib = qBlue(pix_i) / 255.0;

              Vector3d Lab = rgb2lab(Ir, Ig, Ib);
              Vector3d a_Lab = rgb2lab(ar, ag, ab);

              Y.row(j) = sphericalharmonics(nx, ny, nz).transpose();

              A.row(j) = Y.row(j) * a_Lab[0]; b(j) = Lab[0];
            }

            // Lighting regularization
            const double w_reg = 0.0001 * num_constraints;
            MatrixXd Afinal(num_constraints+9, 9);
            Afinal.topRows(num_constraints) = A;
            Afinal.bottomRows(9) = MatrixXd::Identity(9, 9) * w_reg;
            VectorXd bfinal(num_constraints+9);
            bfinal.topRows(num_constraints) = b;
            bfinal.bottomRows(9) = VectorXd::Zero(9);

            // Apply weights to
            Afinal.rightCols(5) *= second_order_weights;

            // ====================================================================
            // solve linear least squares
            // ====================================================================
            l_i = Afinal.colPivHouseholderQr().solve(bfinal);
          } else {
            A = MatrixXd(num_constraints * 3, num_dof);
            b = VectorXd(num_constraints * 3);

            #if 0

            Y.col(0) = VectorXd::Ones(num_constraints);
            Y.col(1) = normals_i.col(0);
            Y.col(2) = normals_i.col(1);
            Y.col(3) = normals_i.col(2);
            Y.col(4) = normals_i.col(0).cwiseProduct(normals_i.col(1));
            Y.col(5) = normals_i.col(0).cwiseProduct(normals_i.col(2));
            Y.col(6) = normals_i.col(1).cwiseProduct(normals_i.col(2));
            Y.col(7) = normals_i.col(0).cwiseProduct(normals_i.col(0)) - normals_i.col(1).cwiseProduct(normals_i.col(1));
            Y.col(8) = 3 * normals_i.col(2).cwiseProduct(normals_i.col(2)) - VectorXd::Ones(num_constraints);

            VectorXd a_vec(num_constraints * 3);
            a_vec.topRows(num_constraints) = albedos_i.col(0);
            a_vec.middleRows(num_constraints, num_constraints) = albedos_i.col(1);
            a_vec.bottomRows(num_constraints) = albedos_i.col(2);

            A.topRows(num_constraints) = Y;
            A.middleRows(num_constraints, num_constraints) = Y;
            A.bottomRows(num_constraints) = Y;
            for (int k = 0; k < num_dof; ++k) {
              A.col(k) = A.col(k).cwiseProduct(a_vec);
            }

            b.topRows(num_constraints) = pixels_i.col(0);
            b.middleRows(num_constraints, num_constraints) = pixels_i.col(1);
            b.bottomRows(num_constraints) = pixels_i.col(2);

            #else

            for(int j=0;j<num_constraints;++j) {
              int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;

              cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(r, c);
              double nx, ny, nz;
              nx = pix[0], ny = pix[1], nz = pix[2];

              cv::Vec3d pix_albedo = albedos[i].at<cv::Vec3d>(r, c);
              double ar = pix_albedo[0], ag = pix_albedo[1], ab = pix_albedo[2];

              auto pix_i = bundle.image.pixel(c, r);
              double Ir = qRed(pix_i) / 255.0;
              double Ig = qGreen(pix_i) / 255.0;
              double Ib = qBlue(pix_i) / 255.0;

              Y.row(j) = sphericalharmonics(nx, ny, nz).transpose();

              A.row(j*3) = Y.row(j) * ar; b(j*3) = Ir;
              A.row(j*3+1) = Y.row(j) * ag; b(j*3+1) = Ig;
              A.row(j*3+2) = Y.row(j) * ab; b(j*3+2) = Ib;
            }

            #endif

            // Lighting regularization
            const double w_reg = double(global_settings["lighting"]["w_reg"]) * num_constraints;
            MatrixXd Afinal(num_constraints*3+9, 9);
            Afinal.topRows(num_constraints*3) = A;
            Afinal.bottomRows(9) = MatrixXd::Identity(9, 9) * w_reg;
            VectorXd bfinal(num_constraints*3+9);
            bfinal.topRows(num_constraints*3) = b;
            bfinal.bottomRows(9) = VectorXd::Zero(9);

            // Apply weights to
            Afinal.rightCols(5) *= second_order_weights;

            // ====================================================================
            // solve linear least squares
            // ====================================================================
            l_i = Afinal.colPivHouseholderQr().solve(bfinal);
          }

          const double relax_factor = global_settings["lighting"]["relaxation"];
          lighting_coeffs[i] = (1.0 - relax_factor) * lighting_coeffs[i] + relax_factor * l_i;
          cout << l_i.transpose() << endl;

          // ====================================================================
          // [Optional] output result of estimated lighting
          // ====================================================================
          QImage image_with_lighting(num_cols, num_rows, QImage::Format_ARGB32);
          image_with_lighting.fill(0);
          for (int y = 0; y < normal_maps[i].rows; ++y) {
            for (int x = 0; x < normal_maps[i].cols; ++x) {
              float zval = zmaps[i].at<float>(y, x);
              if (zval < -1e5) continue;
              else {
                cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(y, x);
                double nx = pix[0], ny = pix[1], nz = pix[2];

                VectorXd Y_ij = sphericalharmonics(nx, ny, nz);

                double LdotY = l_i.transpose() * Y_ij;
                cv::Vec3d rho(0.5, 0.5, 0.5);
                rho *= 255.0 * LdotY;

                image_with_lighting.setPixel(x, y, qRgb(clamp<double>(rho[0], 0, 255),
                                                        clamp<double>(rho[1], 0, 255),
                                                        clamp<double>(rho[2], 0, 255)));
              }
            }
          }
          image_with_lighting.save( (results_path / fs::path("lighting_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );

          QImage lighting_coeffs_image(256, 256, QImage::Format_ARGB32);
          lighting_coeffs_image.fill(0);
          for(int r=0;r<256;++r) {
            double y = (127 - r)/127.0;
            for(int c=0;c<256;++c) {
              double x = (c - 127)/127.0;
              if(x*x + y*y <= 1.0) {
                // x = sin(theta)*sin(phi)
                // y = sin(theta)*cos(phi)
                // z = cos(theta)

                double z = sqrt(1 - x*x - y*y);
                VectorXd Y = sphericalharmonics(x, y, z);
                double LdotY = l_i.transpose() * Y;

                lighting_coeffs_image.setPixel(c, r, jet_color_QRgb(clamp<double>(LdotY / 1.5, 0.0, 1.0)));
              }
            }
          }
          lighting_coeffs_image.save( (results_path / fs::path("lighting_coeffs" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );
        }

        // [Shape from shading] step 2: fix depth and lighting, estimate albedo
        // @NOTE Construct the problem for whole image, then solve for valid pixels only
        {
          const double lambda2 = double(global_settings["albedo"]["lambda"]) / pow(2, (iters - 1));

          // ====================================================================
          // collect valid pixels
          // ====================================================================
          vector<glm::ivec2> pixel_indices_i;

          for (int y = 0; y < num_rows; ++y) {
            for (int x = 0; x < num_cols; ++x) {
              float zval = zmaps[i].at<float>(y, x);
              if (zval < -1e5) continue;
              else {
                pixel_indices_i.push_back(glm::ivec2(y, x));
              }
            }
          }

          // ====================================================================
          // collect constraints from valid pixels
          // ====================================================================
          const int num_constraints = pixel_indices_i.size();
          cout << num_constraints << endl;

          MatrixXd normals_i(num_constraints, 3);
          MatrixXd pixels_i(num_constraints, 3);

          QImage albedo_texture_image(num_cols, num_rows, QImage::Format_ARGB32);
          QImage albedo_normal_image(num_cols, num_rows, QImage::Format_ARGB32);
          for (int j = 0; j < num_constraints; ++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;

            cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(r, c);
            normals_i(j, 0) = pix[0];
            normals_i(j, 1) = pix[1];
            normals_i(j, 2) = pix[2];

            auto pix_i = bundle.image.pixel(c, r);
            pixels_i(j, 0) = qRed(pix_i) / 255.0;
            pixels_i(j, 1) = qGreen(pix_i) / 255.0;
            pixels_i(j, 2) = qBlue(pix_i) / 255.0;

            albedo_normal_image.setPixel(c, r, qRgb((pix[0]+1.0)*0.5*255,
                                                    (pix[1]+1.0)*0.5*255,
                                                    (pix[2]+1.0)*0.5*255));
            albedo_texture_image.setPixel(c, r, pix_i);
          }
          albedo_normal_image.save("albedo_normal_image.png");
          albedo_texture_image.save("albedo_texture_image.png");

          // ====================================================================
          // assemble matrices
          // ====================================================================
          const int num_dof = 9;  // use first order approximation
          MatrixXd Y(num_constraints, num_dof);

          #if 0

          Y.col(0) = VectorXd::Ones(num_constraints);
          Y.col(1) = normals_i.col(0);
          Y.col(2) = normals_i.col(1);
          Y.col(3) = normals_i.col(2);
          Y.col(4) = normals_i.col(0).cwiseProduct(normals_i.col(1));
          Y.col(5) = normals_i.col(0).cwiseProduct(normals_i.col(2));
          Y.col(6) = normals_i.col(1).cwiseProduct(normals_i.col(2));
          Y.col(7) = normals_i.col(0).cwiseProduct(normals_i.col(0)) - normals_i.col(1).cwiseProduct(normals_i.col(1));
          Y.col(8) = 3 * normals_i.col(2).cwiseProduct(normals_i.col(2)) - VectorXd::Ones(num_constraints);

          #else
          for(int j=0;j<num_constraints;++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;

            cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(r, c);
            double nx, ny, nz;
            nx = pix[0], ny = pix[1], nz = pix[2];

            Y.row(j) = sphericalharmonics(nx, ny, nz).transpose();
          }
          #endif

          VectorXd LdotY = Y * lighting_coeffs[i];

          vector<bool> is_valid_pixel(num_rows*num_cols, false);
          vector<int> pixel_index_map(num_rows*num_cols, -1);
          for(int j=0;j<num_constraints;++j) {
            int pidx = pixel_indices_i[j].x * num_cols + pixel_indices_i[j].y;
            is_valid_pixel[pidx] = true;
            pixel_index_map[pidx] = j;
          }

          PhGUtils::message("Assembling matrices ...");
          vector<Tripletd> A_coeffs;
          A_coeffs.reserve(num_constraints + num_constraints * 25);
          for(int j=0;j<num_constraints;++j) {
            A_coeffs.push_back(Tripletd(j, j, LdotY(j)));
          }

#if 1
          for(int j=0;j<LoG_coeffs.size();++j) {
              auto& item_j = LoG_coeffs[j];
              if(is_valid_pixel[item_j.row()] && is_valid_pixel[item_j.col()]) {
                int new_i = pixel_index_map[item_j.row()] + num_constraints;
                int new_j = pixel_index_map[item_j.col()];
                A_coeffs.push_back(Tripletd(new_i, new_j, item_j.value() * lambda2));
              }
          }
#else
          for(int j=0;j<num_constraints;++j) {
            int r0 = pixel_indices_i[j].x, c0 = pixel_indices_i[j].y;
            int pidx = r0 * num_cols + c0;

            for(int kr = -kLoG, r=0; kr <= kLoG; ++kr, ++r) {
              for(int kc = -kLoG, c=0; kc <= kLoG; ++kc, ++c) {
                int qidx = (r0 + kr) * num_cols + (c0 + kc);
                if(qidx < 0 || qidx >= num_rows * num_cols) continue;
                if(is_valid_pixel[qidx]) {
                  A_coeffs.push_back(Tripletd(j + num_constraints,
                                              pixel_index_map[qidx],
                                              LoG(r, c) * lambda2));
                }
              }
            }
          }
#endif

#if 1
          ofstream fout("A.txt");
          for(auto ttt : A_coeffs) {
            fout << ttt.row() << ' ' << ttt.col() << ' ' << ttt.value() << '\n';
          }
          fout.close();
#endif

          Eigen::SparseMatrix<double> A(num_constraints * 2, num_constraints);
          A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());
          //A.makeCompressed();

          PhGUtils::message("A assembled ...");

          // fill each channel individually
          MatrixXd B(num_constraints*2, 3);
          for(int j=0;j<num_constraints;++j) {
            int pidx = pixel_indices_i[j].x * num_cols + pixel_indices_i[j].y;
            B.row(j) = pixels_i.row(j);
            B.row(j + num_constraints) = (albedo_ref_LoG_i.row(pidx) * lambda2).eval();
          }

          PhGUtils::message("done.");

          // ====================================================================
          // solve linear least squares
          // ====================================================================
          //const double epsilon = 0.0;
          //Eigen::SparseMatrix<double> eye(num_constraints, num_constraints);
          //for(int j=0;j<num_constraints;++j) eye.insert(j, j) = epsilon;

          cout << "Computing AtA ..." << endl;
          Eigen::SparseMatrix<double> AtA = A.transpose() * A;
          //AtA.makeCompressed();

          cout << AtA.rows() << 'x' << AtA.cols() << endl;
          cout << AtA.nonZeros() << endl;

          //AtA += eye;

          // AtA is symmetric, so it is okay to use it as column major?
          CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
          solver.compute(AtA);
          if(solver.info()!=Success) {
            cout << "Failed to decompose matrix A." << endl;
            exit(-1);
          }

          MatrixXd rho(num_rows*num_cols, 3);
          for(int cidx=0;cidx<3;++cidx) {
            VectorXd Atb = A.transpose() * B.col(cidx);
            VectorXd x = solver.solve(Atb);

            if(solver.info()!=Success) {
              cout << "Failed to solve A\\b." << endl;
              exit(-1);
            }

            for(int j=0;j<num_constraints;++j) {
              int pidx = pixel_indices_i[j].x * num_cols + pixel_indices_i[j].y;
              rho(pidx, cidx) = x(j);
            }
          }

          // update albedo
          for(int j=0;j<num_constraints;++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            int pidx = r * num_cols + c;
            albedos[i].at<cv::Vec3d>(r, c) = cv::Vec3d(rho(pidx, 0), rho(pidx, 1), rho(pidx, 2));
          }

          // ====================================================================
          // [Optional] output result of estimated albedo
          // ====================================================================
          QImage image_with_albedo(num_cols, num_rows, QImage::Format_ARGB32);
          QImage image_with_albedo_lighting(num_cols, num_rows, QImage::Format_ARGB32);
          image_with_albedo.fill(0);
          image_with_albedo_lighting.fill(0);
          for (int y = 0; y < num_rows; ++y) {
            for (int x = 0; x < num_cols; ++x) {
              cv::Vec3d pix_albedo = albedos[i].at<cv::Vec3d>(y, x);
              if (pix_albedo[0] == 0 && pix_albedo[1] == 0 && pix_albedo[2] == 0) {
                continue;
              }
              else {
                cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(y, x);

                double nx = pix[0], ny = pix[1], nz = pix[2];
                VectorXd Y_ij = sphericalharmonics(nx, ny, nz);

                double LdotY = lighting_coeffs[i].transpose() * Y_ij;
                cv::Vec3d pix_val = cv::Vec3d(rho(y*num_cols+x, 0), rho(y*num_cols+x, 1), rho(y*num_cols+x, 2));
                pix_val *= 255.0;

                image_with_albedo.setPixel(x, y, qRgb(clamp<double>(pix_val[0], 0, 255),
                                                      clamp<double>(pix_val[1], 0, 255),
                                                      clamp<double>(pix_val[2], 0, 255)));

                pix_val *= LdotY;
                image_with_albedo_lighting.setPixel(x, y, qRgb(clamp<double>(pix_val[0], 0, 255),
                                                               clamp<double>(pix_val[1], 0, 255),
                                                               clamp<double>(pix_val[2], 0, 255)));
              }
            }
          }
          image_with_albedo.save( (results_path / fs::path("albedo_opt_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str());
          image_with_albedo_lighting.save( (results_path / fs::path("albedo_opt_lighting_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str());
        }

        // [Shape from shading] step 3: fix albedo and lighting, estimate normal map
        // @NOTE Construct the problem for whole image, then solve for valid pixels only
        const int iters_depth = global_settings["depth"]["num_iters"];
        for(int iii=0;iii<iters_depth;++iii){

          // ====================================================================
          // collect valid pixels
          // ====================================================================
          vector<glm::ivec2> pixel_indices_i;
          if (valid_depth_pixels[i].empty()) {
            cv::Mat boundary_pixel_image(num_rows, num_cols, CV_8U);
            cv::Mat valid_pixel_image(num_rows, num_cols, CV_8U);
            for (int y = 0; y < normal_maps[i].rows; ++y) {
              for (int x = 0; x < normal_maps[i].cols; ++x) {
                boundary_pixel_image.at<unsigned char>(y, x) = 0;
                valid_pixel_image.at<unsigned char>(y, x) = 0;

                float zval = zmaps[i].at<float>(y, x);
                if (zval < -1e5) continue;
                else {

                  bool flag = false;//face_boundary_indices.count(face_indices_maps[i][y*num_cols+x]);
                  flag |= zmaps[i].at<float>(y-1, x) < -1e5;
                  flag |= zmaps[i].at<float>(y+1, x) < -1e5;
                  flag |= zmaps[i].at<float>(y, x-1) < -1e5;
                  flag |= zmaps[i].at<float>(y, x+1) < -1e5;

                  if(flag) {
                    boundary_pixel_image.at<unsigned char>(y, x) = 255;
                    is_boundary[y*num_cols+x] = true;
                  } else {
                    pixel_indices_i.push_back(glm::ivec2(y, x));
                    valid_pixel_image.at<unsigned char>(y, x) = 255;
                  }
                }
              }
            }

            // Filter out edges
            for(int r=0;r<num_rows;++r) {
              for(int c=0;c<num_cols;++c) {
                cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(r, c);
                double nx = pix[0], ny = pix[1], nz = pix[2];

                cv::Vec3d pix_u = normal_maps[i].at<cv::Vec3d>(r-1, c);
                double nx_u, ny_u, nz_u;
                nx_u = pix_u[0]; ny_u = pix_u[1]; nz_u = pix_u[2];

                cv::Vec3d pix_l = normal_maps[i].at<cv::Vec3d>(r, c-1);
                double nx_l, ny_l, nz_l;
                nx_l = pix_l[0]; ny_l = pix_l[1]; nz_l = pix_l[2];

                auto round_off = [](double val, double eps) {
                  if(fabs(val) < eps) {
                    return val>0?eps:-eps;
                  } else return val;
                };
                double nxnz = nx / round_off(nz, 1e-16);
                double nynz = ny / round_off(nz, 1e-16);
                double nxnz_u = nx_u / round_off(nz_u, 1e-16);
                double nynz_l = ny_l / round_off(nz_l, 1e-16);
                double integrability_val = clamp<double>(fabs((nxnz_u - nxnz) - (nynz - nynz_l)) * 255.0, 0, 255);

                const double integrability_threshold = global_settings["depth"]["integrability_threshold"];
                if(integrability_val > integrability_threshold) {
                  boundary_pixel_image.at<unsigned char>(r, c) = 255;
                  is_boundary[r*num_cols+c] = true;
                }
              }
            }

            cv::imwrite( (results_path / fs::path("boundary_pixels_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str(), boundary_pixel_image );
            cv::imwrite( (results_path / fs::path("valid_pixels_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str(), valid_pixel_image );

            valid_depth_pixels[i] = pixel_indices_i;
          } else {
            pixel_indices_i = valid_depth_pixels[i];
          }


          // ====================================================================
          // collect constraints from valid pixels
          // ====================================================================
          const int num_constraints = pixel_indices_i.size();
          cout << num_constraints << endl;

          MatrixXd albedos_i(num_constraints, 3);
          MatrixXd pixels_i(num_constraints, 3);

          for (int j = 0; j < num_constraints; ++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;

            cv::Vec3d pix_albedo = albedos[i].at<cv::Vec3d>(r, c);
            albedos_i(j, 0) = pix_albedo[0];
            albedos_i(j, 1) = pix_albedo[1];
            albedos_i(j, 2) = pix_albedo[2];

            auto pix_i = bundle.image.pixel(c, r);
            pixels_i(j, 0) = qRed(pix_i) / 255.0;
            pixels_i(j, 1) = qGreen(pix_i) / 255.0;
            pixels_i(j, 2) = qBlue(pix_i) / 255.0;
          }

          // ====================================================================
          // assemble cost functions
          // ====================================================================

          // create a valid pixel map first
          vector<bool> is_valid_pixel(num_rows*num_cols, false);
          vector<int> pixel_index_map(num_rows*num_cols, -1);
          for(int j=0;j<num_constraints;++j) {
            int pidx = pixel_indices_i[j].x * num_cols + pixel_indices_i[j].y;
            is_valid_pixel[pidx] = true;
            pixel_index_map[pidx] = j;
          }

#define USE_THETA_PHI 0
#if USE_THETA_PHI
          ceres::Problem problem;
          VectorXd theta(num_constraints), phi(num_constraints);

          // initialize theta and phi
          for (int j = 0; j < num_constraints; ++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;

            cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(r, c);
            const double nx = pix[0], ny = pix[1], nz = pix[2];
            tie(theta(j), phi(j)) = normal2sphericalcoords(nx, ny, nz);
          }
          VectorXd theta0 = theta, phi0 = phi;

          if(iters == 1 && iii == 0){
            QImage theta_image(num_cols, num_rows, QImage::Format_ARGB32);
            QImage phi_image(num_cols, num_rows, QImage::Format_ARGB32);
            theta_image.fill(0);
            phi_image.fill(0);
            for (int y = 0; y < num_rows; ++y) {
              for (int x = 0; x < num_cols; ++x) {
                cv::Vec3d pix_albedo = albedos[i].at<cv::Vec3d>(y, x);
                float zval = zmaps[i].at<float>(y, x);
                if (zval < -1e5) {
                  continue;
                }
                else {
                  cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(y, x);
                  double nx = pix[0], ny = pix[1], nz = pix[2];
                  double theta, phi;
                  tie(theta, phi) = normal2sphericalcoords(nx, ny, nz);

                  // [0, pi], [-pi, pi]
                  double theta_val, phi_val;
                  tie(theta_val, phi_val) = normal2sphericalcoords(nx, ny, nz);

                  // Add a very small purturbation
                  const double small_value = 5e-3;
                  theta_val += rand()%2?small_value:-small_value;
                  phi_val += rand()%2?small_value:-small_value;

                  double theta_ratio = clamp<double>(theta_val * 2.0 / 3.1415926535897, 0, 1);
                  theta_image.setPixel(x, y, jet_color_QRgb(theta_ratio));

                  double phi_ratio = clamp<double>((phi_val + 3.1415926535897)*0.5/3.1415926535897, 0, 1);
                  phi_image.setPixel(x, y, jet_color_QRgb(phi_ratio));
                }
              }
            }
            theta_image.save( (results_path / fs::path("theta_" + std::to_string(i) + "_" + std::to_string(0) + ".png")).string().c_str() );
            phi_image.save( (results_path / fs::path("phi_" + std::to_string(i) + "_" + std::to_string(0) + ".png")).string().c_str() );
          }

          const double w_data = global_settings["depth"]["w_data"];
          const double w_reg = global_settings["depth"]["w_reg"];
          const double w_integrability = global_settings["depth"]["w_int"];
          const double w_smoothness = global_settings["depth"]["w_smooth"];

          #define USE_ANALYTIC_COST_FUNCTIONS 1
          PhGUtils::message("Assembling cost functions ...");
          {
            boost::timer::auto_cpu_timer timer_solve(
              "[Shape from shading] Cost function assemble time = %w seconds.\n");

            // data term
            for(int j = 0; j < num_constraints; ++j) {
              #if USE_ANALYTIC_COST_FUNCTIONS
              ceres::CostFunction *cost_function =
                new NormalMapDataTerm_analytic(pixels_i(j, 0), pixels_i(j, 1), pixels_i(j, 2),
                                               albedos_i(j, 0), albedos_i(j, 1), albedos_i(j, 2),
                                               lighting_coeffs[i]);
              #else
              ceres::DynamicNumericDiffCostFunction<NormalMapDataTerm> *cost_function =
                new ceres::DynamicNumericDiffCostFunction<NormalMapDataTerm>(
                  new NormalMapDataTerm(pixels_i(j, 0), pixels_i(j, 1), pixels_i(j, 2),
                                        albedos_i(j, 0), albedos_i(j, 1), albedos_i(j, 2),
                                        lighting_coeffs[i], w_data)
                );

              cost_function->AddParameterBlock(1);
              cost_function->AddParameterBlock(1);
              cost_function->SetNumResiduals(3);
              #endif
              problem.AddResidualBlock(cost_function, NULL, theta.data()+j, phi.data()+j);
            }

            //QImage bad_term_image(num_cols, num_rows, QImage::Format_ARGB32);
            //bad_term_image.fill(0);
            int bad_term_count = 0;

            // integrability term
            for(int j = 0; j < num_constraints; ++j) {
              int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
              int pidx = r * num_cols + c;
              if(c < 1 || r < 1 || c >= num_cols || r >= num_rows) continue;

              int left_idx = pidx - 1;
              int right_idx = pidx + 1;
              int up_idx = pidx - num_cols;
              int down_idx = pidx + num_cols;

              if(is_valid_pixel[left_idx] && is_valid_pixel[up_idx]) {
                double dx = 1, dy = 1;
                /*
                cv::Vec3d depth_ij = depth_maps[i].at<cv::Vec3d>(r, c);
                cv::Vec3d depth_ij_l = depth_maps[i].at<cv::Vec3d>(r, c-1);
                cv::Vec3d depth_ij_u = depth_maps[i].at<cv::Vec3d>(r-1, c);
                dx = depth_ij[0] - depth_ij_l[0];
                dy = depth_ij[1] - depth_ij_u[0];
                */
                bool boundary_pixel = is_boundary[pidx];

                #if USE_ANALYTIC_COST_FUNCTIONS
                ceres::CostFunction *cost_function = new NormalMapIntegrabilityTerm_analytic(dx, dy, boundary_pixel?0:w_integrability);
                #else
                ceres::DynamicNumericDiffCostFunction<NormalMapIntegrabilityTerm> *cost_function =
                  new ceres::DynamicNumericDiffCostFunction<NormalMapIntegrabilityTerm>(
                    new NormalMapIntegrabilityTerm(dx, dy, boundary_pixel?0:w_integrability)
                  );

                for(int param_i = 0; param_i < 6; ++param_i) cost_function->AddParameterBlock(1);
                cost_function->SetNumResiduals(1);
                #endif

                problem.AddResidualBlock(cost_function, NULL,
                                         theta.data()+j, phi.data()+j,
                                         theta.data()+pixel_index_map[left_idx], phi.data()+pixel_index_map[left_idx],
                                         theta.data()+pixel_index_map[up_idx], phi.data()+pixel_index_map[up_idx]);
              }
            }
            //bad_term_image.save("bad_term.png");

            // smoothness term
            #if 0
            for(int j = 0; j < num_constraints; ++j) {
              int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
              int pidx = r * num_cols + c;
              if(c < 1 || r < 1 || c >= num_cols || r >= num_rows) continue;

              int left_idx = pidx - 1;
              int up_idx = pidx - num_cols;

              if(is_valid_pixel[left_idx] && is_valid_pixel[up_idx]) {
                double dx = 1, dy = 1;
                /*
                cv::Vec3d depth_ij = depth_maps[i].at<cv::Vec3d>(r, c);
                cv::Vec3d depth_ij_l = depth_maps[i].at<cv::Vec3d>(r, c-1);
                cv::Vec3d depth_ij_u = depth_maps[i].at<cv::Vec3d>(r-1, c);
                dx = depth_ij[0] - depth_ij_l[0];
                dy = depth_ij[1] - depth_ij_u[0];
                */
                bool boundary_pixel = is_boundary[pidx];

                #if 0//USE_ANALYTIC_COST_FUNCTIONS
                ceres::CostFunction *cost_function = new NormalMapIntegrabilityTerm_analytic(dx, dy, boundary_pixel?0:w_integrability);
                #else
                ceres::DynamicNumericDiffCostFunction<NormalMapSmoothnessTerm> *cost_function =
                  new ceres::DynamicNumericDiffCostFunction<NormalMapSmoothnessTerm>(
                    new NormalMapSmoothnessTerm(dx, dy, boundary_pixel?0:w_smoothness)
                  );

                for(int param_i = 0; param_i < 6; ++param_i) cost_function->AddParameterBlock(1);
                cost_function->SetNumResiduals(4);
                #endif

                problem.AddResidualBlock(cost_function, NULL,
                                         theta.data()+j, phi.data()+j,
                                         theta.data()+pixel_index_map[left_idx], phi.data()+pixel_index_map[left_idx],
                                         theta.data()+pixel_index_map[up_idx], phi.data()+pixel_index_map[up_idx]);
              }
            }
            #endif

            // regularization term
            #if 0
            for(int j = 0; j < num_constraints; ++j) {
              int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
              int pidx = r * num_cols + c;

              vector<pair<int, double>> reginfo;
              for(auto p : LoG_coeffs_perpixel[pidx]) {
                if(is_valid_pixel[p.first]) reginfo.push_back(p);
              }
              if(reginfo.empty()) continue;
              else {
                Vector3d normal_LoG_j(normal_map_ref_LoG_i(pidx, 0), normal_map_ref_LoG_i(pidx, 1), normal_map_ref_LoG_i(pidx, 2));
                #if USE_ANALYTIC_COST_FUNCTIONS
                ceres::CostFunction *cost_function =
                  new NormalMapRegularizationTerm_analytic(reginfo, normal_LoG_j, w_reg);
                #else
                ceres::DynamicNumericDiffCostFunction<NormalMapRegularizationTerm> *cost_function =
                  new ceres::DynamicNumericDiffCostFunction<NormalMapRegularizationTerm>(
                    new NormalMapRegularizationTerm(reginfo, normal_LoG_j, w_reg)
                  );

                cost_function->SetNumResiduals(3);
                for(auto ri : reginfo) {
                  cost_function->AddParameterBlock(1);
                  cost_function->AddParameterBlock(1);
                }
                #endif

                vector<double*> params_ptrs;
                for(auto ri : reginfo) {
                  params_ptrs.push_back(theta.data()+pixel_index_map[ri.first]);
                  params_ptrs.push_back(phi.data()+pixel_index_map[ri.first]);
                }
                problem.AddResidualBlock(cost_function, NULL, params_ptrs);
              }
            }
            #endif
          }

          PhGUtils::message("done.");

          PhGUtils::message("Solving non-linear least squares ...");
          {
            boost::timer::auto_cpu_timer timer_solve(
              "[Shape from shading] Problem solve time = %w seconds.\n");
            ceres::Solver::Options options;
            options.max_num_iterations = 3;
            options.num_threads = 8;
            options.num_linear_solver_threads = 8;

            options.initial_trust_region_radius = 10.0;

            //options.min_trust_region_radius = 1.0;
            //options.max_trust_region_radius = 1.0;

            options.min_lm_diagonal = 1.0;
            options.max_lm_diagonal = 1.0;

            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            cout << summary.BriefReport() << endl;
          }

#define THRESHOLD_NORMAL_CHANGE 1
#if THRESHOLD_NORMAL_CHANGE
          VectorXd dtheta = theta - theta0;
          VectorXd dphi = phi - phi0;

          const double dlimit_val = 3.1415926535897 * 0.25;

          for(int j=0;j<num_constraints;++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            int pidx = r * num_cols + c;
            if(fabs(dtheta(j)) > dlimit_val) is_boundary[pidx] = true;
            if(fabs(dphi(j)) > dlimit_val) is_boundary[pidx] = true;
          }

          VectorXd dlimit = VectorXd::Ones(num_constraints) * dlimit_val;
          dtheta = dtheta.cwiseMax(-dlimit); dtheta = dtheta.cwiseMin(dlimit);
          dphi = dphi.cwiseMax(-dlimit); dphi = dphi.cwiseMin(dlimit);

          const double angle_update_relax_factor = 0.5;
          theta = theta0 + dtheta * angle_update_relax_factor;
          phi = phi0 + dphi * angle_update_relax_factor;
#endif  // THRESHOLD_NORMAL_CHANGE

          // update normal map
          cv::Mat normal_map_blurred(num_rows, num_cols, CV_32FC3);
          for(int j=0;j<num_constraints;++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            int pidx = r * num_cols + c;

            double nx, ny, nz;
            tie(nx, ny, nz) = sphericalcoords2normal<double>(theta(j), phi(j));
            //nz = max(0.0, nz);

            normal_map_blurred.at<cv::Vec3f>(r, c) = cv::Vec3f(nx, ny, nz);
          }

          #if 0
          // @FIXME fill the holes
          if(iters < max_iters - 1) {
            cv::medianBlur(normal_map_blurred, normal_map_blurred, 3);
          }
          #endif

          for(int j=0;j<num_constraints;++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            int pidx = r * num_cols + c;
            cv::Vec3f pix_j = normal_map_blurred.at<cv::Vec3f>(r, c);
            normal_maps[i].at<cv::Vec3d>(r, c) = cv::Vec3d(pix_j[0], pix_j[1], pix_j[2]);
          }

#else  // USE_THETA_PHI
          {
            // Optimize for depth directly
            ceres::Problem problem;
            VectorXd z_value(num_constraints);

            // initialize nx and ny
            double mean_z_val = 0;
            for (int j = 0; j < num_constraints; ++j) {
              int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;

              z_value(j) = zmaps[i].at<float>(r, c);
              mean_z_val += z_value(j);
            }
            mean_z_val /= num_constraints;
            cout << "mean z = " << mean_z_val << endl;

            const double w_data = global_settings["depth"]["w_data"];
            const double w_reg = global_settings["depth"]["w_reg"];
            const double w_integrability = global_settings["depth"]["w_int"];
            const double w_smoothness = global_settings["depth"]["w_smooth"];

            PhGUtils::message("Assembling cost functions ...");
            {
              boost::timer::auto_cpu_timer timer_solve(
                "[Shape from shading] Cost function assemble time = %w seconds.\n");

              // data term
              double mean_dz_val = 0; int mean_dz_count = 0;
              for(int j = 0; j < num_constraints; ++j) {
                int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
                int pidx = r * num_cols + c;
                if(c < 1 || r < 1 || c >= num_cols || r >= num_rows) continue;

                int left_idx = pidx - 1;
                int right_idx = pidx + 1;
                int up_idx = pidx - num_cols;
                int down_idx = pidx + num_cols;

                if(is_valid_pixel[left_idx] && is_valid_pixel[up_idx]) {
                  mean_dz_val += fabs(z_value(j) - z_value(pixel_index_map[left_idx]));
                  mean_dz_val += fabs(z_value(j) - z_value(pixel_index_map[up_idx]));
                  ++mean_dz_count;

                  cv::Vec3d depth_ij = depth_maps[i].at<cv::Vec3d>(r, c);
                  cv::Vec3d depth_ij_l = depth_maps[i].at<cv::Vec3d>(r, c-1);
                  cv::Vec3d depth_ij_u = depth_maps[i].at<cv::Vec3d>(r-1, c);
                  double dx = -fabs(depth_ij[0] - depth_ij_l[0]);
                  double dy = -fabs(depth_ij_u[1] - depth_ij[1]);

                  ceres::DynamicNumericDiffCostFunction<DepthMapDataTerm> *cost_function =
                    new ceres::DynamicNumericDiffCostFunction<DepthMapDataTerm>(
                      new DepthMapDataTerm(
                        pixels_i(j, 0), pixels_i(j, 1), pixels_i(j, 2),
                        albedos_i(j, 0), albedos_i(j, 1), albedos_i(j, 2),
                        lighting_coeffs[i], dx, dy)
                    );
                  cost_function->AddParameterBlock(1);
                  cost_function->AddParameterBlock(1);
                  cost_function->AddParameterBlock(1);
                  cost_function->SetNumResiduals(3);
                  problem.AddResidualBlock(cost_function, NULL,
                                           z_value.data()+j,
                                           z_value.data()+pixel_index_map[left_idx],
                                           z_value.data()+pixel_index_map[up_idx]);
                }
              }

              mean_dz_val /= mean_dz_count;
              cout << "mean dz = " << mean_dz_val << endl;

              // integrability term
              for(int j = 0; j < num_constraints; ++j) {
                int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
                int pidx = r * num_cols + c;
                if(c < 1 || r < 1 || c >= num_cols || r >= num_rows) continue;

                int left_idx = pidx - 1;
                int right_idx = pidx + 1;
                int up_idx = pidx - num_cols;
                int down_idx = pidx + num_cols;
                int up_left_idx = pidx - num_cols - 1;
                int up_up_idx = pidx - num_cols - num_cols;
                int left_left_idx = pidx - 2;

                if(is_valid_pixel[left_idx] && is_valid_pixel[up_idx]
                   && is_valid_pixel[up_left_idx]
                   && is_valid_pixel[up_up_idx] && is_valid_pixel[left_left_idx]) {

                  cv::Vec3d depth_ij = depth_maps[i].at<cv::Vec3d>(r, c);
                  cv::Vec3d depth_ij_l = depth_maps[i].at<cv::Vec3d>(r, c-1);
                  cv::Vec3d depth_ij_u = depth_maps[i].at<cv::Vec3d>(r-1, c);
                  double dx = -fabs(depth_ij[0] - depth_ij_l[0]);
                  double dy = -fabs(depth_ij_u[1] - depth_ij[1]);

                  bool boundary_pixel = is_boundary[pidx];

                  ceres::DynamicNumericDiffCostFunction<DepthMapIntegrabilityTerm>
                    *cost_function =
                      new ceres::DynamicNumericDiffCostFunction<DepthMapIntegrabilityTerm>(
                        new DepthMapIntegrabilityTerm(dx, dy, boundary_pixel?0:w_integrability));
                  cost_function->AddParameterBlock(1);
                  cost_function->AddParameterBlock(1);
                  cost_function->AddParameterBlock(1);
                  cost_function->AddParameterBlock(1);
                  cost_function->AddParameterBlock(1);
                  cost_function->AddParameterBlock(1);
                  cost_function->SetNumResiduals(1);

                  problem.AddResidualBlock(cost_function, NULL,
                                           z_value.data()+j,
                                           z_value.data()+pixel_index_map[left_idx],
                                           z_value.data()+pixel_index_map[up_idx],
                                           z_value.data()+pixel_index_map[up_left_idx],
                                           z_value.data()+pixel_index_map[up_up_idx],
                                           z_value.data()+pixel_index_map[left_left_idx]);
                }
              }

              // regularization term
              for(int j = 0; j < num_constraints; ++j) {
                int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
                int pidx = r * num_cols + c;

                vector<pair<int, double>> reginfo;
                for(auto p : LoG_coeffs_perpixel[pidx]) {
                  if(is_valid_pixel[p.first]) reginfo.push_back(p);
                }
                if(reginfo.empty()) continue;
                else {
                  double z_LoG_j = normal_map_ref_LoG_i(pidx);

                  ceres::DynamicNumericDiffCostFunction<DepthMapRegularizationTerm> *cost_function =
                    new ceres::DynamicNumericDiffCostFunction<DepthMapRegularizationTerm>(
                      new DepthMapRegularizationTerm(reginfo, z_LoG_j, w_reg)
                    );

                  cost_function->SetNumResiduals(1);
                  for(auto ri : reginfo) {
                    cost_function->AddParameterBlock(1);
                  }

                  vector<double*> params_ptrs;
                  for(auto ri : reginfo) {
                    params_ptrs.push_back(z_value.data()+pixel_index_map[ri.first]);
                  }
                  problem.AddResidualBlock(cost_function, NULL, params_ptrs);
                }
              }
          }
          PhGUtils::message("done.");

          PhGUtils::message("Solving non-linear least squares ...");
          {
            boost::timer::auto_cpu_timer timer_solve(
              "[Shape from shading] Problem solve time = %w seconds.\n");
            ceres::Solver::Options options;
            options.max_num_iterations = global_settings["depth"]["optimization"]["max_iters"];
            options.num_threads = 8;
            options.num_linear_solver_threads = 8;

            options.initial_trust_region_radius = global_settings["depth"]["optimization"]["init_tr_radius"];

            //options.min_trust_region_radius = 1.0;
            //options.max_trust_region_radius = 1.0;

            options.min_lm_diagonal = 1.0;
            options.max_lm_diagonal = 1.0;

            options.minimizer_progress_to_stdout = true;
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);
            cout << summary.BriefReport() << endl;
          }

          // update normal map
          for(int j=0;j<num_constraints;++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            zmaps[i].at<float>(r, c) = z_value(j);
          }

          for(int j=0;j<num_constraints;++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            int pidx = r * num_cols + c;
            if (is_valid_pixel[pidx-1] && is_valid_pixel[pidx-num_cols]) {

              cv::Vec3d depth_ij = depth_maps[i].at<cv::Vec3d>(r, c);
              cv::Vec3d depth_ij_l = depth_maps[i].at<cv::Vec3d>(r, c-1);
              cv::Vec3d depth_ij_u = depth_maps[i].at<cv::Vec3d>(r-1, c);
              double dx = -fabs(depth_ij[0] - depth_ij_l[0]);
              double dy = -fabs(depth_ij_u[1] - depth_ij[1]);

              double p = (zmaps[i].at<float>(r, c) - zmaps[i].at<float>(r, c-1)) / dx;
              double q = (zmaps[i].at<float>(r-1, c) - zmaps[i].at<float>(r, c)) / dy;

              double N = p * p + q * q + 1;
              normal_maps[i].at<cv::Vec3d>(r, c) = cv::Vec3d(p/N, q/N, 1/N);
            }
          }
        }
#endif  // USE_THETA_PHI

          PhGUtils::message("done.");

          // ====================================================================
          // [Optional] output result of estimated lighting
          // ====================================================================
          QImage normal_image(num_cols, num_rows, QImage::Format_ARGB32);
          QImage image_with_albedo_normal_lighting(num_cols, num_rows, QImage::Format_ARGB32);
          QImage image_error(num_cols, num_rows, QImage::Format_ARGB32);
          QImage integrability_image(num_cols, num_rows, QImage::Format_ARGB32);
          QImage smoothness_image(num_cols, num_rows, QImage::Format_ARGB32);
          QImage theta_image(num_cols, num_rows, QImage::Format_ARGB32);
          QImage phi_image(num_cols, num_rows, QImage::Format_ARGB32);

          normal_image.fill(0);
          image_with_albedo_normal_lighting.fill(0);
          image_error.fill(0);
          integrability_image.fill(0);
          smoothness_image.fill(0);
          theta_image.fill(0);
          phi_image.fill(0);

          const int num_dof = 9;
          for (int y = 0; y < num_rows; ++y) {
            for (int x = 0; x < num_cols; ++x) {
              cv::Vec3d pix_albedo = albedos[i].at<cv::Vec3d>(y, x);
              float zval = zmaps[i].at<float>(y, x);
              if (zval < -1e5) {
                continue;
              }
              else {
                cv::Vec3d pix = normal_maps[i].at<cv::Vec3d>(y, x);
                double nx = pix[0], ny = pix[1], nz = pix[2];
                double theta, phi;
                tie(theta, phi) = normal2sphericalcoords(nx, ny, nz);

                VectorXd Y_ij = sphericalharmonics(nx, ny, nz);

                normal_image.setPixel(x, y, qRgb((nx+1)*0.5*255.0,
                                                 (ny+1)*0.5*255.0,
                                                 (nz+1)*0.5*255.0));

                cv::Vec3d rho = pix_albedo;
                double LdotY = lighting_coeffs[i].transpose() * Y_ij;
                cv::Vec3d pix_val = cv::Vec3d(rho(0), rho(1), rho(2));
                pix_val *= 255.0 * LdotY;

                image_with_albedo_normal_lighting.setPixel(x, y, qRgb(clamp<double>(pix_val[0], 0, 255),
                                                                      clamp<double>(pix_val[1], 0, 255),
                                                                      clamp<double>(pix_val[2], 0, 255)));

                auto pix_ij = bundle.image.pixel(x, y);
                cv::Vec3d pix_diff(fabs(pix_val(0) - qRed(pix_ij)),
                                   fabs(pix_val(1) - qGreen(pix_ij)),
                                   fabs(pix_val(2) - qBlue(pix_ij)));
                image_error.setPixel(x, y, qRgb(pix_diff(0), pix_diff(1), pix_diff(2)));

                cv::Vec3d pix_u = normal_maps[i].at<cv::Vec3d>(y-1, x);
                double nx_u, ny_u, nz_u;
                nx_u = pix_u[0]; ny_u = pix_u[1]; nz_u = pix_u[2];
                double theta_u, phi_u;
                tie(theta_u, phi_u) = normal2sphericalcoords(nx_u, ny_u, nz_u);

                cv::Vec3d pix_l = normal_maps[i].at<cv::Vec3d>(y, x-1);
                double nx_l, ny_l, nz_l;
                nx_l = pix_l[0]; ny_l = pix_l[1]; nz_l = pix_l[2];
                double theta_l, phi_l;
                tie(theta_l, phi_l) = normal2sphericalcoords(nx_l, ny_l, nz_l);

                auto round_off = [](double val, double eps) {
                  if(fabs(val) < eps) {
                    return val>0?eps:-eps;
                  } else return val;
                };
                double nxnz = nx / round_off(nz, 1e-16);
                double nynz = ny / round_off(nz, 1e-16);
                double nxnz_u = nx_u / round_off(nz_u, 1e-16);
                double nynz_l = ny_l / round_off(nz_l, 1e-16);
                double integrability_val = fabs((nxnz_u - nxnz) - (nynz - nynz_l));
                integrability_image.setPixel(x, y, jet_color_QRgb(integrability_val * 10.0));

                double smoothness_val = (theta - theta_u) * (theta - theta_u) + (theta - theta_l) * (theta - theta_l)
                                      + (phi - phi_u) * (phi - phi_u) + (phi - phi_l) * (phi - phi_l);
                smoothness_image.setPixel(x, y, jet_color_QRgb(smoothness_val * 2.0));

                // [0, pi], [-pi, pi]
                double theta_val, phi_val;
                tie(theta_val, phi_val) = normal2sphericalcoords(nx, ny, nz);
                double theta_ratio = clamp<double>(theta_val * 2.0 / 3.1415926535897, 0, 1);
                theta_image.setPixel(x, y, jet_color_QRgb(theta_ratio));

                double phi_ratio = clamp<double>((phi_val + 3.1415926535897)*0.5/3.1415926535897, 0, 1);
                phi_image.setPixel(x, y, jet_color_QRgb(phi_ratio));
              }
            }
          }
          normal_image.save( (results_path / fs::path("normal_opt_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );
          image_with_albedo_normal_lighting.save( (results_path / fs::path("normal_opt_lighting_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );
          image_error.save( (results_path / fs::path("error_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );
          integrability_image.save( (results_path / fs::path("integrability_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );
          smoothness_image.save( (results_path / fs::path("smoothness_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );
          theta_image.save( (results_path / fs::path("theta_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );
          phi_image.save( (results_path / fs::path("phi_" + std::to_string(i) + "_" + std::to_string(iters) + ".png")).string().c_str() );
        }
      } // [Shape from shading] main loop

      // Depth recovery
      {

        PhGUtils::message("[Shape from shading] Depth recovery.");
        const int num_cols = bundle.image.width(), num_rows = bundle.image.height();

        // [Depth recovery] step 1: prepare depth map and LoG of depth map
        cv::Mat depth_map_i = depth_maps_ref[i], depth_map_LoG_i;

        // render the original mesh to obtain depth map
        cv::filter2D(depth_map_i, depth_map_LoG_i, -1, LoG_kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

        // [Depth recovery] step 2: assemble matrices

        // ====================================================================
        // collect valid pixels
        // ====================================================================
        vector<glm::ivec2> pixel_indices_i;

        const int wsize = 0;
        for (int y = wsize; y < num_rows-wsize; ++y) {
          for (int x = wsize; x < num_cols-wsize; ++x) {

            bool valid = true;
            for(int dr=-wsize;dr<=wsize;++dr) {
              for(int dc=-wsize;dc<=wsize;++dc) {
                valid &= depth_map_i.at<double>(y+dr, x+dc) > -1e5;
              }
            }

            if (valid) {
              pixel_indices_i.push_back(glm::ivec2(y, x));
            }
          }
        }

        const int num_constraints = pixel_indices_i.size();

        vector<bool> is_valid_pixel(num_rows*num_cols, false);
        vector<int> pixel_index_map(num_rows*num_cols, -1);
        for(int j=0;j<num_constraints;++j) {
          int pidx = pixel_indices_i[j].x * num_cols + pixel_indices_i[j].y;
          is_valid_pixel[pidx] = true;
          pixel_index_map[pidx] = j;
        }

#if USE_THETA_PHI

#define USE_IMAGE_GRID 0

        PhGUtils::message("[Shape from shading] Depth recovery: assembling matrix.");
        vector<Tripletd> A_coeffs;
        VectorXd B(num_constraints * 4);
#if USE_IMAGE_GRID
        const double w_LoG = 0.01, w_diff = 0.0;
#else
        const double w_LoG = 0.01, w_diff = 0.01;
#endif
        // ====================================================================
        // part 1: normal constraints
        // ====================================================================
        for(int j=0;j<num_constraints;++j) {
          int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
          int pidx = r * num_cols + c;

          cv::Vec3d normal_ij = normal_maps[i].at<cv::Vec3d>(r, c);
          cv::Vec3d depth_ij = depth_maps[i].at<cv::Vec3d>(r, c);
          double nx = normal_ij[0], ny = normal_ij[1], nz = normal_ij[2];

          bool boundary_pixel = is_boundary[pidx];
          double w_pixel = boundary_pixel?0.0:1;
          //double w_pixel = 1.0;

          if(r > 0 && r < num_rows - 1) {
            int pidx_u = pidx - num_cols;
            //int pidx_d = pidx + num_cols;
            cv::Vec3d depth_ij_u = depth_maps[i].at<cv::Vec3d>(r-1, c);
            //cv::Vec3d depth_ij_d = depth_maps[i].at<cv::Vec3d>(r+1, c);
            if(is_valid_pixel[pidx_u] ) {
              //&& is_valid_pixel[pidx_d]) {
              A_coeffs.push_back(Tripletd(j*2, j, -nz * w_pixel));
              A_coeffs.push_back(Tripletd(j*2, pixel_index_map[pidx_u], nz * w_pixel));
              //A_coeffs.push_back(Tripletd(j*2, pixel_index_map[pidx_d], -nz));
#if USE_IMAGE_GRID
              B(j*2) = ny;
#else
              B(j*2) = -ny * (depth_ij_u[1] - depth_ij[1]) * w_pixel;
#endif
            }
          }

          if(c > 0 && c < num_cols - 1) {
            int pidx_l = pidx - 1;
            //int pidx_r = pidx + 1;
            cv::Vec3d depth_ij_l = depth_maps[i].at<cv::Vec3d>(r, c-1);
            //cv::Vec3d depth_ij_r = depth_maps[i].at<cv::Vec3d>(r, c+1);
            if(is_valid_pixel[pidx_l] ){
              //&& is_valid_pixel[pidx_r]) {
              A_coeffs.push_back(Tripletd(j*2+1, j, nz * w_pixel));
              A_coeffs.push_back(Tripletd(j*2+1, pixel_index_map[pidx_l], -nz * w_pixel));
              //A_coeffs.push_back(Tripletd(j*2+1, pixel_index_map[pidx_r], nz));

              // this negative here is critical for aligning the point cloud correctly
              // with the reference mesh
#if USE_IMAGE_GRID
              B(j*2+1) = nx;
#else
              B(j*2+1) = -nx * (depth_ij[0] - depth_ij_l[0]) * w_pixel;
#endif
            }
          }
        }
        cout << "part 1 done." << endl;

        // ====================================================================
        // part 2: LoG constaints
        // ====================================================================
        for(int j = 0; j < num_constraints; ++j) {
          int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
          int pidx = r * num_cols + c;

          for(auto p : LoG_coeffs_perpixel[pidx]) {
            int qidx;
            double val_LoG;
            tie(qidx, val_LoG) = p;

            bool boundary_pixel = is_boundary[pidx] || is_boundary[qidx];
            double w_pixel = boundary_pixel?0.0:1.0;

            auto good_LoG = [&](int pixel_index) {
              int y = pixel_index / num_cols, x = pixel_index % num_cols;
              bool flag = true;
              for(int dr=-2;dr<=2;++dr) {
                for(int dc=-2;dc<=2;++dc) {
                  flag &= depth_map_i.at<double>(y+dr, x+dc) > -1e5;
                }
              }
              return flag;
            };

            bool good_constraint = good_LoG(pidx) && good_LoG(qidx);
            if(!good_constraint) w_pixel = 0;

            if(is_valid_pixel[pidx] && is_valid_pixel[qidx]) {
              int new_i = pixel_index_map[pidx] + num_constraints * 2;
              int new_j = pixel_index_map[qidx];
              A_coeffs.push_back(Tripletd(new_i, new_j, val_LoG * w_LoG * w_pixel));
              B(new_i) = depth_map_LoG_i.at<double>(r, c) * w_LoG * w_pixel;
            }
          }
        }
        cout << "part 2 done." << endl;

        // ====================================================================
        // part 3: difference constaints
        // ====================================================================
        for(int j=0;j<num_constraints;++j) {
          int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
          int pidx = r * num_cols + c;
          int new_i = j + num_constraints * 3;
          int new_j = j;

          bool boundary_pixel = is_boundary[pidx];
          double w_discont = boundary_pixel?10.0:1.0;

          A_coeffs.push_back(Tripletd(new_i, new_j, w_diff*w_discont));
          B(new_i) = depth_map_i.at<double>(r, c) * w_diff*w_discont;
        }
        cout << "part 3 done." << endl;

        Eigen::SparseMatrix<double> A(num_constraints * 4, num_constraints);
        A.setFromTriplets(A_coeffs.begin(), A_coeffs.end());
        //A.makeCompressed();

        PhGUtils::message("done.");

        // [Depth recovery] step 3: solve linear least sqaures and generate point cloud / mesh
        Eigen::SparseMatrix<double> eye(num_constraints, num_constraints);
        for(int j=0;j<num_constraints;++j) eye.insert(j, j) = 1e-16;

        Eigen::SparseMatrix<double> AtA = A.transpose() * A;
        //AtA.makeCompressed();

        cout << AtA.rows() << 'x' << AtA.cols() << endl;
        cout << AtA.nonZeros() << endl;

        AtA += eye;

        // AtA is symmetric, so it is okay to use it as column major?
        CholmodSupernodalLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(AtA);
        if(solver.info()!=Success) {
          cerr << "Failed to decompose matrix A." << endl;

          {
            ofstream fout("A.txt");
            for(auto tt : A_coeffs) {
              fout << tt.row() << ' ' << tt.col() << ' ' << tt.value() << endl;
            }
            fout.close();
          }

          exit(-1);
        }

        VectorXd Atb = A.transpose() * B;
        VectorXd new_depth = solver.solve(Atb);
        if(solver.info()!=Success) {
          cerr << "Failed to solve A\b." << endl;
          exit(-1);
        }
        PhGUtils::message("solved.");

        // update depth map
        cv::Mat depth_map_final(num_rows, num_cols, CV_32F);
        for(int j=0;j<num_constraints;++j) {
          int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
          cv::Vec3d old_depth = depth_maps[i].at<cv::Vec3d>(r, c);
          depth_maps[i].at<cv::Vec3d>(r, c) = cv::Vec3d(old_depth[0], old_depth[1], new_depth(j));
          depth_map_final.at<float>(r, c) = new_depth(j);
        }

        // apply median filter to depth map
        //cv::medianBlur(depth_map_final, depth_map_final, 3);

        glm::dmat4 Rmat = glm::eulerAngleYXZ(bundle.params.params_model.R[0],
                                             bundle.params.params_model.R[1],
                                             bundle.params.params_model.R[2]);
        glm::dmat4 Rmat_inv = glm::eulerAngleZ(-bundle.params.params_model.R[2])
                            * glm::eulerAngleX(-bundle.params.params_model.R[1])
                            * glm::eulerAngleY(-bundle.params.params_model.R[0]);

        glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                         glm::dvec3(bundle.params.params_model.T[0],
                                                    bundle.params.params_model.T[1],
                                                    bundle.params.params_model.T[2]));
        glm::dmat4 Mview = Rmat;
        glm::dmat4 Mview_inv = glm::inverse(Mview);

        vector<glm::dvec3> depth_final;
        vector<glm::dvec3> depth_final_raw;
        for(int j=0;j<num_constraints;++j) {
          int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
          cv::Vec3d old_depth = depth_maps[i].at<cv::Vec3d>(r, c);
          float d_j = depth_map_final.at<float>(r, c);

          #if USE_IMAGE_GRID
          depth_final.push_back(glm::vec3(c, r, d_j));
          #else
          glm::dvec3 pt = glm::vec3(old_depth[0], old_depth[1], d_j);
          depth_final.push_back(pt);

          glm::dvec4 pt0 =  Rmat_inv * glm::dvec4(pt.x, pt.y, pt.z, 1.0);
          depth_final_raw.push_back(glm::dvec3(pt0.x, pt0.y, pt0.z));
          #endif
        }

#else  // USE_THETA_PHI

        glm::dmat4 Rmat = glm::eulerAngleYXZ(bundle.params.params_model.R[0],
                                             bundle.params.params_model.R[1],
                                             bundle.params.params_model.R[2]);
        glm::dmat4 Rmat_inv = glm::eulerAngleZ(-bundle.params.params_model.R[2])
                            * glm::eulerAngleX(-bundle.params.params_model.R[1])
                            * glm::eulerAngleY(-bundle.params.params_model.R[0]);

        glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                         glm::dvec3(bundle.params.params_model.T[0],
                                                    bundle.params.params_model.T[1],
                                                    bundle.params.params_model.T[2]));
        glm::dmat4 Mview = Rmat;
        glm::dmat4 Mview_inv = glm::inverse(Mview);

        vector<glm::dvec3> depth_final;
        vector<glm::dvec3> depth_final_raw;
        for(int j=0;j<num_constraints;++j) {
          int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
          cv::Vec3d old_depth = depth_maps[i].at<cv::Vec3d>(r, c);

          float d_j = zmaps[i].at<float>(r, c);

          #if USE_IMAGE_GRID
          depth_final.push_back(glm::vec3(c, r, d_j));
          #else
          glm::dvec3 pt = glm::vec3(old_depth[0], old_depth[1], d_j);
          depth_final.push_back(pt);

          glm::dvec4 pt0 =  Rmat_inv * glm::dvec4(pt.x, pt.y, pt.z, 1.0);
          depth_final_raw.push_back(glm::dvec3(pt0.x, pt0.y, pt0.z));
          #endif
        }

#endif

        // write out the new depth map
        {
          ofstream fout( (results_path / fs::path("point_cloud_opt" + std::to_string(i) + ".txt")).string() );
          for(auto p : depth_final) {
            fout << p.x << ' ' << p.y << ' ' << p.z << endl;
          }
          fout.close();
        }

        {
          ofstream fout( (results_path / fs::path("point_cloud_opt_raw" + std::to_string(i) + ".txt")).string() );
          for(auto p : depth_final_raw) {
            fout << p.x << ' ' << p.y << ' ' << p.z << endl;
          }
          fout.close();
        }

        // write out the depth mesh
        vector<int> depth_node_map(num_rows * num_cols, -1);
        for(int j=0;j<num_constraints;++j) {
          int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
          depth_node_map[r*num_cols+c] = j + 1;
        }

        {
          ofstream fout((results_path / fs::path("point_cloud_opt" + std::to_string(i) + ".obj")).string());
          for(int j=0;j<depth_final_raw.size();++j) {
            auto& p = depth_final[j];
            fout << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
          }
          for(int j=0;j<depth_final_raw.size();++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            int idx = r * num_cols + c;
            int lidx = idx - 1;
            int ridx = idx + 1;
            int uidx = idx - num_cols;
            int didx = idx + num_cols;
            if(ridx < num_cols * num_rows && didx < num_cols * num_rows) {
              if(depth_node_map[ridx] > 0 && depth_node_map[didx] > 0) {
                fout << "f " << depth_node_map[idx] << " " << depth_node_map[didx] << " " << depth_node_map[ridx] << '\n';
              }
            }
            if(lidx >= 0 && uidx >= 0) {
              if(depth_node_map[lidx] > 0 && depth_node_map[uidx] > 0) {
                fout << "f " << depth_node_map[idx] << " " << depth_node_map[uidx] << " " << depth_node_map[lidx] << '\n';
              }
            }
          }
          fout.close();
        }

        {
          ofstream fout((results_path / fs::path("point_cloud_opt_raw" + std::to_string(i) + ".obj")).string());
          for(int j=0;j<depth_final_raw.size();++j) {
            auto& p = depth_final_raw[j];
            fout << "v " << p.x << ' ' << p.y << ' ' << p.z << '\n';
          }
          for(int j=0;j<depth_final_raw.size();++j) {
            int r = pixel_indices_i[j].x, c = pixel_indices_i[j].y;
            int idx = r * num_cols + c;
            int lidx = idx - 1;
            int ridx = idx + 1;
            int uidx = idx - num_cols;
            int didx = idx + num_cols;
            if(ridx < num_cols * num_rows && didx < num_cols * num_rows) {
              if(depth_node_map[ridx] > 0 && depth_node_map[didx] > 0) {
                fout << "f " << depth_node_map[idx] << " " << depth_node_map[didx] << " " << depth_node_map[ridx] << '\n';
              }
            }
            if(lidx >= 0 && uidx >= 0) {
              if(depth_node_map[lidx] > 0 && depth_node_map[uidx] > 0) {
                fout << "f " << depth_node_map[idx] << " " << depth_node_map[uidx] << " " << depth_node_map[lidx] << '\n';
              }
            }
          }
          fout.close();
        }
      }
    } // per-image shape estimation

  } // [Shape from shading]

  return 0;
}

#endif  // FACE_SHAPE_FROM_SHADING
