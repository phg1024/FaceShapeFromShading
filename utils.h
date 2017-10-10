#ifndef FACESHAPEFROMSHADING_UTILS_H
#define FACESHAPEFROMSHADING_UTILS_H

#include <common.h>

#include "Geometry/geometryutils.hpp"
#include "Utils/utility.hpp"

#include <QFile>
#include <QImage>
#include <QColor>

#include <MultilinearReconstruction/basicmesh.h>
#include <MultilinearReconstruction/costfunctions.h>
#include <MultilinearReconstruction/ioutilities.h>
#include <MultilinearReconstruction/multilinearmodel.h>
#include <MultilinearReconstruction/parameters.h>
#include <MultilinearReconstruction/OffscreenMeshVisualizer.h>
#include <MultilinearReconstruction/statsutils.h>
#include <MultilinearReconstruction/utils.hpp>

#include "defs.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include <boost/timer/timer.hpp>

namespace fs = boost::filesystem;

#include "json/src/json.hpp"
using json = nlohmann::json;

template <typename T>
pair<T, T> normal2sphericalcoords(T nx, T ny, T nz) {
  // nx = sin(theta) * sin(phi)
  // ny = sin(theta) * cos(phi)
  // nz = cos(theta)
  return make_pair(acos(nz), atan2(nx, ny));
}

template <typename T>
tuple<T, T, T> sphericalcoords2normal(double theta, double phi) {
  // nx = sin(theta) * sin(phi)
  // ny = sin(theta) * cos(phi)
  // nz = cos(theta)
  return make_tuple(sin(theta)*sin(phi), sin(theta)*cos(phi), cos(theta));
}

inline Vector3d dnormal_dtheta(double theta, double phi) {
  // nx = sin(theta) * sin(phi)
  // ny = sin(theta) * cos(phi)
  // nz = cos(theta)

  // dnx_dtheta = cos(theta) * sin(phi)
  // dny_dtheta = cos(theta) * cos(phi)
  // dnz_dtheta = -sin(theta)
  return Vector3d(cos(theta)*sin(phi), cos(theta)*cos(phi), -sin(theta));
}

inline Vector3d dnormal_dphi(double theta, double phi) {
  // nx = sin(theta) * sin(phi)
  // ny = sin(theta) * cos(phi)
  // nz = cos(theta)

  // dnx_dtheta = sin(theta) * cos(phi)
  // dny_dtheta = -sin(theta) * sin(phi)
  // dnz_dtheta = 0
  return Vector3d(sin(theta)*cos(phi), -sin(theta)*sin(phi), 0);
}

inline VectorXd sphericalharmonics(double nx, double ny, double nz) {
  VectorXd Y(9);
  Y(0) = 1.0;
  Y(1) = nx; Y(2) = ny; Y(3) = nz;
  Y(4) = nx * ny; Y(5) = nx * nz; Y(6) = ny * nz;
  Y(7) = nx * nx - ny * ny; Y(8) = 3 * nz * nz - 1;
  return Y;
}

inline MatrixXd dY_dnormal(double nx, double ny, double nz) {
  MatrixXd dYdnormal(9, 3);
  dYdnormal(0, 0) = 0; dYdnormal(0, 1) = 0; dYdnormal(0, 2) = 0;

  dYdnormal(1, 0) = 1; dYdnormal(1, 1) = 0; dYdnormal(1, 2) = 0;
  dYdnormal(2, 0) = 0; dYdnormal(2, 1) = 1; dYdnormal(2, 2) = 0;
  dYdnormal(3, 0) = 0; dYdnormal(3, 1) = 0; dYdnormal(3, 2) = 1;

  dYdnormal(4, 0) = ny; dYdnormal(4, 1) = nx; dYdnormal(4, 2) = 0;
  dYdnormal(5, 0) = nz; dYdnormal(5, 1) = 0; dYdnormal(5, 2) = nx;
  dYdnormal(6, 0) = 0; dYdnormal(6, 1) = nz; dYdnormal(6, 2) = ny;

  dYdnormal(7, 0) = 2*nx; dYdnormal(7, 1) = -2*ny; dYdnormal(7, 2) = 0;
  dYdnormal(8, 0) = 0; dYdnormal(8, 1) = 0; dYdnormal(8, 2) = 6 * nz;

  return dYdnormal;
}

template <typename T>
T clamp(T val, T lower, T upper) {
  return std::max(lower, std::min(upper, val));
}

inline void encode_index(int idx, unsigned char& r, unsigned char& g, unsigned char& b) {
  r = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
  g = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
  b = static_cast<unsigned char>(idx & 0xff);
}

inline int decode_index(unsigned char r, unsigned char g, unsigned char b, int& idx) {
  idx = b; idx <<= 8; idx |= g; idx <<= 8; idx |= r;
  return idx;
}

inline glm::dvec3 bilinear_sample(const QImage& img, double x, double y) {
  int x0 = floor(x), x1 = x0 + 1;
  int y0 = floor(y), y1 = y0 + 1;

  if(x0 < 0 || y0 < 0) return glm::dvec3(-1, -1, -1);
  if(x1 >= img.width() || y1 >= img.height()) return glm::dvec3(-1, -1, -1);

  double c0 = x - x0, c0c = 1 - c0;
  double c1 = y - y0, c1c = 1 - c1;

  QRgb p00 = img.pixel(x0, y0);
  QRgb p01 = img.pixel(x1, y0);
  QRgb p10 = img.pixel(x0, y1);
  QRgb p11 = img.pixel(x1, y1);

  double r = c0c * c1c * qRed(p00) + c0c * c1 * qRed(p01) + c0 * c1c * qRed(p10) + c0 * c1 * qRed(p11);
  double g = c0c * c1c * qGreen(p00) + c0c * c1 * qGreen(p01) + c0 * c1c * qGreen(p10) + c0 * c1 * qGreen(p11);
  double b = c0c * c1c * qBlue(p00) + c0c * c1 * qBlue(p01) + c0 * c1c * qBlue(p10) + c0 * c1 * qBlue(p11);

  return glm::dvec3(r, g, b);
}

inline QRgb jet_color_QRgb(double ratio) {
  double r = max(0.0, min(1.0, (ratio - 0.5) / 0.25));
  double g = 0;
  double b = 1.0 - max(0.0, min(1.0, (ratio - 0.25) / 0.25 ));
  if(ratio < 0.5) {
    g = min(1.0, ratio / 0.25);
  } else {
    g = 1.0 - max(0.0, (ratio - 0.75) / 0.25);
  }
  return qRgb(r*255, g*255, b*255);
}

inline glm::dvec3 jet_color(double ratio) {
  double r = max(0.0, min(1.0, (ratio - 0.5) / 0.25));
  double g = 0;
  double b = 1.0 - max(0.0, min(1.0, (ratio - 0.25) / 0.25 ));
  if(ratio < 0.5) {
    g = min(1.0, ratio / 0.25);
  } else {
    g = 1.0 - max(0.0, (ratio - 0.75) / 0.25);
  }
  return glm::dvec3(r*255, g*255, b*255);
}

inline int get_image_index(const string& filename) {
  return std::stoi(filename.substr(0, filename.size()-4));
}

inline pair<set<int>, vector<int>> FindTrianglesIndices(const QImage& img) {
  int w = img.width(), h = img.height();
  set<int> S;
  vector<int> indices_map(w*h);
  for(int i=0, pidx = 0;i<h;++i) {
    for(int j=0;j<w;++j, ++pidx) {
      QRgb pix = img.pixel(j, i);
      unsigned char r = static_cast<unsigned char>(qRed(pix));
      unsigned char g = static_cast<unsigned char>(qGreen(pix));
      unsigned char b = static_cast<unsigned char>(qBlue(pix));

      if(r == 0 && g == 0 && b == 0) {
        indices_map[pidx] = -1;
        continue;
      }
      else {
        int idx;
        decode_index(r, g, b, idx);
        S.insert(idx);
        indices_map[pidx] = idx;
      }
    }
  }
  return make_pair(S, indices_map);
}

inline MatrixXd ComputeLoGKernel(int k, double sigma) {
  MatrixXd kernel(2*k+1, 2*k+1);
  const double sigma2 = sigma * sigma;
  const double sigma4 = sigma2 * sigma2;
  const double sigma6 = sigma2 * sigma4;

  double S = 0.0;
  for(int y=-k, i=0;y<=k;++y, ++i) {
    double y2 = y * y;
    for(int x=-k, j=0;x<=k;++x, ++j) {
      double x2 = x * x;
      double val = exp(-(x2 + y2) / (2 * sigma2));
      kernel(i, j) = val;
      S += val;
    }
  }

  const double PI = 3.1415926535897;
  double S2 = 0.0;
  for(int y=-k, i=0;y<=k;++y, ++i) {
    double y2 = y * y;
    for(int x=-k, j=0;x<=k;++x, ++j) {
      double x2 = x * x;
      kernel(i, j) *= (x2 + y2 - 2 * sigma2);
      kernel(i, j) /= S;
      S2 += kernel(i, j);
    }
  }

  S2 /= ((2*k+1) * (2*k+1));

  for(int i=0;i<2*k+1;++i) {
    for(int j=0;j<2*k+1;++j) {
      kernel(i, j) -= S2;
    }
  }

  return kernel;
}

inline Vector3d rgb2lab(double r, double g, double b) {
  Vector3d rgb(r, g, b);
  Matrix3d RGB2LMS;
  RGB2LMS << 0.3811, 0.5783, 0.0402,
             0.1967, 0.7244, 0.0782,
             0.0241, 0.1288, 0.8444;
  Matrix3d mb, mc;
  mb << 1.0/sqrt(3.0), 0, 0,
       0, 1.0/sqrt(6.0), 0,
       0, 0, 1.0/sqrt(2.0);
  mc << 1, 1, 1,
       1, 1, -2,
       1, -1, 0;
  Matrix3d LMS2lab = mb * mc;
  Vector3d Lab = LMS2lab * RGB2LMS * rgb;
  return Lab;
}

static QImage TransferColor(const QImage& source, const QImage& target,
                            const vector<int>& valid_pixels_s,
                            const vector<int>& valid_pixels_t) {
  // Make a copy
  QImage result = source;

  const int num_rows_s = source.height(), num_cols_s = source.width();
  const int num_rows_t = target.height(), num_cols_t = target.width();
  const size_t num_pixels_s = valid_pixels_s.size();
  const size_t num_pixels_t = valid_pixels_t.size();

  Matrix3d RGB2LMS, LMS2RGB;
  RGB2LMS << 0.3811, 0.5783, 0.0402,
             0.1967, 0.7244, 0.0782,
             0.0241, 0.1288, 0.8444;
  LMS2RGB << 4.4679, -3.5873, 0.1193,
            -1.2186, 2.3809, -0.1624,
             0.0497, -0.2439, 1.2045;

  Matrix3d b, c, b2, c2;
  b << 1.0/sqrt(3.0), 0, 0,
       0, 1.0/sqrt(6.0), 0,
       0, 0, 1.0/sqrt(2.0);
  c << 1, 1, 1,
       1, 1, -2,
       1, -1, 0;
  b2 << sqrt(3.0)/3.0, 0, 0,
        0, sqrt(6.0)/6.0, 0,
        0, 0, sqrt(2.0)/2.0;
  c2 << 1, 1, 1,
        1, 1, -1,
        1, -2, 0;
  Matrix3d LMS2lab = b * c;
  Matrix3d lab2LMS = c2 * b2;

  auto unpack_pixel = [](QRgb pix) {
    int r = max(1, qRed(pix)), g = max(1, qGreen(pix)), b = max(1, qBlue(pix));
    return make_tuple(r, g, b);
  };

  auto compute_image_stats = [&](const QImage& img, const vector<int>& valid_pixels) {
    const size_t num_pixels = valid_pixels.size();
    const int num_cols = img.width(), num_rows  = img.height();
    MatrixXd pixels(3, num_pixels);

    cout << num_cols << 'x' << num_rows << endl;

    for(size_t i=0;i<num_pixels;++i) {
      int y = valid_pixels[i] / num_cols;
      int x = valid_pixels[i] % num_cols;

      int r, g, b;
      tie(r, g, b) = unpack_pixel(img.pixel(x, y));
      pixels.col(i) = Vector3d(r / 255.0, g / 255.0, b / 255.0);
    }

    MatrixXd pixels_LMS = RGB2LMS * pixels;

    for(int i=0;i<3;i++) {
      for(int j=0;j<num_pixels;++j) {
        pixels_LMS(i, j) = log10(pixels_LMS(i, j));
      }
    }

    MatrixXd pixels_lab = LMS2lab * pixels_LMS;

    Vector3d mean = pixels_lab.rowwise().mean();
    Vector3d stdev(0, 0, 0);
    for(int i=0;i<num_pixels;++i) {
      Vector3d diff = pixels_lab.col(i) - mean;
      stdev += Vector3d(diff[0]*diff[0], diff[1]*diff[1], diff[2]*diff[2]);
    }
    stdev /= (num_pixels - 1);

    for(int i=0;i<3;++i) stdev[i] = sqrt(stdev[i]);

    cout << "mean: " << mean << endl;
    cout << "std: " << stdev << endl;

    return make_tuple(pixels_lab, mean, stdev);
  };

  // Compute stats of both images
  MatrixXd lab_s, lab_t;
  Vector3d mean_s, std_s, mean_t, std_t;
  tie(lab_s, mean_s, std_s) = compute_image_stats(source, valid_pixels_s);
  tie(lab_t, mean_t, std_t) = compute_image_stats(target, valid_pixels_t);

  // Do the transfer
  MatrixXd res(3, num_pixels_s);

  // Transfer color by shifting
  for(int i=0;i<3;++i) {
    for(int j=0;j<num_pixels_s;++j) {
      //res(i, j) = (lab_s(i, j) - mean_s[i]) * std_t[i] / std_s[i] + mean_t[i];
      res(i, j) = lab_s(i, j) - mean_s[i] + mean_t[i];
    }
  }

  MatrixXd LMS_res = lab2LMS * res;
  for(int i=0;i<3;++i) {
    for(int j=0;j<num_pixels_s;++j) {
      LMS_res(i, j) = pow(10, LMS_res(i, j));
    }
  }

  MatrixXd est_im = LMS2RGB * LMS_res;
  for(size_t i=0;i<num_pixels_s;++i) {
    int y = valid_pixels_s[i] / num_cols_s;
    int x = valid_pixels_s[i] % num_cols_s;
    result.setPixel(x, y, qRgb(clamp<double>(est_im(0, i) * 255.0, 0, 255),
                               clamp<double>(est_im(1, i) * 255.0, 0, 255),
                               clamp<double>(est_im(2, i) * 255.0, 0, 255)));
  }
  return result;
}

inline QImage GetIndexMap(const string& albedo_index_map_filename,
                   const BasicMesh& mesh,
                   bool generate_index_map = true,
                   int tex_size = 2048) {
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
  return albedo_index_map;
}

inline pair<QImage, vector<vector<PixelInfo>>> GetPixelCoordinatesMap(
  const string& albedo_pixel_map_filename,
  const QImage& albedo_index_map,
  const BasicMesh& mesh,
  bool gen_pixel_map = false,
  int tex_size = 2048) {

  vector<vector<PixelInfo>> albedo_pixel_map(tex_size, vector<PixelInfo>(tex_size));

  // Generate pixel map for albedo
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
    //pixel_map_image.save("albedo_pixel.png");
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
    }
    pixel_map_image.save("albedo_pixel.jpg");
    PhGUtils::message("done.");
  }

  return make_pair(pixel_map_image, albedo_pixel_map);
}

inline void ApplyWeights(
  BasicMesh& mesh,
  const vector<BasicMesh>& blendshapes,
  const VectorXd& weights
) {
  const int num_blendshapes = 46;
  MatrixX3d verts0 = blendshapes[0].vertices();
  MatrixX3d verts = verts0;
  for(int j=1;j<=num_blendshapes;++j) {
    verts += (blendshapes[j].vertices() - verts0) * weights(j);
  }
  mesh.vertices() = verts;
  mesh.ComputeNormals();
}

inline tuple<QImage, vector<vector<int>>> GenerateMeanTexture(
  const vector<ImageBundle> image_bundles,
  MultilinearModel& model,
  const vector<BasicMesh>& blendshapes,
  BasicMesh& mesh,
  int tex_size,
  vector<vector<PixelInfo>>& albedo_pixel_map,
  vector<vector<glm::dvec3>>& mean_texture,
  vector<vector<double>>& mean_texture_weight,
  cv::Mat& mean_texture_mat,
  const string& mean_albedo_filename,
  const fs::path& results_path,
  const string& options) {
  QImage mean_texture_image;
  vector<vector<int>> face_indices_maps;
  {
    json settings = json::parse(options);

    cout << settings << endl;

    bool generate_mean_texture = settings["generate_mean_texture"];
    bool use_blendshapes = settings["use_blendshapes"];

    // use a larger scale when generating mean texture with blendshapes
    // since blendshapes are subdivided meshes and each triangle is much smaller
    double scale_factor = 1.0;
    if(use_blendshapes) scale_factor = 2.0;

    for(auto& bundle : image_bundles) {
      // get the geometry of the mesh, update normal
      if(use_blendshapes) {
        ApplyWeights(mesh, blendshapes, bundle.params.params_model.Wexp_FACS);
      } else {
        model.ApplyWeights(bundle.params.params_model.Wid, bundle.params.params_model.Wexp);
        mesh.UpdateVertices(model.GetTM());
        mesh.ComputeNormals();
      }

      // for each image bundle, render the mesh to FBO with culling to get the visible triangles
      OffscreenMeshVisualizer visualizer(bundle.image.width() * scale_factor, bundle.image.height() * scale_factor);
      visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
      visualizer.SetRenderMode(OffscreenMeshVisualizer::Mesh);
      visualizer.BindMesh(mesh);
      visualizer.SetCameraParameters(bundle.params.params_cam);
      visualizer.SetMeshRotationTranslation(bundle.params.params_model.R, bundle.params.params_model.T);
      visualizer.SetIndexEncoded(true);
      visualizer.SetEnableLighting(false);
      QImage img = visualizer.Render();
      img.save("mesh.png");

      // find the visible triangles from the index map
      auto triangles_indices_pair = FindTrianglesIndices(img);
      set<int> triangles = triangles_indices_pair.first;
      face_indices_maps.push_back(triangles_indices_pair.second);
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

      if(generate_mean_texture) {
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

            if(texel.r < 0 && texel.g < 0 && texel.b < 0) continue;

            mean_texture[i][j] += texel;
            mean_texture_weight[i][j] += 1.0;
          }
        }
      }
    }

    // [Optional]: render the mesh with texture to verify the texel values
    if(generate_mean_texture) {
      mean_texture_image = QImage(tex_size, tex_size, QImage::Format_ARGB32);
      mean_texture_image.fill(0);
      for(int i=0;i<tex_size;++i) {
        for (int j = 0; j < (tex_size/2); ++j) {
          double weight_ij = mean_texture_weight[i][j];
          double weight_ij_s = mean_texture_weight[i][tex_size-1-j];

          if(weight_ij == 0 && weight_ij_s == 0) {
            mean_texture_mat.at<cv::Vec3d>(i, j) = cv::Vec3d(0, 0, 0);
            continue;
          } else {
            glm::dvec3 texel = (mean_texture[i][j] + mean_texture[i][tex_size-1-j]) / (weight_ij + weight_ij_s);
            mean_texture[i][j] = texel;
            mean_texture[i][tex_size-1-j] = texel;
            mean_texture_image.setPixel(j, i, qRgb(texel.r, texel.g, texel.b));
            mean_texture_image.setPixel(tex_size-1-j, i, qRgb(texel.r, texel.g, texel.b));

            mean_texture_mat.at<cv::Vec3d>(i, j) = cv::Vec3d(texel.x, texel.y, texel.z);
            mean_texture_mat.at<cv::Vec3d>(i, tex_size-1-j) = cv::Vec3d(texel.x, texel.y, texel.z);
          }
        }
      }

      string refine_method = settings["refine_method"];

      cv::Mat mean_texture_refined_mat = mean_texture_mat;
      if(refine_method == "mean_shift") {
        cv::resize(mean_texture_mat, mean_texture_mat, cv::Size(), 0.25, 0.25);
        mean_texture_refined_mat = StatsUtils::MeanShiftSegmentation(mean_texture_refined_mat, 5.0, 30.0, 0.5);
        mean_texture_refined_mat = 0.25 * mean_texture_mat + 0.75 * mean_texture_refined_mat;
        mean_texture_refined_mat = StatsUtils::MeanShiftSegmentation(mean_texture_refined_mat, 10.0, 30.0, 0.5);
        mean_texture_refined_mat = 0.25 * mean_texture_mat + 0.75 * mean_texture_refined_mat;
        mean_texture_refined_mat = StatsUtils::MeanShiftSegmentation(mean_texture_refined_mat, 20.0, 30.0, 0.5);
        mean_texture_refined_mat = 0.25 * mean_texture_mat + 0.75 * mean_texture_refined_mat;
        cv::resize(mean_texture_refined_mat, mean_texture_refined_mat, cv::Size(), 4.0, 4.0);
      } else if (refine_method == "hsv") {
        cout << "Refine using hsv method ..." << endl;
        // refine using core face region and clustering in hsv space
        const string core_face_region_filename = settings["core_face_region_filename"];
        auto core_face_region = cv::imread(core_face_region_filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

        vector<glm::ivec2> valid_pixels;
        for(int i=0;i<core_face_region.rows;++i) {
          for(int j=0;j<core_face_region.cols;++j) {
            unsigned char c = core_face_region.at<unsigned char>(i, j);
            if( c > 0 ) valid_pixels.push_back(glm::ivec2(i, j));
          }
        }
        cout << "valid pixels = " << valid_pixels.size() << endl;

        glm::dvec3 mean_color(0, 0, 0);
        for(auto p : valid_pixels) {
          cv::Vec3d pix = mean_texture_refined_mat.at<cv::Vec3d>(p.x, p.y);
          mean_color.r += pix[0] / 255.0;
          mean_color.g += pix[1] / 255.0;
          mean_color.b += pix[2] / 255.0;
        }
        mean_color /= valid_pixels.size();
        cv::Vec3d mean_color_vec(mean_color.r*255.0, mean_color.g*255.0, mean_color.b*255.0);

        QColor mean_color_qt = QColor::fromRgbF(mean_color.r, mean_color.g, mean_color.b);
        glm::dvec3 mean_hsv;
        mean_color_qt.getHsvF(&mean_hsv.r, &mean_hsv.g, &mean_hsv.b);

        const double distance_threshold = settings["hsv_threshold"];
        // convert the entire image to hsv
        for(int i=0;i<mean_texture_mat.rows;++i) {
          for(int j=0;j<mean_texture_mat.cols;++j) {
            cv::Vec3d pix = mean_texture_mat.at<cv::Vec3d>(i, j);
            QColor pix_color = QColor::fromRgb(pix[0], pix[1], pix[2]);
            glm::dvec3 pix_hsv;
            pix_color.getHsvF(&pix_hsv.r, &pix_hsv.g, &pix_hsv.b);
            double d_ij = glm::distance2(pix_hsv, mean_hsv);
            if(d_ij < distance_threshold) {
              // Change this ratio to control how much details to include in the albedo
              const double mix_ratio = settings["mix_ratio"];
              mean_texture_refined_mat.at<cv::Vec3d>(i, j) = mean_color_vec * mix_ratio + mean_texture_refined_mat.at<cv::Vec3d>(i, j) * (1-mix_ratio);
            }
          }
        }
      } else {
        // no refinement
      }

      QImage mean_texture_image_refined(tex_size, tex_size, QImage::Format_ARGB32);
      for(int i=0;i<tex_size;++i) {
        for(int j=0;j<tex_size;++j) {
          cv::Vec3d pix = mean_texture_refined_mat.at<cv::Vec3d>(i, j);
          mean_texture_image_refined.setPixel(j, i, qRgb(pix[0], pix[1], pix[2]));
        }
      }

      mean_texture_image.save( (results_path / fs::path("mean_texture.png")).string().c_str() );
      mean_texture_image_refined.save( (results_path / fs::path("mean_texture_refined.png")).string().c_str() );
      mean_texture_image = mean_texture_image_refined;
    } else {
      mean_texture_image = QImage(mean_albedo_filename.c_str());
      mean_texture_image.save( (results_path / fs::path("mean_texture.png")).string().c_str() );
    }
  }

  return make_tuple(mean_texture_image, face_indices_maps);
}

#endif //FACESHAPEFROMSHADING_UTILS_H
