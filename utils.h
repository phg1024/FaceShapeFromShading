#ifndef FACESHAPEFROMSHADING_UTILS_H
#define FACESHAPEFROMSHADING_UTILS_H

#include <common.h>

#include <QImage>

#include <MultilinearReconstruction/utils.hpp>

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
  for(int i=0;i<3;++i) {
    for(int j=0;j<num_pixels_s;++j) {
      res(i, j) = (lab_s(i, j) - mean_s[i]) * std_t[i] / std_s[i] + mean_t[i];
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

#endif //FACESHAPEFROMSHADING_UTILS_H
