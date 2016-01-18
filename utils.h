#ifndef FACESHAPEFROMSHADING_UTILS_H
#define FACESHAPEFROMSHADING_UTILS_H

#include <common.h>

#include <QImage>

#include <MultilinearReconstruction/utils.hpp>

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

  if(x0 < 0 || y0 < 0) return glm::dvec3(0, 0, 0);
  if(x1 >= img.width() || y1 >= img.height()) return glm::dvec3(0, 0, 0);

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
  for(int y=-k, i=0;y<=k;++y, ++i) {
    double y2 = y * y;
    for(int x=-k, j=0;x<=k;++x, ++j) {
      double x2 = x * x;
      double val = (x2 + y2) / (2 * sigma2);
      kernel(i, j) = (val - 1) * exp(-val);
    }
  }
  const double PI = 3.1415926535897;
  kernel /= (PI * sigma4);
  return kernel;
}

#endif //FACESHAPEFROMSHADING_UTILS_H
