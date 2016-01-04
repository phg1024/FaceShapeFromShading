#ifndef FACESHAPEFROMSHADING_UTILS_H
#define FACESHAPEFROMSHADING_UTILS_H

#include <common.h>

#include <QImage>

#include <MultilinearReconstruction/utils.hpp>

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

inline set<int> FindTrianglesIndices(const QImage& img) {
  set<int> S;
  int w = img.width(), h = img.height();
  for(int i=0;i<h;++i) {
    for(int j=0;j<w;++j) {
      QRgb pix = img.pixel(j, i);
      unsigned char r = static_cast<unsigned char>(qRed(pix));
      unsigned char g = static_cast<unsigned char>(qGreen(pix));
      unsigned char b = static_cast<unsigned char>(qBlue(pix));

      if(r == 0 && g == 0 && b == 0) continue;
      else {
        int idx;
        decode_index(r, g, b, idx);
        S.insert(idx);
      }
    }
  }
  return S;
}

#endif //FACESHAPEFROMSHADING_UTILS_H
