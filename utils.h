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
