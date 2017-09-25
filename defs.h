#ifndef DEFS_H
#define DEFS_H

struct PixelInfo {
  PixelInfo() : fidx(-1) {}
  PixelInfo(int fidx, glm::vec3 bcoords) : fidx(fidx), bcoords(bcoords) {}

  int fidx;           // trinagle index
  glm::vec3 bcoords;  // bary centric coordinates
};

struct ImageBundle {
  ImageBundle() {}
  ImageBundle(const string& filename,
              const QImage& image,
              const vector<Constraint2D>& points,
              const ReconstructionResult& params)
    : filename(filename), image(image), points(points), params(params) {}
  string filename;
  QImage image;
  vector<Constraint2D> points;
  ReconstructionResult params;
};

#endif
