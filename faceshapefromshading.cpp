#ifndef FACE_SHAPE_FROM_SHADING_H
#define FACE_SHAPE_FROM_SHADING_H

#include "Geometry/geometryutils.hpp"
#include "Utils/utility.hpp"

#include <QApplication>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOffscreenSurface>
#include <GL/freeglut_std.h>

#include "common.h"
#include "glm/glm.hpp"
#include "MultilinearReconstruction/basicmesh.h"

struct PixelInfo {
  PixelInfo() : fidx(-1) {}
  PixelInfo(int fidx, glm::vec3 bcoords) : fidx(fidx), bcoords(bcoords) {}

  int fidx;           // trinagle index
  glm::vec3 bcoords;  // bary centric coordinates
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

  BasicMesh mesh(template_mesh_filename);

  const int tex_size = 2048;

  auto encode_index = [](int idx, unsigned char& r, unsigned char& g, unsigned char& b) {
    r = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
    g = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
    b = static_cast<unsigned char>(idx & 0xff);
  };

  auto decode_index = [](unsigned char r, unsigned char g, unsigned char b, int& idx) {
    idx = b; idx <<= 8; idx |= g; idx <<= 8; idx |= r;
  };

  // Generate index map for albedo
  PhGUtils::message("generate index map for albedo.");
  QImage albedo_index_map;
  {
    QSurfaceFormat format;
    format.setMajorVersion(3);
    format.setMinorVersion(3);

    QOffscreenSurface surface;
    surface.setFormat(format);
    surface.create();

    QOpenGLContext context;
    context.setFormat(format);
    if (!context.create())
      qFatal("Cannot create the requested OpenGL context!");
    context.makeCurrent(&surface);

    const QRect drawRect(0, 0, tex_size, tex_size);
    const QSize drawRectSize = drawRect.size();

    QOpenGLFramebufferObjectFormat fboFormat;
    fboFormat.setSamples(16);
    fboFormat.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);

    QOpenGLFramebufferObject fbo(drawRectSize, fboFormat);
    fbo.bind();

    // draw the triangles

    // setup OpenGL viewing
    glShadeModel(GL_FLAT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, 1.0, 0.0, 1.0);
    glViewport(0, 0, tex_size, tex_size);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    PhGUtils::message("rendering index map.");
    for(int face_i = 0; face_i < mesh.NumFaces(); ++face_i) {
      auto normal_i = mesh.normal(face_i);
      auto f = mesh.face_texture(face_i);
      auto t0 = mesh.texture_coords(f[0]), t1 = mesh.texture_coords(f[1]), t2 = mesh.texture_coords(f[2]);
      unsigned char r, g, b;
      encode_index(face_i, r, g, b);
      glBegin(GL_TRIANGLES);
      glColor4ub(r, g, b, 255);
      glVertex2f(t0[0], t0[1]);
      glVertex2f(t1[0], t1[1]);
      glVertex2f(t2[0], t2[1]);
      glEnd();
    }
    PhGUtils::message("done.");

    // get the bitmap and save it as an image
    QImage img = fbo.toImage();
    fbo.release();
    img.save("texture_map.png");
    albedo_index_map = img;
  }

  // Compute the barycentric coordinates for each pixel
  vector<vector<PixelInfo>> albedo_pixel_map(tex_size, vector<PixelInfo>(tex_size));

  // Generate pixel map for albedo
  PhGUtils::message("generating pixel map for albedo ...");
  for(int i=0;i<tex_size;++i) {
    for(int j=0;j<tex_size;++j) {
      double y = i / static_cast<double>(tex_size - 1);
      double x = j / static_cast<double>(tex_size - 1);

      QRgb pix = albedo_index_map.pixel(j, i);
      unsigned char r = static_cast<unsigned char>(qRed(pix));
      unsigned char g = static_cast<unsigned char>(qGreen(pix));
      unsigned char b = static_cast<unsigned char>(qBlue(pix));
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
      albedo_pixel_map[i][j] = PixelInfo(fidx, glm::vec3(bcoords.x, bcoords.y, bcoords.z));
    }
  }
  PhGUtils::message("done.");

  // Collect texture information from each input (image, mesh) pair


  return 0;
}

#endif  // FACE_SHAPE_FROM_SHADING
