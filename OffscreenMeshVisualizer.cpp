#include "OffscreenMeshVisualizer.h"

#include <GL/freeglut_std.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

void OffscreenMeshVisualizer::SetupViewing() const {
  switch(mode) {
    case OrthoNormal: {
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluOrtho2D(0.0, 1.0, 0.0, 1.0);
      glViewport(0, 0, width, height);

      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      break;
    }
    case CamPerspective: {
      glMatrixMode(GL_PROJECTION);

      const double aspect_ratio =
        camera_params.image_size.x / camera_params.image_size.y;

      const double far = camera_params.far;
      // near is the focal length
      const double near = camera_params.focal_length;
      const double top = near * tan(0.5 * camera_params.fovy);
      const double right = top * aspect_ratio;
      glm::dmat4 Mproj = glm::dmat4(near/right, 0, 0, 0,
                                    0, near/top, 0, 0,
                                    0, 0, -(far+near)/(far-near), -1,
                                    0, 0, -2.0 * far * near / (far - near), 0.0);

      glLoadMatrixd(&Mproj[0][0]);

      glViewport(0, 0, width, height);

      glm::dmat4 Rmat = glm::eulerAngleYXZ(mesh_rotation[0],
                                           mesh_rotation[1],
                                           mesh_rotation[2]);

      glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                       glm::dvec3(mesh_translation[0],
                                                  mesh_translation[1],
                                                  mesh_translation[2]));

      glm::dmat4 MV = Tmat * Rmat;
      glMatrixMode(GL_MODELVIEW);
      glLoadMatrixd(&MV[0][0]);
    }
  }
}

QImage OffscreenMeshVisualizer::Render() const {
  PhGUtils::message("generating index map for albedo.");
  boost::timer::auto_cpu_timer t("index map for albedo generation time = %w seconds.\n");
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

  const QRect drawRect(0, 0, width, height);
  const QSize drawRectSize = drawRect.size();

  QOpenGLFramebufferObjectFormat fboFormat;
  fboFormat.setSamples(16);
  fboFormat.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);

  auto encode_index = [](int idx, unsigned char& r, unsigned char& g, unsigned char& b) {
    r = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
    g = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
    b = static_cast<unsigned char>(idx & 0xff);
  };

  auto decode_index = [](unsigned char r, unsigned char g, unsigned char b, int& idx) {
    idx = b; idx <<= 8; idx |= g; idx <<= 8; idx |= r;
    return idx;
  };

  QOpenGLFramebufferObject fbo(drawRectSize, fboFormat);
  fbo.bind();

  // draw the triangles

  // setup OpenGL viewing
#define DEBUG_GEN 0   // Change this to 1 to generate albedo pixel map
#if DEBUG_GEN
  glShadeModel(GL_SMOOTH);
#else
  glShadeModel(GL_FLAT);
#endif
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  SetupViewing();

  switch(render_mode) {
    case Texture: {
      PhGUtils::message("rendering index map.");
      for(int face_i = 0; face_i < mesh.NumFaces(); ++face_i) {
        auto normal_i = mesh.normal(face_i);
        auto f = mesh.face_texture(face_i);
        auto t0 = mesh.texture_coords(f[0]), t1 = mesh.texture_coords(f[1]), t2 = mesh.texture_coords(f[2]);
        unsigned char r, g, b;
        encode_index(face_i, r, g, b);
        int tmp_idx;
        assert(decode_index(r, g, b, tmp_idx) == face_i);
        glBegin(GL_TRIANGLES);

#if DEBUG_GEN
        glColor3f(1, 0, 0);
      glVertex2f(t0[0], t0[1]);
      glColor3f(0, 1, 0);
      glVertex2f(t1[0], t1[1]);
      glColor3f(0, 0, 1);
      glVertex2f(t2[0], t2[1]);
#else
        glColor4ub(r, g, b, 255);
        glVertex2f(t0[0], t0[1]);
        glVertex2f(t1[0], t1[1]);
        glVertex2f(t2[0], t2[1]);
#endif
        glEnd();
      }
      PhGUtils::message("done.");
      break;
    }
    case Mesh: {
      PhGUtils::message("rendering index map.");
      for(int face_i = 0; face_i < mesh.NumFaces(); ++face_i) {
        auto normal_i = mesh.normal(face_i);
        auto f = mesh.face(face_i);
        auto v0 = mesh.vertex(f[0]), v1 = mesh.vertex(f[1]), v2 = mesh.vertex(f[2]);
        unsigned char r, g, b;
        encode_index(face_i, r, g, b);
        int tmp_idx;
        assert(decode_index(r, g, b, tmp_idx) == face_i);
        glBegin(GL_TRIANGLES);

        glColor4ub(r, g, b, 255);
        glVertex3f(v0[0], v0[1], v0[2]);
        glVertex3f(v1[0], v1[1], v1[2]);
        glVertex3f(v2[0], v2[1], v2[2]);

        glEnd();
      }
      PhGUtils::message("done.");
    }
  }

  // get the bitmap and save it as an image
  QImage img = fbo.toImage();
  fbo.release();
  return img;
}