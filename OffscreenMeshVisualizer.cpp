#include "OffscreenMeshVisualizer.h"
#include "utils.h"

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

QImage OffscreenMeshVisualizer::Render(bool multi_sampled) const {
  boost::timer::auto_cpu_timer t("render time = %w seconds.\n");
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
  // Disable sampling to avoid blending along edges
  if(multi_sampled) fboFormat.setSamples(16);
  else fboFormat.setSamples(0);
  fboFormat.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);

  QOpenGLFramebufferObject fbo(drawRectSize, fboFormat);
  fbo.bind();

  // draw the triangles

  // setup OpenGL viewing
#define DEBUG_GEN 0   // Change this to 1 to generate albedo pixel map
#if DEBUG_GEN
  glShadeModel(GL_SMOOTH);
  glDisable(GL_BLEND);
#else
  glShadeModel(GL_FLAT);
#endif

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  SetupViewing();

  switch(render_mode) {
    case Texture: {
      PhGUtils::message("rendering texture.");
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
        glColor4f(1, 0, 0, 1);
        glVertex2f(t0[0], t0[1]);
        glColor4f(0, 1, 0, 1);
        glVertex2f(t1[0], t1[1]);
        glColor4f(0, 0, 1, 1);
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
      PhGUtils::message("rendering mesh.");
      for(int face_i = 0; face_i < mesh.NumFaces(); ++face_i) {
        auto normal_i = mesh.normal(face_i);
        auto f = mesh.face(face_i);
        auto v0 = mesh.vertex(f[0]), v1 = mesh.vertex(f[1]), v2 = mesh.vertex(f[2]);
        auto n = mesh.normal(face_i);
        unsigned char r, g, b;
        encode_index(face_i, r, g, b);
        int tmp_idx;
        assert(decode_index(r, g, b, tmp_idx) == face_i);

        glShadeModel(GL_FLAT);

        glBegin(GL_TRIANGLES);

        glNormal3f(n[0], n[1], n[2]);
        glColor4ub(r, g, b, 255);

        //glColor3f(1, 0, 0);
        glVertex3f(v0[0], v0[1], v0[2]);
        //glColor3f(0, 1, 0);
        glVertex3f(v1[0], v1[1], v1[2]);
        //glColor3f(0, 0, 1);
        glVertex3f(v2[0], v2[1], v2[2]);

        glEnd();
      }
      PhGUtils::message("done.");
      break;
    }
    case Normal: {
      PhGUtils::message("rendering mesh.");
      for(int face_i = 0; face_i < mesh.NumFaces(); ++face_i) {
        auto normal_i = mesh.normal(face_i);
        auto f = mesh.face(face_i);
        auto v0 = mesh.vertex(f[0]), v1 = mesh.vertex(f[1]), v2 = mesh.vertex(f[2]);
        auto n = mesh.normal(face_i);
        Vector3d nv0 = (mesh.vertex_normal(f[0]) + Vector3d(1.0, 1.0, 1.0)) * 0.5;
        Vector3d nv1 = (mesh.vertex_normal(f[1]) + Vector3d(1.0, 1.0, 1.0)) * 0.5;
        Vector3d nv2 = (mesh.vertex_normal(f[2]) + Vector3d(1.0, 1.0, 1.0)) * 0.5;

        glShadeModel(GL_SMOOTH);

        glBegin(GL_TRIANGLES);

        glNormal3f(n[0], n[1], n[2]);
        glColor3f(nv0[0], nv0[1], nv0[2]);
        glVertex3f(v0[0], v0[1], v0[2]);
        glColor3f(nv1[0], nv1[1], nv1[2]);
        glVertex3f(v1[0], v1[1], v1[2]);
        glColor3f(nv2[0], nv2[1], nv2[2]);
        glVertex3f(v2[0], v2[1], v2[2]);

        glEnd();
      }
      PhGUtils::message("done.");
      break;
    }
    case TexturedMesh: {
      PhGUtils::message("rendering mesh.");
      glEnable(GL_TEXTURE);

      GLuint image_tex;
      // generate texture
      glEnable(GL_TEXTURE_2D);
      glGenTextures(1, &image_tex);
      glBindTexture(GL_TEXTURE_2D, image_tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture.width(), texture.height(), 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, texture.bits());
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

      for(int face_i = 0; face_i < mesh.NumFaces(); ++face_i) {
        auto normal_i = mesh.normal(face_i);
        auto f = mesh.face(face_i);
        auto v0 = mesh.vertex(f[0]), v1 = mesh.vertex(f[1]), v2 = mesh.vertex(f[2]);
        auto n = mesh.normal(face_i);
        auto tf = mesh.face_texture(face_i);
        auto t0 = mesh.texture_coords(tf[0]), t1 = mesh.texture_coords(tf[1]), t2 = mesh.texture_coords(tf[2]);

        glShadeModel(GL_SMOOTH);

        glBegin(GL_TRIANGLES);

        glNormal3f(n[0], n[1], n[2]);

        glTexCoord2f(t0[0], 1.0-t0[1]); glVertex3f(v0[0], v0[1], v0[2]);
        glTexCoord2f(t1[0], 1.0-t1[1]); glVertex3f(v1[0], v1[1], v1[2]);
        glTexCoord2f(t2[0], 1.0-t2[1]); glVertex3f(v2[0], v2[1], v2[2]);

        glEnd();
      }
      PhGUtils::message("done.");
      break;
    }
  }

  // get the bitmap and save it as an image
  QImage img = fbo.toImage();
  fbo.release();
  return img;
}
