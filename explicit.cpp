#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2

#include <MultilinearReconstruction/basicmesh.h>
#include "Geometry/geometryutils.hpp"

#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

typedef CGAL::Simple_cartesian<double> K;
typedef K::FT FT;
typedef K::Line_3 Line;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::vector<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm
  double x, y, z;                  // position, also color (r,g,b)
  Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; }
  Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); }
  Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); }
  Vec operator*(double b) const { return Vec(x*b,y*b,z*b); }
  Vec mult(const Vec &b) const { return Vec(x*b.x,y*b.y,z*b.z); }
  Vec& norm(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
  double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // cross:
  Vec operator%(Vec&b){return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
  friend ostream& operator<<(ostream& os, Vec v);
};
inline ostream& operator<<(ostream& os, Vec v) {
  os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
  return os;
}
struct Ray { Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} };
enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

struct Object {
  enum Type {
    SPHERE = 0,
    MESH,
  };
  Object(Type t) : type(t) {}

  Type type;
};

struct Sphere : Object{
  double rad;       // radius
  Vec p, e, c;      // position, emission, color
  Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)
  Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_):
    Object(Object::SPHERE),
    rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {}
  double intersect(const Ray &r) const { // returns distance, 0 if nohit
    Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    double t, eps=1e-4, b=op.dot(r.d), det=b*b-op.dot(op)+rad*rad;
    if (det<0) return 0; else det=sqrt(det);
    return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0);
  }
};

Vec compute_barycentric_coordinates(Vec p, Vec q1, Vec q2, Vec q3) {
	Vec e23 = q3 - q2, e21 = q1 - q2;
	Vec d1 = q1 - p, d2 = q2 - p, d3 = q3 - p;
	Vec oriN = e23 % e21;
	Vec n = oriN;
  n.norm();

	double invBTN = 1.0 / n.dot(oriN);
	Vec bcoord;
	bcoord.x = n.dot(d2 % d3) * invBTN;
	bcoord.y = n.dot(d3 % d1) * invBTN;
	bcoord.z = 1 - bcoord.x - bcoord.y;

	return bcoord;
}

int good_count = 0;

bool rayIntersectsTriangle(Ray r, Vec v0, Vec v1, Vec v2,
                           double& t) {
	Vec e1 = v1 - v0;
	Vec e2 = v2 - v0;
  Vec N = (e1 % e2).norm();

  double a2 = sqrt(N.dot(N));

  double NdotD = N.dot(r.d);

  if(fabs(NdotD) < 1e-16) {
    //cout << "Failed at NdotD" << endl;
    return false;
  }

  //cout << "N = " << N << endl;
  //cout << "r.o = " << r.o << endl;
  //cout << "v0 = " << v0 << endl;
  t = -N.dot(r.o - v0) / NdotD;
  if(t < 0) {
    //cout << "Failed at t<0" << endl;
    //t = 1000.0;
    return false;
  }

  //t -= min(0.5 * t, 1e-3);

  // compute the intersection point using equation 1
  Vec P = r.o + r.d * t;

  // Step 2: inside-outside test
  Vec C; // vector perpendicular to triangle's plane

  // edge 0
  Vec vp0 = P - v0;
  C = e1 % vp0;
  if (N.dot(C) < 0) {
    //cout << "Failed at e0" << endl;
    return false; // P is on the right side
  }

  // edge 1
  Vec e3 = v2 - v1;
  Vec vp1 = P - v1;
  C = e3 % vp1;
  if (N.dot(C) < 0)  {
    //cout << "Failed at e1" << endl;
    return false; // P is on the right side
  }

  // edge 2
  Vec vp2 = P - v2;
  C = e2 % vp2;
  if (N.dot(C) > 0) {
    //cout << "Failed at e2" << endl;
    return false; // P is on the right side;
  }

  t -= 1e-3;
  ++good_count;
  return true; // this ray hits the triangle
}

struct Mesh : Object {
  Mesh() : Object(Object::MESH) {}
  Mesh(const string& filename) : Object(Object::MESH), mesh(filename) {
    mesh.ComputeNormals();
  }
  void buildTree(double scale, Point translation) {
    int nfaces = mesh.NumFaces();
    triangles.reserve(nfaces);
    face_indices_map.resize(nfaces);
    for(int i=0,ioffset=0;i<nfaces;++i) {
      face_indices_map[i] = i;
      auto face_i = mesh.face(i);
      int v1 = face_i[0], v2 = face_i[1], v3 = face_i[2];
      auto p1 = mesh.vertex(v1), p2 = mesh.vertex(v2), p3 = mesh.vertex(v3);
      Point a(p1[0] * scale + translation.x(), p1[1] * scale + translation.y(), p1[2] * scale + translation.z());
      Point b(p2[0] * scale + translation.x(), p2[1] * scale + translation.y(), p2[2] * scale + translation.z());
      Point c(p3[0] * scale + translation.x(), p3[1] * scale + translation.y(), p3[2] * scale + translation.z());

      triangles.push_back(Triangle(a, b, c));
    }

    tree.reset(new Tree(triangles.begin(), triangles.end()));
    tree->accelerate_distance_queries();
  }

  bool intersect(const Ray &r, double& t, Vec& n, Vec& nl, Vec& f) const { // returns distance, 0 if nohit
    K::Ray_3 ray(Point(r.o.x, r.o.y, r.o.z), K::Direction_3(K::Vector_3(r.d.x, r.d.y, r.d.z)));
    vector<Primitive::Id> hits;
    tree->all_intersected_primitives(ray, back_inserter(hits));
    if(hits.empty()) return false;

    Vector3d dir(r.d.x, r.d.y, r.d.z);
    Vector3d ori(r.o.x, r.o.y, r.o.z);

    bool has_intersection = false;
    //cout <<  hits.size() << endl;
    double inf = 1e10;
    t = inf;

    for(int i=0;i<hits.size();++i) {
      int tidx = face_indices_map[hits[i] - triangles.begin()];
      auto face_t = mesh.face(tidx);
      int v0idx = face_t[0], v1idx = face_t[1], v2idx = face_t[2];

      auto tri = triangles[hits[i] - triangles.begin()];

      auto v00 = tri[0];
      auto v10 = tri[1];
      auto v20 = tri[2];

      Vec v0(v00.x(), v00.y(), v00.z());
      Vec v1(v10.x(), v10.y(), v10.z());
      Vec v2(v20.x(), v20.y(), v20.z());

      double ti;
      if(rayIntersectsTriangle(r, v0, v1, v2, ti)) {
        has_intersection = true;
        Vec x = r.o + r.d * ti;
        Vec bc = compute_barycentric_coordinates(x, v0, v1, v2);

        if( t > ti ) {
          t = min(t, ti);
          auto n0 = mesh.vertex_normal(v0idx);
          auto n1 = mesh.vertex_normal(v1idx);
          auto n2 = mesh.vertex_normal(v2idx);

          Vector3d nn = n0 * bc.x + n1 * bc.y + n2 * bc.z;
          nn.normalize();
          n=Vec(nn[0], nn[1], nn[2]);
          nl=n.dot(r.d)<0?n:n*-1;
        }

        f = Vec(0.75, 0.5, 0.5);
      }
    }

    return has_intersection;
  }

  BasicMesh mesh;
  shared_ptr<Tree> tree;
  std::vector<Triangle> triangles;
  vector<int> face_indices_map;
};

Sphere spheres[] = {//Scene: radius, position, emission, color, material
  //Sphere(1e5, Vec( 1e5+1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left
  //Sphere(1e5, Vec(-1e5+99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght
  //Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back
  //Sphere(1e5, Vec(50,40.8,-1e5+170), Vec(),Vec(),           DIFF),//Frnt
  //Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm
  //Sphere(1e5, Vec(50,-1e5+81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top
  //Sphere(600, Vec(50,681.6-.27,81.6),Vec(12,12,12),  Vec(), DIFF) //Lite
  //Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,.5,.5)*.999, DIFF),//Glas
  //Sphere(10.0, Vec(27,43,47),       Vec(),Vec(.25,.25, .95)*.999, DIFF),//Mirr
  //Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(.75,.75, .75)*.999, DIFF),//Mirr
  //Sphere(1e5, Vec(0,-1e5,0),         Vec(),Vec(.25,0.5,0.25)*.999, DIFF),
  //Sphere(1e5, Vec(0,0,-1e5),         Vec(),Vec(0.5, 0.5,0.25)*.999, DIFF),
};

Mesh mesh;

inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }
inline int toInt(double x){ return int(pow(clamp(x),1/2.2)*255+.5); }
inline bool intersect_spheres(const Ray &r, double &t, int &id){
  double n=sizeof(spheres)/sizeof(Sphere), d, inf=t=1e20;
  for(int i=int(n);i--;) if((d=spheres[i].intersect(r))&&d<t){t=d;id=i;}
  return t<inf;
}
const double light_coeffs[] = {
  0.25, 1.5, 0.25, 0.125, 0.1, -0.2, 0.3, 1.03, -0.57,
  //0.7202, 0.9954, 0.1494, 0.2478, 0.0803, 0.1102, 0.0208, 0.4304, -0.2894
};

enum RenderingMode {
  GlobalIllumination = 0,
  Lambertian,
  Normal
};

RenderingMode render_mode = RenderingMode::GlobalIllumination;

Vec sample_light(const Ray& r) {
  double nx = r.d.x, ny = r.d.y, nz = r.d.z;
  double L = light_coeffs[0];
  L += light_coeffs[1] * nx;
  L += light_coeffs[2] * ny;
  L += light_coeffs[3] * nz;
  L += light_coeffs[4] * nx * ny;
  L += light_coeffs[5] * nx * nz;
  L += light_coeffs[6] * ny * nz;
  L += light_coeffs[7] * (nx * nx - ny * ny);
  L += light_coeffs[8] * (3 * nz * nz - 1);
  L = (L>0)?L:-0.12345;

  return Vec(L, L, L);
}

struct path_info {
  path_info() : valid(false) {}
  path_info(int idx) : valid(false), pixel_id(idx), c(Vec(1, 1, 1)) {}

  bool valid;
  Vec light_dir;
  int pixel_id;
  Vec c;
  Vec cfinal;
};

Vec radiance(const Ray &r, int depth, unsigned short *Xi, path_info& info){
  switch(render_mode) {
    case GlobalIllumination: {
      double t;                               // distance to intersection
      int id=0;                               // id of intersected object
      double mesh_t=0;

      Vec n, nl, f;
      bool intersects_mesh = mesh.intersect(r, mesh_t, n, nl, f);
      //bool intersects_spheres = intersect_spheres(r, t, id);
      bool intersects_spheres = false;

      //if(intersects_mesh) return (n + Vec(1, 1, 1)).mult(Vec(.5, .5, .5));

      /*
      if ((intersects_spheres && !intersects_mesh) || (intersects_spheres && intersects_mesh && t < mesh_t)) {
        const Sphere &obj = spheres[id];        // the hit object
        Vec x=r.o+r.d*t;
        n=(x-obj.p).norm(), nl=n.dot(r.d)<0?n:n*-1, f=obj.c;
        double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl
        if (++depth>5) if (erand48(Xi)<p) f=f*(1/p); else return obj.e; //R.R.
        if (obj.refl == DIFF){                  // Ideal DIFFUSE reflection
          double r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2);
          Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u;
          Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
          return obj.e + f.mult(radiance(Ray(x,d),depth,Xi, info));
        }
      }else */if(intersects_mesh) {
        info.valid = true;
        Vec x=r.o+r.d*(mesh_t - 1e-3);
        double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl
        if (++depth>5) {
          // maximum depth is 5
          if (erand48(Xi)<p) f=f*(1/p);
          else {
            info.valid = false;
            return Vec(); //R.R.
          }
        }

        double r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2);
        Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u;
        Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
        info.c = info.c.mult(f);
        return f.mult(radiance(Ray(x,d),depth,Xi, info));
      }
      if(depth == 0) {
        info.valid = false;
        return Vec();
      } else {
        info.valid = true;
        info.light_dir = r.d;
        return sample_light(r);
      }

    #if 0
      else if (obj.refl == SPEC)            // Ideal SPECULAR reflection
        return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi,info));

      Ray reflRay(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION
      bool into = n.dot(nl)>0;                // Ray from outside going in?
      double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.d.dot(nl), cos2t;
      if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection
        return obj.e + f.mult(radiance(reflRay,depth,Xi,info));
      Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
      double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n));
      double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P);
      return obj.e + f.mult(depth>2 ? (erand48(Xi)<P ?   // Russian roulette
        radiance(reflRay,depth,Xi,info)*RP:radiance(Ray(x,tdir),depth,Xi,info)*TP) :
        radiance(reflRay,depth,Xi,info)*Re+radiance(Ray(x,tdir),depth,Xi,info)*Tr);
    #endif
      break;
    }
    case Lambertian: {
      double t;                               // distance to intersection
      int id=0;                               // id of intersected object
      double mesh_t=0;

      Vec n, nl, f;
      bool intersects_mesh = mesh.intersect(r, mesh_t, n, nl, f);
      bool intersects_spheres = intersect_spheres(r, t, id);

      //if(intersects_mesh) return (n + Vec(1, 1, 1)).mult(Vec(.5, .5, .5));

      if ((intersects_spheres && !intersects_mesh) || (intersects_spheres && intersects_mesh && t < mesh_t)) {
        const Sphere &obj = spheres[id];        // the hit object
        Vec x=r.o+r.d*t;
        n=(x-obj.p).norm(), nl=n.dot(r.d)<0?n:n*-1, f=obj.c;
        double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl
        if (++depth>2) if (erand48(Xi)<p) f=f*(1/p); else return obj.e; //R.R.
        if (obj.refl == DIFF){                  // Ideal DIFFUSE reflection
          double r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2);
          Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u;
          Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
          return obj.e + f.mult(radiance(Ray(x,d),depth,Xi,info));
        }
      }else if(intersects_mesh) {
        Vec x=r.o+r.d*(mesh_t - 1e-3);
        double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl
        if (++depth>5) if (erand48(Xi)<p) f=f*(1/p); else return Vec(); //R.R.

        /*
        double r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2);
        Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u;
        Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
        */

        Vec d = r.d - nl * 2.0 * r.d.dot(nl);

        return f.mult(sample_light(Ray(x, d)));
      }
      if(depth == 0) return Vec();
      else return sample_light(r);
      break;
    }
    case Normal: {
      double t;                               // distance to intersection
      int id=0;                               // id of intersected object
      double mesh_t=0;

      Vec n, nl, f;
      bool intersects_mesh = mesh.intersect(r, mesh_t, n, nl, f);
      bool intersects_spheres = intersect_spheres(r, t, id);

      if(intersects_mesh) return (n + Vec(1, 1, 1)).mult(Vec(.5, .5, .5));
      else return Vec();
    }
  }
}

int main(int argc, char *argv[]){
  int w=640, h=480, samps = argc>=2 ? atoi(argv[1])/4 : 1; // # samples
  if(argc>2) {
    mesh = Mesh(argv[2]);
    mesh.buildTree(25.0, Point(45,40,55));
  }

  if(argc>3) {
    render_mode = RenderingMode(atoi(argv[3]));
  }

  vector<vector<path_info>> traces;
  traces.resize(w*h, vector<path_info>());

  Ray cam(Vec(50,52,295.6), Vec(0,-0.042612,-1).norm()); // cam pos, dir
  Vec cx=Vec(w*.5135/h), cy=(cx%cam.d).norm()*.5135, r, *c=new Vec[w*h];
#pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP
  for (int y=0; y<h; y++){                       // Loop over image rows
    fprintf(stderr,"\rRendering (%d spp) %5.2f%%",samps*4,100.*y/(h-1));
    for (unsigned short x=0, Xi[3]={0,0,y*y*y}; x<w; x++)   // Loop cols
      for (int sy=0, i=(h-y-1)*w+x; sy<2; sy++)     // 2x2 subpixel rows
        for (int sx=0; sx<2; sx++, r=Vec()){        // 2x2 subpixel cols
          for (int s=0; s<samps; s++){
            path_info info(x*h+(h-y));
            double r1=2*erand48(Xi), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
            double r2=2*erand48(Xi), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);
            Vec d = cx*( ( (sx+.5 + dx)/2 + x)/w - .5) +
                    cy*( ( (sy+.5 + dy)/2 + y)/h - .5) + cam.d;
            Vec c_cur = radiance(Ray(cam.o,d.norm()),0,Xi, info)*(1./samps);
            r = r + c_cur;

            if(info.valid) {
              info.cfinal = c_cur;
              traces[info.pixel_id].push_back(info);
            }
          } // Camera rays are pushed ^^^^^ forward to start in interior
          c[i] = c[i] + Vec(clamp(r.x),clamp(r.y),clamp(r.z))*.25;
        }
  }
  FILE *f = fopen(argv[4], "w");         // Write image to PPM file.
  fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
  for (int i=0; i<w*h; i++)
    fprintf(f,"%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));


  // write the path traces
  cout << traces.size() << endl;
  FILE *fp = fopen("path.bin", "wb");
  int nsamples = samps*4;
  fwrite(&nsamples, sizeof(int), 1, fp);

  int ntraces = 0;
  for(int i=0;i<traces.size();++i) {
    if(traces[i].empty()) continue;
    else ++ntraces;
  }
  fwrite(&ntraces, sizeof(int), 1, fp);

  vector<int> traces_id;
  vector<int> traces_size;
  for(int i=0;i<traces.size();++i) {
    if(traces[i].empty()) continue;
    traces_id.push_back(i);
    traces_size.push_back(traces[i].size());
  }
  fwrite(&traces_id[0], sizeof(int), ntraces, fp);
  fwrite(&traces_size[0], sizeof(int), ntraces, fp);

  vector<float> traces_data;
  for(int i=0;i<traces.size();++i) {
    for(auto& p : traces[i]) {
      float v[] = {p.c.x, p.c.y, p.c.z,
                   p.light_dir.x, p.light_dir.y, p.light_dir.z,
                   p.cfinal.x, p.cfinal.y, p.cfinal.z};
      traces_data.insert(traces_data.end(), v, v+9);
    }
  }
  fwrite(&traces_data[0], sizeof(float), traces_data.size(), fp);
}
