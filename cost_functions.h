#ifndef FACESHAPEFROMSHADING_COST_FUNCTIONS_H
#define FACESHAPEFROMSHADING_COST_FUNCTIONS_H

#include <common.h>

struct NormalMapDataTerm {
  NormalMapDataTerm(int cons_idx,
                    double Ir, double Ig, double Ib,
                    double ar, double ag, double ab,
                    const VectorXd& lighting_coeffs)
    : cons_idx(cons_idx), Ir(Ir), Ig(Ig), Ib(Ib), ar(ar), ag(ag), ab(ab),
      lighting_coeffs(lighting_coeffs) {
  }

  ~NormalMapDataTerm() {
  }

  bool operator()(double const * const * parameters, double *residuals) const {
    const int num_dof = 9;

    double theta = parameters[0][0], phi = parameters[1][0];
    double nx = cos(theta) * sin(phi);
    double ny = sin(theta) * sin(phi);
    double nz = cos(phi);

    VectorXd Y(num_dof);
    Y(0) = 1;
    Y(1) = nx; Y(2) = ny; Y(3) = nz;
    Y(4) = nx * ny; Y(5) = nx * nz; Y(6) = ny * nz;
    Y(7) = nx * nx - ny * ny;
    Y(8) = 3.0 * nz * nz - 1.0;

    double LdotY = lighting_coeffs.transpose() * Y;

    double dr = Ir - ar * LdotY;
    double dg = Ig - ag * LdotY;
    double db = Ib - ab * LdotY;

    residuals[0] = sqrt(dr*dr+dg*dg+db*db);

    return true;
  }

  int cons_idx;
  double Ir, Ig, Ib, ar, ag, ab;
  VectorXd lighting_coeffs;
};

struct NormalMapIntegrabilityTerm {
  NormalMapIntegrabilityTerm(double weight) : weight(weight) {}

  bool operator()(double const * const * parameters, double *residuals) const {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_r = parameters[2][0], phi_r = parameters[3][0];
    double theta_d = parameters[4][0], phi_d = parameters[5][0];

    double nx = cos(theta) * sin(phi);
    double ny = sin(theta) * sin(phi);
    double nz = cos(phi);

    double nx_r = cos(theta_r) * sin(phi_r);
    double ny_r = sin(theta_r) * sin(phi_r);
    double nz_r = cos(phi_r);

    double nx_d = cos(theta_d) * sin(phi_d);
    double ny_d = sin(theta_d) * sin(phi_d);
    double nz_d= cos(phi_d);

    const double epsilon = 1e-6;
    double nxnz = nx / (nz + epsilon);
    double nynz = ny / (nz + epsilon);

    double nynz_r = ny_r / (nz_r + epsilon);

    double nxnz_d = nx_d / (nz_d + epsilon);

    residuals[0] = ((nxnz_d - nxnz) - (nynz_r - nynz)) * weight;

    return true;
  }

  double weight;
};

struct NormalMapRegularizationTerm {
  NormalMapRegularizationTerm(const vector<pair<int, double>>& info,
                              const Vector3d& normal_ref_LoG,
                              double weight)
    : info(info), normal_ref_LoG(normal_ref_LoG), weight(weight) {}

  bool operator()(double const * const * parameters, double *residuals) const {
    Vector3d normal_LoG(0, 0, 0);

    for(int i=0;i<info.size();++i) {
      auto& reginfo = info[i];
      double kval = reginfo.second;

      double theta = parameters[i*2][0], phi = parameters[i*2+1][0];
      double nx = cos(theta) * sin(phi);
      double ny = sin(theta) * sin(phi);
      double nz = cos(phi);

      normal_LoG += Vector3d(nx, ny, nz) * kval;
    }

    double dr = normal_LoG(0) - normal_ref_LoG(0);
    double dg = normal_LoG(1) - normal_ref_LoG(1);
    double db = normal_LoG(2) - normal_ref_LoG(2);

    residuals[0] = sqrt(dr*dr+dg*dg+db*db);

    return true;
  }

  vector<pair<int, double>> info;
  Vector3d normal_ref_LoG;
  double weight;
};

#endif  // FACESHAPEFROMSHADING_COST_FUNCTIONS_H
