#ifndef FACESHAPEFROMSHADING_COST_FUNCTIONS_H
#define FACESHAPEFROMSHADING_COST_FUNCTIONS_H

#include <common.h>

#include "ceres/ceres.h"
#include "utils.h"

struct NormalMapDataTerm {
  NormalMapDataTerm(double Ir, double Ig, double Ib,
                    double ar, double ag, double ab,
                    const VectorXd& lighting_coeffs,
                    double weight = 1.0)
    : I(Vector3d(Ir, Ig, Ib)), a(Vector3d(ar, ag, ab)),
      lighting_coeffs(lighting_coeffs), weight(weight) {}

  ~NormalMapDataTerm() {}

  bool operator()(double const * const * parameters, double *residuals) const {
    double theta = parameters[0][0], phi = parameters[1][0];

    double nx, ny, nz;
    tie(nx, ny, nz) = sphericalcoords2normal<double>(theta, phi);

    VectorXd Y = sphericalharmonics(nx, ny, nz);

    double LdotY = lighting_coeffs.transpose() * Y;

    Vector3d f = I - a * LdotY;

    residuals[0] = f[0] * weight;
    residuals[1] = f[1] * weight;
    residuals[2] = f[2] * weight;

    return true;
  }

  Vector3d I, a;
  VectorXd lighting_coeffs;
  double weight;
};

struct NormalMapDataTerm_analytic : public ceres::CostFunction {
  NormalMapDataTerm_analytic(double Ir, double Ig, double Ib,
                             double ar, double ag, double ab,
                             const VectorXd& lighting_coeffs,
                             double weight = 1.0)
    : I(Vector3d(Ir, Ig, Ib)), a(Vector3d(ar, ag, ab)), lighting_coeffs(lighting_coeffs), weight(weight) {

    mutable_parameter_block_sizes()->clear();
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    set_num_residuals(3);
  }

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const
  {
    const int num_dof = 9;

    double theta = parameters[0][0], phi = parameters[1][0];

    double cosTheta = cos(theta), sinTheta = sin(theta);
    double cosPhi = cos(phi), sinPhi = sin(phi);

    double nx, ny, nz;
    tie(nx, ny, nz) = sphericalcoords2normal<double>(theta, phi);

    VectorXd Y = sphericalharmonics(nx, ny, nz);

    double LdotY = lighting_coeffs.transpose() * Y;

    Vector3d f = I - a * LdotY;

    residuals[0] = f[0] * weight;
    residuals[1] = f[1] * weight;
    residuals[2] = f[2] * weight;

    if (jacobians != NULL) {
      assert(jacobians[0] != NULL);
      assert(jacobians[1] != NULL);

      MatrixXd dYdnormal = dY_dnormal(nx, ny, nz);

      VectorXd dYdtheta = dYdnormal * dnormal_dtheta(theta, phi);
      VectorXd dYdphi = dYdnormal * dnormal_dphi(theta, phi);

      double LdotdYdtheta = lighting_coeffs.transpose() * dYdtheta;
      double LdotdYdphi = lighting_coeffs.transpose() * dYdphi;

      // jacobians[0][i] = \frac{\partial E}{\partial \theta}
      jacobians[0][0] = -a[0] * LdotdYdtheta * weight;
      jacobians[0][1] = -a[1] * LdotdYdtheta * weight;
      jacobians[0][2] = -a[2] * LdotdYdtheta * weight;

      // jacobians[1][i] = \frac{\partial E}{\partial \phi}
      jacobians[1][0] = -a[0] * LdotdYdphi * weight;
      jacobians[1][1] = -a[1] * LdotdYdphi * weight;
      jacobians[1][2] = -a[2] * LdotdYdphi * weight;
    }
    return true;
  }

  Vector3d I, a;
  VectorXd lighting_coeffs;
  double weight;
};

struct NormalMapIntegrabilityTerm {
  NormalMapIntegrabilityTerm(double dx, double dy, double weight) : dx(dx), dy(dy), weight(weight) {}

  double safe_division(double numer, double denom, double eps) const {
    if(fabs(denom) < eps) {
      denom = (denom<0)?-eps:eps;
    }
    return numer / denom;
  }

  double round_off(double val, double eps) const {
    if(fabs(val) < eps) {
      if(val < 0) return -eps;
      else return eps;
    } else return val;
  }

  bool operator()(double const * const * parameters, double *residuals) const {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_l = parameters[2][0], phi_l = parameters[3][0];
    double theta_u = parameters[4][0], phi_u = parameters[5][0];

    double nx, ny, nz;
    tie(nx, ny, nz) = sphericalcoords2normal<double>(theta, phi);

    double nx_l, ny_l, nz_l;
    tie(nx_l, ny_l, nz_l) = sphericalcoords2normal<double>(theta_l, phi_l);

    double nx_u, ny_u, nz_u;
    tie(nx_u, ny_u, nz_u) = sphericalcoords2normal<double>(theta_u, phi_u);

    if(weight == 0) {
      residuals[0] = 0;
    } else {
      nz = round_off(nz, 1e-16);
      nz_l = round_off(nz_l, 1e-16);
      nz_u = round_off(nz_u, 1e-16);

      double nxnz = nx / nz;
      double nxnz_u = nx_u / nz_u;
      double nynz = ny / nz;
      double nynz_l = ny_l / nz_l;

      residuals[0] = ((nxnz_u - nxnz) - (nynz - nynz_l)) * weight;
    }

    return true;
  }

  double dx, dy;
  double weight;
};

struct NormalMapIntegrabilityTerm_analytic : public ceres::CostFunction {
  NormalMapIntegrabilityTerm_analytic(double dx, double dy, double weight) : dx(dx), dy(dy), weight(weight) {
    mutable_parameter_block_sizes()->clear();
    for(int param_i=0;param_i<6;++param_i)
      mutable_parameter_block_sizes()->push_back(1);
    set_num_residuals(1);
  }

  double round_off(double val, double eps) const {
    if(fabs(val) < eps) {
      if(val < 0) return -eps;
      else return eps;
    } else return val;
  }

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const
  {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_l = parameters[2][0], phi_l = parameters[3][0];
    double theta_u = parameters[4][0], phi_u = parameters[5][0];

    double nx, ny, nz;
    tie(nx, ny, nz) = sphericalcoords2normal<double>(theta, phi);

    double nx_l, ny_l, nz_l;
    tie(nx_l, ny_l, nz_l) = sphericalcoords2normal<double>(theta_l, phi_l);

    double nx_u, ny_u, nz_u;
    tie(nx_u, ny_u, nz_u) = sphericalcoords2normal<double>(theta_u, phi_u);

    if(weight == 0) {
      residuals[0] = 0;
      if(jacobians != NULL) {
        for(int param_i=0;param_i<6;++param_i) {
          if(jacobians[param_i] != NULL) jacobians[param_i][0] = 0;
        }
      }
    } else {
      nz = round_off(nz, 1e-16);
      nz_l = round_off(nz_l, 1e-16);
      nz_u = round_off(nz_u, 1e-16);

      double nxnz = nx / nz;
      double nxnz_u = nx_u / nz_u;
      double nynz = ny / nz;
      double nynz_l = ny_l / nz_l;

      residuals[0] = ((nxnz_u - nxnz) - (nynz - nynz_l)) * weight;

      if(jacobians != NULL) {
        for(int param_i=0;param_i<6;++param_i) assert(jacobians[param_i] != NULL);

        {
          double nz2 = nz * nz + 1e-16;

          Vector3d dE_dn(-nz / nz2, -nz / nz2, (nx + ny) / nz2);

          Vector3d dn_dtheta = dnormal_dtheta(theta, phi);
          // jacobians[0][0] = \frac{\partial E}{\partial \theta}
          jacobians[0][0] = dE_dn.dot(dn_dtheta) * weight;

          Vector3d dn_dphi = dnormal_dphi(theta, phi);
          // jacobians[1][0] = \frac{\partial E}{\partial \phi}
          jacobians[1][0] = dE_dn.dot(dn_dphi) * weight;
        }

        {
          double nz_l2 = nz_l * nz_l + 1e-16;

          Vector3d dE_dn(0, nz_l / nz_l2, -ny_l / nz_l2);

          Vector3d dn_dtheta = dnormal_dtheta(theta_l, phi_l);
          // jacobians[2][0] = \frac{\partial E}{\partial \theta_l}
          jacobians[2][0] = dE_dn.dot(dn_dtheta) * weight;

          Vector3d dn_dphi = dnormal_dphi(theta_l, phi_l);
          // jacobians[3][0] = \frac{\partial E}{\partial \phi_l}
          jacobians[3][0] = dE_dn.dot(dn_dphi) * weight;
        }

        {
          double nz_u2 = nz_u * nz_u + 1e-16;

          Vector3d dE_dn(nz_u / nz_u2, 0, -nx_u / nz_u2);

          Vector3d dn_dtheta = dnormal_dtheta(theta_u, phi_u);
          // jacobians[4][0] = \frac{\partial E}{\partial \theta_u}
          jacobians[4][0] = dE_dn.dot(dn_dtheta)  * weight;

          Vector3d dn_dphi = dnormal_dphi(theta_u, phi_u);
          // jacobians[5][0] = \frac{\partial E}{\partial \phi_u}
          jacobians[5][0] = dE_dn.dot(dn_dphi) * weight;
        }
      }
    }

    return true;
  }

  double dx, dy;
  double weight;
};

struct NormalMapSmoothnessTerm {
  NormalMapSmoothnessTerm(double dx, double dy, double weight) : dx(dx), dy(dy), weight(weight) {}

  double safe_division(double numer, double denom, double eps) const {
    if(fabs(denom) < eps) {
      denom = (denom<0)?-eps:eps;
    }
    return numer / denom;
  }

  double round_off(double val, double eps) const {
    if(fabs(val) < eps) {
      if(val < 0) return -eps;
      else return eps;
    } else return val;
  }

  bool operator()(double const * const * parameters, double *residuals) const {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_l = parameters[2][0], phi_l = parameters[3][0];
    double theta_u = parameters[4][0], phi_u = parameters[5][0];

    double nx, ny, nz;
    tie(nx, ny, nz) = sphericalcoords2normal<double>(theta, phi);

    double nx_l, ny_l, nz_l;
    tie(nx_l, ny_l, nz_l) = sphericalcoords2normal<double>(theta_l, phi_l);

    double nx_u, ny_u, nz_u;
    tie(nx_u, ny_u, nz_u) = sphericalcoords2normal<double>(theta_u, phi_u);

    if(weight == 0) {
      residuals[0] = 0;
      residuals[1] = 0;
      residuals[2] = 0;
      residuals[3] = 0;
    } else {
      nz = round_off(nz, 1e-16);
      nz_l = round_off(nz_l, 1e-16);
      nz_u = round_off(nz_u, 1e-16);

      residuals[0] = (nx_u * nz - nx * nz_u) * weight;
      residuals[1] = (ny_u * nz - ny * nz_u) * weight;
      residuals[2] = (nx_l * nz - nx * nz_l) * weight;
      residuals[3] = (ny_l * nz - ny * nz_l) * weight;
    }

    return true;
  }

  double dx, dy;
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

      double nx, ny, nz;
      tie(nx, ny, nz) = sphericalcoords2normal<double>(theta, phi);

      normal_LoG += Vector3d(nx, ny, nz) * kval;
    }

    residuals[0] = (normal_LoG(0) - normal_ref_LoG(0)) * weight;
    residuals[1] = (normal_LoG(1) - normal_ref_LoG(1)) * weight;
    residuals[2] = (normal_LoG(2) - normal_ref_LoG(2)) * weight;

    return true;
  }

  vector<pair<int, double>> info;
  Vector3d normal_ref_LoG;
  double weight;
};

struct NormalMapRegularizationTerm_analytic : public ceres::CostFunction {
  NormalMapRegularizationTerm_analytic(const vector<pair<int, double>>& info,
                              const Vector3d& normal_ref_LoG,
                              double weight)
    : info(info), normal_ref_LoG(normal_ref_LoG), weight(weight) {
    mutable_parameter_block_sizes()->clear();
    for(int i=0;i<info.size()*2;++i) mutable_parameter_block_sizes()->push_back(1);
    set_num_residuals(3);
  }

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const {
    Vector3d normal_LoG(0, 0, 0);

    for(int i=0;i<info.size();++i) {
      auto& reginfo = info[i];
      double kval = reginfo.second;

      double theta = parameters[i*2][0], phi = parameters[i*2+1][0];

      double nx, ny, nz;
      tie(nx, ny, nz) = sphericalcoords2normal<double>(theta, phi);

      normal_LoG += Vector3d(nx, ny, nz) * kval;
    }

    residuals[0] = (normal_LoG(0) - normal_ref_LoG(0)) * weight;
    residuals[1] = (normal_LoG(1) - normal_ref_LoG(1)) * weight;
    residuals[2] = (normal_LoG(2) - normal_ref_LoG(2)) * weight;

    if (jacobians != NULL) {
      for(int i=0;i<info.size()*2;++i) assert(jacobians[i] != NULL);

      for(int i=0;i<info.size();++i) {
        double w = info[i].second;
        double theta = parameters[i*2][0], phi = parameters[i*2+1][0];

        Vector3d dn_dtheta = dnormal_dtheta(theta, phi);

        // dF_dtheta
        jacobians[i*2][0] = w * dn_dtheta[0] * weight;
        jacobians[i*2][1] = w * dn_dtheta[1] * weight;
        jacobians[i*2][2] = w * dn_dtheta[2] * weight;

        Vector3d dn_dphi = dnormal_dphi(theta, phi);

        // dF_dphi
        jacobians[i*2+1][0] = w * dn_dphi[0] * weight;
        jacobians[i*2+1][1] = w * dn_dphi[1] * weight;
        jacobians[i*2+1][2] = w * dn_dphi[2] * weight;
      }
    }

    return true;
  }

  vector<pair<int, double>> info;
  Vector3d normal_ref_LoG;
  double weight;
};

struct NormalMapAngleRegularizationTerm : public ceres::CostFunction {
  NormalMapAngleRegularizationTerm(double weight) : weight(weight) {
    mutable_parameter_block_sizes()->clear();
    mutable_parameter_block_sizes()->push_back(1);
    set_num_residuals(1);
  }

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const {
    residuals[0] = parameters[0][0] * weight;
    if (jacobians != NULL) {
      if( jacobians[0] != NULL ) {
        jacobians[0][0] = weight;
      }
    }
    return true;
  }

  double weight;
};

#endif  // FACESHAPEFROMSHADING_COST_FUNCTIONS_H
