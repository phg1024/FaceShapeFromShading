#ifndef FACESHAPEFROMSHADING_COST_FUNCTIONS_H
#define FACESHAPEFROMSHADING_COST_FUNCTIONS_H

#include <common.h>

#include "ceres/ceres.h"
#include "utils.h"

struct DepthMapDataTerm {
  DepthMapDataTerm(double Ir, double Ig, double Ib,
                   double ar, double ag, double ab,
                   const VectorXd& lighting_coeffs,
                   double dx, double dy,
                   double weight = 1.0)
    : I(Vector3d(Ir, Ig, Ib)), a(Vector3d(ar, ag, ab)),
      lighting_coeffs(lighting_coeffs), dx(dx), dy(dy), weight(weight) {}

  ~DepthMapDataTerm() {}

  bool operator()(double const * const * parameters, double *residuals) const {
    double z = parameters[0][0], z_l = parameters[1][0], z_u = parameters[2][0];

    double p = (z - z_l) / dx;
    double q = (z_u - z) / dy;

    double nx, ny, nz;

    double N = p * p + q * q + 1;
    nx = p / N; ny = q / N; nz = 1 / N;

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
  double dx, dy;
  double weight;
};

struct DepthMapIntegrabilityTerm {
  DepthMapIntegrabilityTerm(double dx, double dy, double weight = 1.0)
    : dx(dx), dy(dy), weight(weight) {}

  ~DepthMapIntegrabilityTerm() {}

  bool operator()(double const * const * parameters, double *residuals) const {
    double z = parameters[0][0], z_l = parameters[1][0];
    double z_u = parameters[2][0], z_ul = parameters[3][0];
    double z_uu = parameters[4][0], z_ll = parameters[5][0];

    const double epsilon = 1e-16;

    auto get_normal = [&epsilon](double z, double z_u, double z_l, double dx, double dy) {
      double p = (z - z_l) / dx;
      double q = (z_u - z) / dy;

      double nx, ny, nz;

      double N = p * p + q * q + 1;

      nx = p / N; ny = q / N; nz = 1 / N;
      return make_tuple(nx, ny, nz);
    };

    double nx, ny, nz; tie(nx, ny, nz) = get_normal(z, z_u, z_l, dx, dy);
    double nx_l, ny_l, nz_l; tie(nx_l, ny_l, nz_l) = get_normal(z_l, z_ul, z_ll, dx, dy);
    double nx_u, ny_u, nz_u; tie(nx_u, ny_u, nz_u) = get_normal(z_u, z_uu, z_ul, dx, dy);

    double denom = nz * nz_u * nz_l;
    double int_val = (nx_u * nz * nz_l - nx * nz_l * nz_u) - (ny * nz_l * nz_u - ny_l * nz * nz_u);

    if( fabs(denom) < epsilon) denom = denom>0?epsilon:-epsilon;
    int_val = int_val / epsilon;

    residuals[0] = int_val * weight;

    return true;
  }

  double dx, dy;
  double weight;
};

struct DepthMapRegularizationTerm {
  DepthMapRegularizationTerm(const vector<pair<int, double>>& info,
                             double z_ref_LoG,
                             double weight)
    : info(info), z_ref_LoG(z_ref_LoG), weight(weight) {}

  bool operator()(double const * const * parameters, double *residuals) const {
    double z_LoG = 0;

    for(int i=0;i<info.size();++i) {
      auto& reginfo = info[i];
      double kval = reginfo.second;

      double zval = parameters[i][0];

      z_LoG += zval * kval;
    }

    residuals[0] = (z_LoG - z_ref_LoG) * weight;

    return true;
  }

  vector<pair<int, double>> info;
  double z_ref_LoG;
  double weight;
};

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

    double tan_theta = tan(theta), sin_phi = sin(phi), cos_phi = cos(phi);
    double tan_theta_l = tan(theta_l), sin_phi_l = sin(phi_l), cos_phi_l = cos(phi_l);
    double tan_theta_u = tan(theta_u), sin_phi_u = sin(phi_u), cos_phi_u = cos(phi_u);

    if(weight == 0) {
      residuals[0] = 0;
    } else {
      if(fabs(theta) < 1e-3 && fabs(phi) < 1e-3) {
        // singular point
        residuals[0] = ((phi - phi_l) + (phi_u - phi) + (theta - theta_l) + (theta_u - theta)) * weight;
      } else {
        residuals[0] = ((tan_theta_u * sin_phi_u - tan_theta * sin_phi) -
                        (tan_theta * cos_phi - tan_theta_l * cos_phi_l)) * weight;
      }
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

    double tan_theta = tan(theta), sin_phi = sin(phi), cos_phi = cos(phi);
    double tan_theta_l = tan(theta_l), sin_phi_l = sin(phi_l), cos_phi_l = cos(phi_l);
    double tan_theta_u = tan(theta_u), sin_phi_u = sin(phi_u), cos_phi_u = cos(phi_u);

    if(weight == 0) {
      residuals[0] = 0;
      if(jacobians != NULL) {
        for(int param_i=0;param_i<6;++param_i) {
          if(jacobians[param_i] != NULL) jacobians[param_i][0] = 0;
        }
      }
    } else {
      if(fabs(theta) < 1e-3 && fabs(phi) < 1e-3) {
        // singular point
        residuals[0] = ((phi - phi_l) + (phi_u - phi) + (theta - theta_l) + (theta_u - theta)) * weight;
        if(jacobians != NULL) {
          for(int param_i=0;param_i<6;++param_i) assert(jacobians[param_i] != NULL);
          jacobians[0][0] = 0;
          jacobians[1][0] = 0;
          jacobians[2][0] = -1;
          jacobians[3][0] = -1;
          jacobians[4][0] = 1;
          jacobians[5][0] = 1;
        }
      } else {
        residuals[0] = ((tan_theta_u * sin_phi_u - tan_theta * sin_phi) -
                        (tan_theta * cos_phi - tan_theta_l * cos_phi_l)) * weight;

                        residuals[0] = ((tan_theta_u * sin_phi_u - tan_theta * sin_phi) -
                                        (tan_theta * cos_phi - tan_theta_l * cos_phi_l)) * weight;
        if(jacobians != NULL) {
          for(int param_i=0;param_i<6;++param_i) assert(jacobians[param_i] != NULL);

          {
            double dEdtheta = -(sin_phi + cos_phi) * 2.0 / max(cos(theta*2.0) + 1.0, 1e-16);
            // jacobians[0][0] = \frac{\partial E}{\partial \theta}
            jacobians[0][0] = dEdtheta * weight;

            double dEdphi = -tan_theta * (cos_phi - sin_phi);
            // jacobians[1][0] = \frac{\partial E}{\partial \phi}
            jacobians[1][0] = dEdphi * weight;
          }

          {
            double dEdtheta = cos_phi_l * 2.0 / max(cos(theta_l*2) + 1.0, 1e-16);
            // jacobians[2][0] = \frac{\partial E}{\partial \theta_l}
            jacobians[2][0] = dEdtheta * weight;

            double dEdphi = -tan_theta_l * sin_phi_l;
            // jacobians[3][0] = \frac{\partial E}{\partial \phi_l}
            jacobians[3][0] = dEdphi * weight;
          }

          {
            double dEdtheta = sin_phi_u * 2.0 / max(cos(theta_u*2.0) + 1.0, 1e-16);
            // jacobians[4][0] = \frac{\partial E}{\partial \theta_u}
            jacobians[4][0] = dEdtheta  * weight;

            double dEdphi = cos_phi_u * tan_theta_u;
            // jacobians[5][0] = \frac{\partial E}{\partial \phi_u}
            jacobians[5][0] = dEdphi * weight;
          }
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
