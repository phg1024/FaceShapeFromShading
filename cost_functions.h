#ifndef FACESHAPEFROMSHADING_COST_FUNCTIONS_H
#define FACESHAPEFROMSHADING_COST_FUNCTIONS_H

#include <common.h>

#include "ceres/ceres.h"

struct NormalMapDataTerm {
  NormalMapDataTerm(double Ir, double Ig, double Ib,
                    double ar, double ag, double ab,
                    const VectorXd& lighting_coeffs,
                    double weight = 1.0)
    : Ir(Ir), Ig(Ig), Ib(Ib), ar(ar), ag(ag), ab(ab),
      lighting_coeffs(lighting_coeffs), weight(weight) {}

  ~NormalMapDataTerm() {}

  bool operator()(double const * const * parameters, double *residuals) const {
    const int num_dof = 9;

    double theta = parameters[0][0], phi = parameters[1][0];

    // nx = cos(theta)
    // ny = sin(theta) * cos(phi)
    // nz = sin(theta) * sin(phi)

    double nx = cos(theta);
    double ny = sin(theta) * cos(phi);
    double nz = sin(theta) * sin(phi);

    VectorXd Y(num_dof);
    Y(0) = 1;
    Y(1) = nx; Y(2) = ny; Y(3) = nz;
    Y(4) = nx * ny; Y(5) = nx * nz; Y(6) = ny * nz;
    Y(7) = nx * nx - ny * ny;
    Y(8) = 3.0 * nz * nz - 1.0;

    double LdotY = lighting_coeffs.transpose() * Y;

    residuals[0] = (Ir - ar * LdotY) * weight;
    residuals[1] = (Ig - ag * LdotY) * weight;
    residuals[2] = (Ib - ab * LdotY) * weight;

    return true;
  }

  double Ir, Ig, Ib, ar, ag, ab;
  VectorXd lighting_coeffs;
  double weight;
};

struct NormalMapDataTerm_analytic : public ceres::CostFunction {
  NormalMapDataTerm_analytic(double Ir, double Ig, double Ib,
                             double ar, double ag, double ab,
                             const VectorXd& lighting_coeffs,
                             double weight = 1.0)
    : Ir(Ir), Ig(Ig), Ib(Ib), ar(ar), ag(ag), ab(ab), lighting_coeffs(lighting_coeffs), weight(weight) {

    mutable_parameter_block_sizes()->clear();
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    set_num_residuals(3);
  }

  VectorXd computeY(double theta, double phi) const {
    double sinTheta = sin(theta), cosTheta = cos(theta);
    double sinPhi = sin(phi), cosPhi = cos(phi);

    double nz = cosTheta * sinPhi;
    double nx = sinTheta * sinPhi;
    double ny = cosPhi;

    const int num_dof = 9;
    VectorXd Y(num_dof);
    Y(0) = 1;
    Y(1) = nx; Y(2) = ny; Y(3) = nz;
    Y(4) = nx * ny; Y(5) = nx * nz; Y(6) = ny * nz;
    Y(7) = nx * nx - ny * ny;
    Y(8) = 3.0 * nz * nz - 1.0;
    return Y;
  }

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const
  {
    const int num_dof = 9;

    double theta = parameters[0][0], phi = parameters[1][0];

    // nx = cos(theta)
    // ny = sin(theta) * cos(phi)
    // nz = sin(theta) * sin(phi)

    double cosTheta = cos(theta), sinTheta = sin(theta);
    double cosPhi = cos(phi), sinPhi = sin(phi);

    double nx = cosTheta;
    double ny = sinTheta * cosPhi;
    double nz = sinTheta * sinPhi;

    VectorXd Y(num_dof);
    Y(0) = 1;
    Y(1) = nx; Y(2) = ny; Y(3) = nz;
    Y(4) = nx * ny; Y(5) = nx * nz; Y(6) = ny * nz;
    Y(7) = nx * nx - ny * ny;
    Y(8) = 3.0 * nz * nz - 1.0;

    double LdotY = lighting_coeffs.transpose() * Y;
    LdotY = max(LdotY, 0.0);

    residuals[0] = (Ir - ar * LdotY) * weight;
    residuals[1] = (Ig - ag * LdotY) * weight;
    residuals[2] = (Ib - ab * LdotY) * weight;

    if (jacobians != NULL) {
      assert(jacobians[0] != NULL);
      assert(jacobians[1] != NULL);

#if 1
      VectorXd dYdtheta(num_dof);

      double sin2Theta = sin(2 * theta);
      double cos2Theta = cos(2 * theta);

      dYdtheta(0) = 0;
      dYdtheta(1) = -sinTheta;
      dYdtheta(2) = cosTheta * cosPhi;
      dYdtheta(3) = cosTheta * sinPhi;
      dYdtheta(4) = cos2Theta * cosPhi;
      dYdtheta(5) = cos2Theta * sinPhi;
      dYdtheta(6) = sin2Theta * sinPhi * cosPhi;
      dYdtheta(7) = -sin2Theta * ( 1 + cosPhi * cosPhi);
      dYdtheta(8) = 3 * sin2Theta * sinPhi * sinPhi;

      VectorXd dYdphi(num_dof);

      double sin2Phi = sin(2 * phi);
      double cos2Phi = cos(2 * phi);

      dYdphi(0) = 0;
      dYdphi(1) = 0;
      dYdphi(2) = -sinTheta * sinPhi;
      dYdphi(3) = sinTheta * cosPhi;
      dYdphi(4) = -sinTheta * cosTheta * sinPhi;
      dYdphi(5) = sinTheta * cosTheta * cosPhi;
      dYdphi(6) = sinTheta * sinTheta * cos2Phi;
      dYdphi(7) = sinTheta * sinTheta * sin2Phi;
      dYdphi(8) = 3 * sinTheta * sinTheta * sin2Phi;
#else
      const double eps = 1e-6;
      VectorXd dYdtheta = ((computeY(theta+eps, phi) - Y)/eps).eval();
      VectorXd dYdphi = ((computeY(theta, phi+eps) - Y)/eps).eval();
#endif

      double LdotdYdtheta = lighting_coeffs.transpose() * dYdtheta;
      double LdotdYdphi = lighting_coeffs.transpose() * dYdphi;

      // jacobians[i][0] = \frac{\partial E}{\partial \theta}
      jacobians[0][0] = -ar * LdotdYdtheta;
      jacobians[0][1] = -ag * LdotdYdtheta;
      jacobians[0][2] = -ab * LdotdYdtheta;

      // jacobians[i][1] = \frac{\partial E}{\partial \phi}
      jacobians[1][0] = -ar * LdotdYdphi;
      jacobians[1][1] = -ag * LdotdYdphi;
      jacobians[1][2] = -ab * LdotdYdphi;
    }
    return true;
  }

  double Ir, Ig, Ib, ar, ag, ab;
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

    // nx = cos(theta)
    // ny = sin(theta) * cos(phi)
    // nz = sin(theta) * sin(phi)

    double nx = cos(theta);
    double ny = sin(theta) * cos(phi);
    double nz = sin(theta) * sin(phi);

    double nx_l = cos(theta_l);
    double ny_l = sin(theta_l) * cos(phi_l);
    double nz_l = sin(theta_l) * sin(phi_l);

    double nx_u = cos(theta_u);
    double ny_u = sin(theta_u) * cos(phi_u);
    double nz_u = sin(theta_u) * sin(phi_u);

#if 1
    if(weight == 0) residuals[0] = 0;
    else {
      double nz2 = nz * nz + 1e-16;
      double part1 = (nx_u * nz - nz_u * nx) * max(fabs(dy), 1e-8);
      double part2 = (nz_l * ny - ny_l * nz) * max(fabs(dx), 1e-8);

      residuals[0] = (part1 - part2) / nz2 * weight;
    }
#else
    Vector3d n(nx, ny, nz);
    Vector3d nu(nx_u, ny_u, nz_u);
    Vector3d nl(nx_l, ny_l, nz_l);
    Vector3d dndy = nu - n;
    Vector3d dndx = n - nl;

    double nz2 = nz * nz + 1e-16;

    const Vector3d xvec(1, 0, 0), yvec(0, 1, 0);
    residuals[0] = n.dot(dndy.cross(yvec)+dndx.cross(xvec)) / nz2 * weight;
#endif

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

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const
  {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_l = parameters[2][0], phi_l = parameters[3][0];
    double theta_u = parameters[4][0], phi_u = parameters[5][0];

    // nx = cos(theta)
    // ny = sin(theta) * cos(phi)
    // nz = sin(theta) * sin(phi)

    double cosTheta = cos(theta), sinTheta = sin(theta);
    double cosPhi = cos(phi), sinPhi = sin(phi);

    double nx = cosTheta;
    double ny = sinTheta * cosPhi;
    double nz = sinTheta * sinPhi;

    double cosTheta_l = cos(theta_l), sinTheta_l = sin(theta_l);
    double cosPhi_l = cos(phi_l), sinPhi_l = sin(phi_l);

    double nx_l = cosTheta_l;
    double ny_l = sinTheta_l * cosPhi_l;
    double nz_l = sinTheta_l * sinPhi_l;

    double cosTheta_u = cos(theta_u), sinTheta_u = sin(theta_u);
    double cosPhi_u = cos(phi_u), sinPhi_u = sin(phi_u);

    double nx_u = cosTheta_u;
    double ny_u = sinTheta_u * cosPhi_u;
    double nz_u = sinTheta_u * sinPhi_u;

    /*
    double nxnz = nx / nz;
    double nynz = ny / nz;
    double nxnz_u = nx_u / nz_u;
    double nynz_l = ny_l / nz_l;

    residuals[0] = (nxnz_u - nxnz - (nynz - nynz_l)) * weight;
    */

    double nz2 = nz * nz;
    double Du = (nx_u * nz - nz_u * nx) * max(fabs(dy), 1e-8);
    double Dl = (nz_l * ny - ny_l * nz) * max(fabs(dx), 1e-8);

    residuals[0] = (Du - Dl) / (nz2 + 1e-16) * weight;

    if (jacobians != NULL) {
      for(int param_i=0;param_i<6;++param_i) assert(jacobians[0] != NULL);

      double nz4 = nz2 * nz2;

      {
        double dnx_dtheta = -sinTheta;
        double dny_dtheta = cosTheta * cosPhi;
        double dnz_dtheta = cosTheta * sinPhi;

        double dnx_dphi = 0;
        double dny_dphi = -sinTheta * sinPhi;
        double dnz_dphi = sinTheta * cosPhi;

        // jacobians[0][0] = \frac{\partial E}{\partial \theta}
        double dDu_dtheta = nx_u * dnz_dtheta - nz_u * dnx_dtheta;
        double dDl_dtheta = nz_l * dny_dtheta - ny_l * dnz_dtheta;
        double dFu_dtheta = dDu_dtheta * nz2 - 2 * Du * nz * dnz_dtheta;
        double dFl_dtheta = dDl_dtheta * nz2 - 2 * Dl * nz * dnz_dtheta;
        jacobians[0][0] = (dFu_dtheta - dFl_dtheta) / (nz4 + 1e-16) * weight;

        // jacobians[1][0] = \frac{\partial E}{\partial \phi}
        double dDu_dphi = nx_u * dnz_dphi - nz_u * dnx_dphi;
        double dDl_dphi = nz_l * dny_dphi - ny_l * dnz_dphi;
        double dFu_dphi = dDu_dphi * nz2 - 2 * Du * nz * dnz_dphi;
        double dFl_dphi = dDl_dphi * nz2 - 2 * Dl * nz * dnz_dphi;
        jacobians[1][0] = (dFu_dphi - dFl_dphi) / (nz4 + 1e-16) * weight;
      }

      {
        double dnx_dtheta = -sinTheta_l;
        double dny_dtheta = cosTheta_l * cosPhi_l;
        double dnz_dtheta = cosTheta_l * sinPhi_l;

        double dnx_dphi = 0;
        double dny_dphi = -sinTheta_l * sinPhi_l;
        double dnz_dphi = sinTheta_l * cosPhi_l;

        // jacobians[2][0] = \frac{\partial E}{\partial \theta_l}
        double dDu_dtheta = 0;
        double dDl_dtheta = ny * dnz_dtheta - nz * dny_dtheta;
        double dFu_dtheta = dDu_dtheta * nz2;
        double dFl_dtheta = dDl_dtheta * nz2;
        jacobians[2][0] = (dFu_dtheta - dFl_dtheta) / (nz4 + 1e-16) * weight;

        // jacobians[3][0] = \frac{\partial E}{\partial \phi_l}
        double dDu_dphi = 0;
        double dDl_dphi = ny * dnz_dphi - nz * dny_dphi;
        double dFu_dphi = dDu_dphi * nz2;
        double dFl_dphi = dDl_dphi * nz2;
        jacobians[3][0] = (dFu_dphi - dFl_dphi) / (nz4 + 1e-16) * weight;
      }

      {
        double dnx_dtheta = -sinTheta_u;
        double dny_dtheta = cosTheta_u * cosPhi_u;
        double dnz_dtheta = cosTheta_u * sinPhi_u;

        double dnx_dphi = 0;
        double dny_dphi = - sinTheta_u * sinPhi_u;
        double dnz_dphi = sinTheta_u * cosPhi_u;

        // jacobians[4][0] = \frac{\partial E}{\partial \theta_u}
        double dDu_dtheta = nz * dnx_dtheta - nx * dnz_dtheta;
        double dDl_dtheta = 0;
        double dFu_dtheta = dDu_dtheta * nz2;
        double dFl_dtheta = dDl_dtheta * nz2;
        jacobians[4][0] = (dFu_dtheta - dFl_dtheta) / (nz4 + 1e-16) * weight;

        // jacobians[5][0] = \frac{\partial E}{\partial \phi_u}
        double dDu_dphi = nz * dnx_dphi - nx * dnz_dphi;
        double dDl_dphi = 0;
        double dFu_dphi = dDu_dphi * nz2;
        double dFl_dphi = dDl_dphi * nz2;
        jacobians[5][0] = (dFu_dphi - dFl_dphi) / (nz4 + 1e-16) * weight;
      }
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

      double nx = cos(theta);
      double ny = sin(theta) * cos(phi);
      double nz = sin(theta) * sin(phi);

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

      double nx = cos(theta);
      double ny = sin(theta) * cos(phi);
      double nz = sin(theta) * sin(phi);

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

        jacobians[i*2][0] = - w * sin(theta) * weight;
        jacobians[i*2][1] = w * cos(theta) * cos(phi) * weight;
        jacobians[i*2][2] = w * cos(theta) * sin(phi) * weight;

        jacobians[i*2+1][0] = 0;
        jacobians[i*2+1][1] = -w * sin(theta) * sin(phi) * weight;
        jacobians[i*2+1][2] = -w * sin(theta) * cos(phi) * weight;
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
