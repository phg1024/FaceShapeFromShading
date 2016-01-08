#ifndef FACESHAPEFROMSHADING_COST_FUNCTIONS_H
#define FACESHAPEFROMSHADING_COST_FUNCTIONS_H

#include <common.h>

#include "ceres/ceres.h"

struct NormalMapDataTerm {
  NormalMapDataTerm(int cons_idx,
                    double Ir, double Ig, double Ib,
                    double ar, double ag, double ab,
                    const VectorXd& lighting_coeffs)
    : cons_idx(cons_idx), Ir(Ir), Ig(Ig), Ib(Ib), ar(ar), ag(ag), ab(ab),
      lighting_coeffs(lighting_coeffs) {}

  ~NormalMapDataTerm() {}

  bool operator()(double const * const * parameters, double *residuals) const {
    const int num_dof = 9;

    double theta = parameters[0][0], phi = parameters[1][0];

    // ny = cos(phi)
    // nx = sin(theta) * sin(phi)
    // nz = cos(theta) * sin(phi)

    double nz = cos(theta) * sin(phi);
    double nx = sin(theta) * sin(phi);
    double ny = cos(phi);

    VectorXd Y(num_dof);
    Y(0) = 1;
    Y(1) = nx; Y(2) = ny; Y(3) = nz;
    Y(4) = nx * ny; Y(5) = nx * nz; Y(6) = ny * nz;
    Y(7) = nx * nx - ny * ny;
    Y(8) = 3.0 * nz * nz - 1.0;

    double LdotY = lighting_coeffs.transpose() * Y;

    residuals[0] = (Ir - ar * LdotY);
    residuals[1] = (Ig - ag * LdotY);
    residuals[2] = (Ib - ab * LdotY);

    return true;
  }

  int cons_idx;
  double Ir, Ig, Ib, ar, ag, ab;
  VectorXd lighting_coeffs;
};

struct NormalMapDataTerm_analytic : public ceres::CostFunction {
  NormalMapDataTerm_analytic(double Ir, double Ig, double Ib,
                             double ar, double ag, double ab,
                             const VectorXd& lighting_coeffs)
    : Ir(Ir), Ig(Ig), Ib(Ib), ar(ar), ag(ag), ab(ab), lighting_coeffs(lighting_coeffs) {

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
                        double **jacobians) const {
    const int num_dof = 9;

    double theta = parameters[0][0], phi = parameters[1][0];
    double sinTheta = sin(theta), cosTheta = cos(theta);
    double sinPhi = sin(phi), cosPhi = cos(phi);

    double nz = cosTheta * sinPhi;
    double nx = sinTheta * sinPhi;
    double ny = cosPhi;

    VectorXd Y(num_dof);
    Y(0) = 1;
    Y(1) = nx; Y(2) = ny; Y(3) = nz;
    Y(4) = nx * ny; Y(5) = nx * nz; Y(6) = ny * nz;
    Y(7) = nx * nx - ny * ny;
    Y(8) = 3.0 * nz * nz - 1.0;

    double LdotY = lighting_coeffs.transpose() * Y;

    residuals[0] = Ir - ar * LdotY;
    residuals[1] = Ig - ag * LdotY;
    residuals[2] = Ib - ab * LdotY;

    if (jacobians != NULL) {
      assert(jacobians[0] != NULL);
      assert(jacobians[1] != NULL);

#if 1
      VectorXd dYdtheta(num_dof);

      double sin2Theta = sin(2 * theta);
      double cos2Theta = cos(2 * theta);

      dYdtheta(0) = 0;
      dYdtheta(1) = cosTheta * sinPhi;
      dYdtheta(2) = 0;
      dYdtheta(3) = -sinTheta * sinPhi;
      dYdtheta(4) = cosTheta * sinPhi * cosPhi;
      dYdtheta(5) = cos2Theta * sinPhi * sinPhi;
      dYdtheta(6) = -sinTheta * sinPhi * cosPhi;
      dYdtheta(7) = sin2Theta * sinPhi * sinPhi;
      dYdtheta(8) = -3 * sin2Theta * sinPhi * sinPhi;

      VectorXd dYdphi(num_dof);

      double sin2Phi = sin(2 * phi);
      double cos2Phi = cos(2 * phi);

      dYdphi(0) = 0;
      dYdphi(1) = sinTheta * cosPhi;
      dYdphi(2) = -sinPhi;
      dYdphi(3) = cosTheta * cosPhi;
      dYdphi(4) = sinTheta * cos2Phi;
      dYdphi(5) = sinTheta * cosTheta * sin2Phi;
      dYdphi(6) = cosTheta * cos2Phi;
      dYdphi(7) = (sinTheta * sinTheta + 1) * sin2Phi;
      dYdphi(8) = 3 * cosTheta * cosTheta * sin2Phi;
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
};

struct NormalMapIntegrabilityTerm {
  NormalMapIntegrabilityTerm(double weight) : weight(weight) {}

  double safe_division(double numer, double denom, double eps) const {
    if(fabs(denom) < eps) {
      denom = (denom<0)?-eps:eps;
    }
    return numer / denom;
  }

  bool operator()(double const * const * parameters, double *residuals) const {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_l = parameters[2][0], phi_l = parameters[3][0];
    double theta_u = parameters[4][0], phi_u = parameters[5][0];

    double nz = cos(theta) * sin(phi);
    double nx = sin(theta) * sin(phi);
    double ny = cos(phi);

    double nz_l = cos(theta_l) * sin(phi_l);
    double nx_l = sin(theta_l) * sin(phi_l);
    double ny_l = cos(phi_l);

    double nz_u = cos(theta_u) * sin(phi_u);
    double nx_u = sin(theta_u) * sin(phi_u);
    double ny_u= cos(phi_u);

    const double epsilon = 1e-3;
    double nxnz = safe_division(nx, nz, epsilon);
    double nynz = safe_division(ny, nz, epsilon);

    double nynz_l = safe_division(ny_l, nz_l, epsilon);
    double nxnz_u = safe_division(nx_u, nz_u, epsilon);

    residuals[0] = ((nxnz_u - nxnz) - (nynz - nynz_l)) * 0.5 * weight;

    return true;
  }

  double weight;
};

struct NormalMapIntegrabilityTerm_analytic : public ceres::CostFunction {
  NormalMapIntegrabilityTerm_analytic(double weight) : weight(weight) {
    mutable_parameter_block_sizes()->clear();
    for(int param_i=0;param_i<6;++param_i)
      mutable_parameter_block_sizes()->push_back(1);
    set_num_residuals(1);
  }

  double safe_division(double numer, double denom, double eps) const {
    if(fabs(denom) < eps) {
      denom = (denom<0)?-eps:eps;
    }
    return numer / denom;
  }

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_l = parameters[2][0], phi_l = parameters[3][0];
    double theta_u = parameters[4][0], phi_u = parameters[5][0];

    double nz = cos(theta) * sin(phi);
    double nx = sin(theta) * sin(phi);
    double ny = cos(phi);

    double nz_l = cos(theta_l) * sin(phi_l);
    double nx_l = sin(theta_l) * sin(phi_l);
    double ny_l = cos(phi_l);

    double nz_u = cos(theta_u) * sin(phi_u);
    double nx_u = sin(theta_u) * sin(phi_u);
    double ny_u = cos(phi_u);

    const double epsilon = 1e-5;
    double nxnz = safe_division(nx, nz, epsilon);
    double nynz = safe_division(ny, nz, epsilon);

    double nynz_l = safe_division(ny_l, nz_l, epsilon);
    double nxnz_u = safe_division(nx_u, nz_u, epsilon);

    residuals[0] = ((nxnz_u - nxnz) - (nynz - nynz_l)) * 0.5 * weight;

    if (jacobians != NULL) {
      for(int param_i=0;param_i<10;++param_i) assert(jacobians[0] != NULL);

      // jacobians[0][0] = \frac{\partial E}{\partial \theta}
      jacobians[0][0] = -safe_division(1 + sin(theta) * cos(phi), cos(theta) * cos(theta) * sin(phi), epsilon) * weight * 0.5;
      // jacobians[1][0] = \frac{\partial E}{\partial \phi}
      jacobians[1][0] = safe_division(1, cos(theta) * sin(phi) * sin(phi), epsilon) * weight * 0.5;

      // jacobians[2][0] = \frac{\partial E}{\partial \theta_l}
      jacobians[2][0] = safe_division(sin(theta_l) * cos(phi_l), cos(theta_l) * cos(theta_l) * sin(phi_l), epsilon) * weight * 0.5;
      // jacobians[3][0] = \frac{\partial E}{\partial \phi_l}
      jacobians[3][0] = safe_division(1, cos(theta_l) * sin(phi_l) * sin(phi_l), epsilon) * weight * 0.5;

      // jacobians[4][0] = \frac{\partial E}{\partial \theta_u}
      jacobians[4][0] = safe_division(1, cos(theta_u) * cos(theta_u), epsilon) * weight * 0.5;
      // jacobians[5][0] = \frac{\partial E}{\partial \phi_u}
      jacobians[5][0] = 0;
    }
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

      double nx = sin(theta) * sin(phi);
      double ny = cos(phi);
      double nz = cos(theta) * sin(phi);

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

      double nx = sin(theta) * sin(phi);
      double ny = cos(phi);
      double nz = cos(theta) * sin(phi);

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

        jacobians[i*2][0] = w * cos(theta) * sin(phi) * weight;
        jacobians[i*2][1] = 0;
        jacobians[i*2][2] = - w * sin(theta) * sin(phi) * weight;

        jacobians[i*2+1][0] = w * sin(theta) * cos(phi) * weight;
        jacobians[i*2+1][1] = -w * sin(phi) * weight;
        jacobians[i*2+1][2] = w * cos(theta) * cos(phi) * weight;
      }
    }

    return true;
  }

  vector<pair<int, double>> info;
  Vector3d normal_ref_LoG;
  double weight;
};

#endif  // FACESHAPEFROMSHADING_COST_FUNCTIONS_H
