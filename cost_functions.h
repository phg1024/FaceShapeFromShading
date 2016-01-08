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

    residuals[0] = Ir - ar * LdotY;
    residuals[1] = Ig - ag * LdotY;
    residuals[2] = Ib - ab * LdotY;

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

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const {
    const int num_dof = 9;

    double theta = parameters[0][0], phi = parameters[1][0];
    double sinTheta = sin(theta), cosTheta = cos(theta);
    double sinPhi = sin(phi), cosPhi = cos(phi);

    double nx = cosTheta * sinPhi;
    double ny = sinTheta * sinPhi;
    double nz = cosPhi;

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
      assert(jacobians[2] != NULL);

      VectorXd dYdtheta(num_dof);

      double sin2Theta = sin(2 * theta);
      double cos2Theta = cos(2 * theta);

      dYdtheta(0) = 0;
      dYdtheta(1) = -sinTheta * sinPhi; dYdtheta(2) = cosTheta * sinPhi; dYdtheta(3) = 0;
      dYdtheta(4) = cos2Theta * sinPhi * sinPhi;
      dYdtheta(5) = -sinTheta * sinPhi * cosPhi;
      dYdtheta(6) = cosTheta * sinPhi * cosPhi;
      dYdtheta(7) = -2 * sin2Theta * sinPhi * sinPhi;
      dYdtheta(8) = 0;

      VectorXd dYdphi(num_dof);

      double sin2Phi = sin(2 * phi);
      double cos2Phi = cos(2 * phi);

      dYdphi(0) = 0;
      dYdphi(1) = cosTheta * cosPhi; dYdphi(2) = sinTheta * cosPhi; dYdphi(3) = -sinPhi;
      dYdphi(4) = sinTheta * cosTheta * sin2Phi;
      dYdphi(5) = cosTheta * cos2Phi;
      dYdphi(6) = sinTheta * cos2Phi;
      dYdphi(7) = (cosTheta * cosTheta - sinTheta * sinTheta) * sin2Phi;
      dYdphi(8) = -3 * sin2Phi;

      double LdotdYdtheta = lighting_coeffs.transpose() * dYdtheta;
      double LdotdYdphi = lighting_coeffs.transpose() * dYdphi;

      // jacobians[i][0] = \frac{\partial E}{\partial \theta}
      jacobians[0][0] = -residuals[0] * ar * LdotdYdtheta;
      jacobians[0][1] = -residuals[1] * ag * LdotdYdtheta;
      jacobians[0][2] = -residuals[2] * ab * LdotdYdtheta;

      // jacobians[i][1] = \frac{\partial E}{\partial \phi}
      jacobians[1][0] = -residuals[0] * ar * LdotdYdphi;
      jacobians[1][1] = -residuals[1] * ag * LdotdYdphi;
      jacobians[1][2] = -residuals[2] * ab * LdotdYdphi;
    }
    return true;
  }

  double Ir, Ig, Ib, ar, ag, ab;
  VectorXd lighting_coeffs;
};

struct NormalMapIntegrabilityTerm {
  NormalMapIntegrabilityTerm(double weight) : weight(weight) {}

  bool operator()(double const * const * parameters, double *residuals) const {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_l = parameters[2][0], phi_l = parameters[3][0];
    double theta_d = parameters[4][0], phi_d = parameters[5][0];

    double nx = cos(theta) * sin(phi);
    double ny = sin(theta) * sin(phi);
    double nz = cos(phi);

    double nx_l = cos(theta_l) * sin(phi_l);
    double ny_l = sin(theta_l) * sin(phi_l);
    double nz_l = cos(phi_l);

    double nx_d = cos(theta_d) * sin(phi_d);
    double ny_d = sin(theta_d) * sin(phi_d);
    double nz_d= cos(phi_d);

    const double epsilon = 1e-6;
    double nxnz = nx / (nz + epsilon);
    double nynz = ny / (nz + epsilon);

    double nynz_l = ny_l / (nz_l + epsilon);

    double nxnz_d = nx_d / (nz_d + epsilon);

    residuals[0] = ((nxnz_d - nxnz) - (nynz - nynz_l)) * weight;

    return true;
  }

  double weight;
};

struct NormalMapIntegrabilityTerm_analytic : public ceres::CostFunction {
  NormalMapIntegrabilityTerm_analytic(double weight) : weight(weight) {
    mutable_parameter_block_sizes()->clear();
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    mutable_parameter_block_sizes()->push_back(1);
    set_num_residuals(1);
  }

  virtual bool Evaluate(double const *const *parameters,
                        double *residuals,
                        double **jacobians) const {
    double theta = parameters[0][0], phi = parameters[1][0];
    double theta_l = parameters[2][0], phi_l = parameters[3][0];
    double theta_d = parameters[4][0], phi_d = parameters[5][0];

    double sinTheta = sin(theta), cosTheta = cos(theta);
    double sinPhi = sin(phi), cosPhi = cos(phi);

    double sinTheta_l = sin(theta_l), cosTheta_l = cos(theta_l);
    double sinPhi_l = sin(phi_l), cosPhi_l = cos(phi_l);

    double sinTheta_d = sin(theta_d), cosTheta_d = cos(theta_d);
    double sinPhi_d = sin(phi_d), cosPhi_d = cos(phi_d);

    double nx = cosTheta * sinPhi;
    double ny = sinTheta * sinPhi;
    double nz = cosPhi;

    double nx_l = cosTheta_l * sinPhi_l;
    double ny_l = sinTheta_l * sinPhi_l;
    double nz_l = cosPhi_l;

    double nx_d = cosTheta_d * sinPhi_d;
    double ny_d = sinTheta_d * sinPhi_d;
    double nz_d= cosPhi_d;

    const double epsilon = 1e-6;
    double nxnz = nx / (nz + epsilon);
    double nynz = ny / (nz + epsilon);

    double nynz_l = ny_l / (nz_l + epsilon);

    double nxnz_d = nx_d / (nz_d + epsilon);

    residuals[0] = ((nxnz_d - nxnz) - (nynz - nynz_l)) * weight;

    if (jacobians != NULL) {
      assert(jacobians[0] != NULL);
      assert(jacobians[1] != NULL);
      assert(jacobians[2] != NULL);
      assert(jacobians[3] != NULL);
      assert(jacobians[4] != NULL);
      assert(jacobians[5] != NULL);

      // jacobians[0][0] = \frac{\partial E}{\partial \theta}
      jacobians[0][0] = 2 * residuals[0] * sinTheta * sinPhi / (cosPhi);
      // jacobians[1][0] = \frac{\partial E}{\partial \phi}
      jacobians[1][0] = -4 * residuals[0] * cosTheta / (cos(2*phi) + 1);

      // jacobians[2][0] = \frac{\partial E}{\partial \theta_l}
      jacobians[2][0] = -residuals[0] * sinTheta_l * sinPhi_l / (cosPhi_l);
      // jacobians[3][0] = \frac{\partial E}{\partial \phi_l}
      jacobians[3][0] = 2 * residuals[0] * cosTheta_l / (cos(2*phi_l) + 1);

      // jacobians[4][0] = \frac{\partial E}{\partial \theta_d}
      jacobians[4][0] = -residuals[0] * sinTheta_d * sinPhi_d / (cosPhi_d);
      // jacobians[5][0] = \frac{\partial E}{\partial \phi_d}
      jacobians[5][0] = 2 * residuals[0] * cosTheta_d / (cos(2*phi_d) + 1);
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
      double nx = cos(theta) * sin(phi);
      double ny = sin(theta) * sin(phi);
      double nz = cos(phi);

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
      double nx = cos(theta) * sin(phi);
      double ny = sin(theta) * sin(phi);
      double nz = cos(phi);

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

        jacobians[i*2][0] = residuals[0] * w * cos(theta) * cos(phi);
        jacobians[i*2][1] = residuals[1] * w * sin(theta) * cos(phi);
        jacobians[i*2][2] = 0;

        jacobians[i*2+1][0] = -residuals[0] * w * sin(theta) * sin(phi);
        jacobians[i*2+1][1] = -residuals[1] * w * cos(theta) * sin(phi);
        jacobians[i*2+1][2] = residuals[2] * w * cos(phi);
      }
    }

    return true;
  }

  vector<pair<int, double>> info;
  Vector3d normal_ref_LoG;
  double weight;
};

#endif  // FACESHAPEFROMSHADING_COST_FUNCTIONS_H
