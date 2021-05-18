#include "constraints/leg_length_constraint.h"

#include <iostream>

#include <towr/variables/variable_names.h>

namespace towr {

LegLengthConstraint::LegLengthConstraint (const HumanoidKinematicModel::Ptr& model,
                                                  double T, double dt,
                                                  const EE& ee,
                                                  const SplineHolder& spline_holder)
    :TimeDiscretizationConstraint(T, dt, "leg-length-" + std::to_string(ee))
{
  std::cout << "leg length: " << ee << std::endl;
  base_linear_  = spline_holder.base_linear_;
  base_angular_ = EulerConverter(spline_holder.base_angular_);
  ee_motion_    = spline_holder.ee_motion_.at(ee);
  std::cout << "leg length past: " << ee << std::endl;

  if (ee == 0 || ee == 1) {
      //it's a toe
      max_leg_length_ = model->GetMaximumToeLength();
  } else {
      // it's a heel
      max_leg_length_ = model->GetMaximumHeelLength();
  }

  std::cout << max_leg_length_ << std::endl;
  ee_hip_offsets_ = model->GetHipOffsets(ee);
  ee_ = ee;

  // one constraint for each time step not an element-wise constraint
  SetRows(GetNumberOfNodes());
}

void
LegLengthConstraint::UpdateConstraintAtInstance (double t, int k, VectorXd& g) const
{
    double total_time = dts_.at(dts_.size() - 1);
    int cur_idx = (int)((t / total_time) * ee_hip_offsets_.size());
    if (cur_idx == ee_hip_offsets_.size()) cur_idx -= 1; // corner case for last time

    Vector3d cur_hip_offset = ee_hip_offsets_.at(cur_idx);

    Vector3d base_W  = base_linear_->GetPoint(t).p();
    Vector3d pos_ee_W = ee_motion_->GetPoint(t).p();
    EulerConverter::MatrixSXd w_R_b = base_angular_.GetRotationMatrixBaseToWorld(t);
    Vector3d pos_hip_W = w_R_b*cur_hip_offset + base_W;

    Vector3d vector_hip_to_ee_W = pos_ee_W - pos_hip_W;
    g(k) = 0.5 * vector_hip_to_ee_W.squaredNorm();
}

void
LegLengthConstraint::UpdateBoundsAtInstance (double t, int k, VecBound& bounds) const
{
    // bounds are always the same
    bounds.at(k) = ifopt::Bounds(0.0, 0.5 * max_leg_length_ * max_leg_length_);
}

void
LegLengthConstraint::UpdateJacobianAtInstance (double t, int k,
                                                   std::string var_set,
                                                   Jacobian& jac) const
{
    double total_time = dts_.at(dts_.size() - 1);
    int cur_idx = (int)((t / total_time) * ee_hip_offsets_.size());
    if (cur_idx == ee_hip_offsets_.size()) cur_idx -= 1; // corner case for last time

    Vector3d cur_hip_offset = ee_hip_offsets_.at(cur_idx);

    Vector3d base_W  = base_linear_->GetPoint(t).p();
    Vector3d pos_ee_W = ee_motion_->GetPoint(t).p();
    EulerConverter::MatrixSXd w_R_b = base_angular_.GetRotationMatrixBaseToWorld(t);
    Vector3d pos_hip_W = w_R_b*cur_hip_offset + base_W;

    Vector3d vector_hip_to_ee_W = pos_ee_W - pos_hip_W;

    if (var_set == id::base_lin_nodes) {
        Jacobian jac_t = base_linear_->GetJacobianWrtNodes(t, kPos);
        VectorXd res = (-1*jac_t.transpose() * vector_hip_to_ee_W).transpose();
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }

    if (var_set == id::base_ang_nodes) {
        Jacobian jac_t = base_angular_.DerivOfRotVecMult(t, cur_hip_offset, false);
        VectorXd res = (-1*jac_t.transpose() * vector_hip_to_ee_W).transpose();
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }

    if (var_set == id::EEMotionNodes(ee_)) {
        Jacobian jac_t = ee_motion_->GetJacobianWrtNodes(t,kPos);
        VectorXd res = (jac_t.transpose() * vector_hip_to_ee_W).transpose();
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }

    if (var_set == id::EESchedule(ee_)) {
        Jacobian jac_t = ee_motion_->GetJacobianOfPosWrtDurations(t);
        VectorXd res = (jac_t.transpose() * vector_hip_to_ee_W).transpose();
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }
}

} /* namespace xpp */

