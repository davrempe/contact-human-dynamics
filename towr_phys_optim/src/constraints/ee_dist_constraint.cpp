#include "constraints/ee_dist_constraint.h"

#include <iostream>

#include <towr/variables/variable_names.h>

namespace towr {

EEDistConstraint::EEDistConstraint (double T, double dt,
                                    const EE& ee1, const EE& ee2,
                                    const SplineHolder& spline_holder,
                                    double dist)
    :TimeDiscretizationConstraint(T, dt, "ee-dist-" + std::to_string(ee1) + "-" + std::to_string(ee2))
{
  ee1_ = ee1;
  ee2_ = ee2;

  ee1_motion_ = std::static_pointer_cast<PhaseSpline>(spline_holder.ee_motion_.at(ee1_));
  ee2_motion_ = std::static_pointer_cast<PhaseSpline>(spline_holder.ee_motion_.at(ee2_));

  dist_apart_ = dist;

  // one constraint for each time step not an element-wise constraint
  SetRows(GetNumberOfNodes());
}

void
EEDistConstraint::UpdateConstraintAtInstance (double t, int k, VectorXd& g) const
{
    Vector3d pos_ee1 = ee1_motion_->GetPoint(t).p();
    Vector3d pos_ee2 = ee2_motion_->GetPoint(t).p();
    g(k) = 0.5 * (pos_ee1 - pos_ee2).squaredNorm();
}   

void
EEDistConstraint::UpdateBoundsAtInstance (double t, int k, VecBound& bounds) const
{
    // bounds are always the same
    bounds.at(k) = ifopt::Bounds(0.5*dist_apart_*dist_apart_, 0.5*dist_apart_*dist_apart_);
}

void
EEDistConstraint::UpdateJacobianAtInstance (double t, int k,
                                                   std::string var_set,
                                                   Jacobian& jac) const
{
    if (var_set == id::EEMotionNodes(ee1_)) {
        Vector3d pos_ee1 = ee1_motion_->GetPoint(t).p();
        Vector3d pos_ee2 = ee2_motion_->GetPoint(t).p();
        Vector3d diff = pos_ee1 - pos_ee2;

        Jacobian jac_t = ee1_motion_->GetJacobianWrtNodes(t,kPos);
        VectorXd res = (jac_t.transpose() * diff).transpose();
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }

    if (var_set == id::EEMotionNodes(ee2_)) {
        Vector3d pos_ee1 = ee1_motion_->GetPoint(t).p();
        Vector3d pos_ee2 = ee2_motion_->GetPoint(t).p();
        Vector3d diff = pos_ee1 - pos_ee2;

        Jacobian jac_t = ee2_motion_->GetJacobianWrtNodes(t,kPos);
        VectorXd res = -1.0 * (jac_t.transpose() * diff).transpose();
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }

    if (var_set == id::EESchedule(ee1_)) {
        Vector3d pos_ee1 = ee1_motion_->GetPoint(t).p();
        Vector3d pos_ee2 = ee2_motion_->GetPoint(t).p();
        Vector3d diff = pos_ee1 - pos_ee2;

        Jacobian jac_t = ee1_motion_->GetJacobianOfPosWrtDurations(t);
        VectorXd res = (jac_t.transpose() * diff).transpose();
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }

    if (var_set == id::EESchedule(ee2_)) {
        Vector3d pos_ee1 = ee1_motion_->GetPoint(t).p();
        Vector3d pos_ee2 = ee2_motion_->GetPoint(t).p();
        Vector3d diff = pos_ee1 - pos_ee2;

        Jacobian jac_t = ee2_motion_->GetJacobianOfPosWrtDurations(t);
        VectorXd res = -1.0 * (jac_t.transpose() * diff).transpose();
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }
}

} /* namespace xpp */

