
#include "constraints/height_constraint.h"

#include <towr/variables/variable_names.h>

#include <iostream>

namespace towr {

HeightConstraint::HeightConstraint (double T, double dt,
                                    const GroundPlane::Ptr& terrain,
                                    const EE& ee, NodeSpline::Ptr spline)
    :TimeDiscretizationConstraint(T, dt, "height-terrain-ee-" + std::to_string(ee))
{
  terrain_ = terrain;

  ee_ = ee;
  node_spline_ = spline;

  SetRows(GetNumberOfNodes());
}

void
HeightConstraint::UpdateConstraintAtInstance (double t, int k, VectorXd& g) const
{
    Vector3d p = node_spline_->GetPoint(t).p();
    g(k) = terrain_->GetNormal().dot(p - terrain_->GetPoint());
}

void
HeightConstraint::UpdateBoundsAtInstance (double t, int k, VecBound& bounds) const
{
    bounds.at(k) = ifopt::BoundGreaterZero;
}

void
HeightConstraint::UpdateJacobianAtInstance (double t, int k,
                                            std::string var_set,
                                            Jacobian& jac) const
{
    if (var_set == id::EESchedule(ee_)) {
        Vector3d norm = terrain_->GetNormal();
        Jacobian dur_jac = node_spline_->GetJacobianOfPosWrtDurations(t);
        Eigen::VectorXd res = norm.transpose() * dur_jac;
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }

    if (var_set == id::EEMotionNodes(ee_)) {
        Vector3d norm = terrain_->GetNormal();
        Jacobian pos_jac = node_spline_->GetJacobianWrtNodes(t,kPos);
        Eigen::VectorXd res = norm.transpose() * pos_jac;
        for (int i = 0; i < res.size(); i++) {
            jac.coeffRef(k, i) = res(i);
        }
    }
}

} /* namespace xpp */

