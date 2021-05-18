#include "costs/vel_smooth_cost.h"

#include <towr/variables/cartesian_dimensions.h>
#include <towr/variables/nodes_variables_phase_based.h>
#include <towr/variables/variable_names.h>

#include <cmath>
#include <iostream>
#include <stdlib.h>

namespace towr {

VelSmoothCost::VelSmoothCost (const std::string& nodes_id, double dt, NodeSpline::Ptr spline, Dx deriv, double weight, bool is_ee)
    : CostTerm(nodes_id + "-deriv" + std::to_string(deriv+1) + "-smooth")
{
  nodes_id_ = nodes_id;
  dt_ = dt;
  node_spline_ = spline;
  weight_ = weight;
  deriv_ = deriv;

  // if it's phase based it's an end effector and has a number id that we need later for schedule jacobian
  is_ee_ = is_ee;
  ee_id_ = -1;
  if (is_ee_) {
      std::string last_char(1, nodes_id_[nodes_id_.size() - 1]);
      ee_id_ = std::atoi(last_char.c_str());
  }
}

void
VelSmoothCost::InitVariableDependedQuantities (const VariablesPtr& x)
{
  nodes_ = x->GetComponent<NodesVariables>(nodes_id_);
}

double
VelSmoothCost::GetCost () const
{
    double cost = 0.0;
    for (double t = 0.0; t < (node_spline_->GetTotalTime() - dt_); t += dt_) {
        Eigen::Vector3d diff = node_spline_->GetPoint(t+dt_).at(deriv_) - node_spline_->GetPoint(t).at(deriv_);
        double diff_norm = diff.squaredNorm();
        cost += diff_norm;
    }

    // std::cout << nodes_id_ + "-deriv" + std::to_string(deriv_+1) + "-smooth: " << cost << std::endl;

    return (0.5 * weight_ * cost);
}

void
VelSmoothCost::FillJacobianBlock (std::string var_set, Jacobian& jac) const
{

    if (is_ee_ && var_set == id::EESchedule(ee_id_) && deriv_ == kPos) {
        // must take care of schedule too
        for (double t = 0.0; t < (node_spline_->GetTotalTime() - dt_); t += dt_) {
            Eigen::Vector3d diff = node_spline_->GetPoint(t+dt_).at(deriv_) - node_spline_->GetPoint(t).at(deriv_);

            Jacobian sched_jac_tp1 = std::static_pointer_cast<PhaseSpline>(node_spline_)->GetJacobianOfPosWrtDurations(t+dt_);
            Jacobian sched_jac_t = std::static_pointer_cast<PhaseSpline>(node_spline_)->GetJacobianOfPosWrtDurations(t);
            Jacobian joint_jac = sched_jac_tp1 - sched_jac_t;

            VectorXd res = (joint_jac.transpose() * diff).transpose();
            for (int j = 0; j < res.size(); j++) {
                jac.coeffRef(0, j) += res(j) * weight_;
            }
        }

        return;
    } else if (is_ee_ && var_set == id::EESchedule(ee_id_) && deriv_ == kVel) {
        // 
        // WARNING: this returns 0 jacobian for duration variable if using acceleration smoothing. Should implement and use.
        //
        // TODO: implement
        std::cout << "Using acceleration smoothing with duration optimization is currently not supported!\n";
        throw std::runtime_error("Using acceleration smoothing with duration optimization is currently not supported!");
    }

    // otherwise it should match the nodes we care about
    if (var_set != nodes_id_) {
        return;
    }

    for (double t = 0.0; t < (node_spline_->GetTotalTime() - dt_); t += dt_) {
        Eigen::Vector3d diff = node_spline_->GetPoint(t+dt_).at(deriv_) - node_spline_->GetPoint(t).at(deriv_);

        Jacobian pos_jac_tp1 = std::static_pointer_cast<PhaseSpline>(node_spline_)->GetJacobianWrtNodes(t+dt_, deriv_);
        Jacobian pos_jac_t = std::static_pointer_cast<PhaseSpline>(node_spline_)->GetJacobianWrtNodes(t, deriv_);
        Jacobian joint_jac = pos_jac_tp1 - pos_jac_t;

        VectorXd res = (joint_jac.transpose() * diff).transpose();
        for (int j = 0; j < res.size(); j++) {
            jac.coeffRef(0, j) += res(j) * weight_;
        }
    }


}

} /* namespace towr */

