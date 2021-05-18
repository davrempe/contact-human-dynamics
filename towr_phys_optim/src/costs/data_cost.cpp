#include "costs/data_cost.h"

#include <towr/variables/cartesian_dimensions.h>
#include <towr/variables/nodes_variables_phase_based.h>
#include <towr/variables/variable_names.h>

#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <time.h>

namespace towr {

DataCost::DataCost (const std::string& nodes_id, double dt, NodeSpline::Ptr spline, bool is_phase_based, 
                    const Eigen::MatrixXd& data_init, double weight, int enforce_every)
    : CostTerm(nodes_id + "-data")
{
  nodes_id_ = nodes_id;
  dt_ = dt;
  node_spline_ = spline;
  data_init_   = data_init;
  weight_ = weight;
  is_phase_based_ = is_phase_based;
  steps_between_ = enforce_every;

  // if it's phase based it's an end effector and has a number id that we need later for schedule jac
  ee_id_ = -1;
  if (is_phase_based_) {
      std::string last_char(1, nodes_id_[nodes_id_.size() - 1]);
      ee_id_ = std::atoi(last_char.c_str());
  }
}

void
DataCost::InitVariableDependedQuantities (const VariablesPtr& x)
{
  nodes_ = x->GetComponent<NodesVariables>(nodes_id_);
}

double
DataCost::GetCost () const
{
    double cost = 0.0;
    double t = 0.0;
    for (int i = 0; i < data_init_.rows(); i += steps_between_) {
        Eigen::Vector3d diff = data_init_.row(i) - node_spline_->GetPoint(t).p().transpose();
        cost += diff.squaredNorm();
        t += steps_between_*dt_;
    }

    // std::cout << "cost (" << nodes_id_ << "): " << cost << std::endl;

    return (0.5 * weight_ * cost);
}

void
DataCost::FillJacobianBlock (std::string var_set, Jacobian& jac) const
{
    if (is_phase_based_ && var_set == id::EESchedule(ee_id_)) {
        // must take care of schedule too
        double t = 0.0;
        for (int i = 0; i < data_init_.rows(); i += steps_between_) {
            Eigen::Vector3d diff = data_init_.row(i) - node_spline_->GetPoint(t).p().transpose();
            Jacobian sched_jac_t = std::static_pointer_cast<PhaseSpline>(node_spline_)->GetJacobianOfPosWrtDurations(t);

            VectorXd res = -1.0*(sched_jac_t.transpose() * diff).transpose();
            for (int j = 0; j < res.size(); j++) {
                jac.coeffRef(0, j) += res(j) * weight_;
            }

            t += dt_*steps_between_;
        }

        return;
    }

    // otherwise it should match the nodes we care about
    if (var_set == nodes_id_) {
        double t = 0.0;
        for (int i = 0; i < data_init_.rows(); i += steps_between_) {
            Eigen::Vector3d diff = data_init_.row(i) - node_spline_->GetPoint(t).p().transpose();
            Jacobian jac_t = node_spline_->GetJacobianWrtNodes(t, kPos);

            VectorXd res = -1.0*(jac_t.transpose() * diff).transpose();
            for (int j = 0; j < res.size(); j++) {
                jac.coeffRef(0, j) += res(j) * weight_;
            }

            t += dt_*steps_between_;
        }

        return;
    }

    return;
}

} /* namespace towr */

