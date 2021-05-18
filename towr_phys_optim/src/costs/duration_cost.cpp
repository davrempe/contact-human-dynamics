#include "costs/duration_cost.h"

#include <towr/variables/cartesian_dimensions.h>
#include <towr/variables/variable_names.h>

#include <cmath>
#include <iostream>

namespace towr {

DurationCost::DurationCost(int ee_id, const std::vector<double> data_init, double weight)
    : CostTerm(id::EESchedule(ee_id) + "-duration")
{
  ee_id_ = ee_id;
  data_init_   = data_init;
  weight_ = weight;
}

void
DurationCost::InitVariableDependedQuantities (const VariablesPtr& x)
{
  phase_durations_ = x->GetComponent<PhaseDurations>(id::EESchedule(ee_id_));
}

double
DurationCost::GetCost () const
{
    // only apply cost to first N-1 b/c last one is total time - sum(N-1)
    PhaseDurations::VecDurations cur_dur = phase_durations_->GetPhaseDurations();
    double cost = 0.0;
    for (int i = 0; i < cur_dur.size() - 1; i++) {
        cost += (data_init_.at(i) - cur_dur.at(i)) * (data_init_.at(i) - cur_dur.at(i));
    }

    // std::cout << "cost (" << id::EESchedule(ee_id_) << "): " << cost << std::endl;

    return (0.5 * weight_ * cost);
}

void
DurationCost::FillJacobianBlock (std::string var_set, Jacobian& jac) const
{
    if (var_set == id::EESchedule(ee_id_)) {
        PhaseDurations::VecDurations cur_dur = phase_durations_->GetPhaseDurations();
        for (int i = 0; i < cur_dur.size() - 1; i++) { 
            double diff = data_init_.at(i) - cur_dur.at(i);
            jac.coeffRef(0, i) += weight_ * (-diff);
        }
    }
}

} /* namespace towr */

