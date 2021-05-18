#ifndef DURATION_COST_H
#define DURATION_COST_H

#include <string>

#include <ifopt/cost_term.h>

#include<towr/variables/phase_durations.h>

namespace towr {

/**
 * @brief  Assigns a cost to durations for an end-effector deviating from some given initialization.
 *
 * @ingroup Costs
 */
class DurationCost : public ifopt::CostTerm {
public:
  /**
   * @brief Constructs a cost term for the optimization problem.
   * @param data_init   An vector holding the duration initialization.
   * @param weight      The weight to give this cost term.
   */
  DurationCost (int ee_id, const std::vector<double> data_init, double weight);
  virtual ~DurationCost () = default;

  void InitVariableDependedQuantities(const VariablesPtr& x) override;

  double GetCost () const override;

private:
  int ee_id_;
  PhaseDurations::Ptr phase_durations_;
  std::vector<double> data_init_;
  double weight_;
  
  void FillJacobianBlock(std::string var_set, Jacobian&) const override;
};

} /* namespace towr */

#endif /* DURATION_COST_H */
