#ifndef VEL_SMOOTH_COST_H
#define VEL_SMOOTH_COST_H

#include <string>

#include <ifopt/cost_term.h>
#include <towr/variables/node_spline.h>
#include <towr/variables/phase_spline.h>

#include <towr/variables/nodes_variables.h>
#include <towr/variables/state.h>

namespace towr {

/**
 * @brief  Assigns a cost to a spline derivative (i.e. velocity or acceleration).
 *
 * @ingroup Costs
 */
class VelSmoothCost : public ifopt::CostTerm {
public:
  /**
   * @brief Constructs a cost term for the optimization problem.
   * @param nodes_id    The name of the node variables.
   * @param dt          The step size to apply the cost at
   * @param spline      The spline to penalize.
   * @param deriv       The derivative to use to calculate the value to penalize (i.e. kPos in order to penalize velocity, kVel in order to penalize acceleration)
   * @param weight      The weight to give this cost term.
   */
  VelSmoothCost (const std::string& nodes_id, double dt, NodeSpline::Ptr spline, Dx deriv, double weight, bool is_ee=false);
  virtual ~VelSmoothCost () = default;

  void InitVariableDependedQuantities(const VariablesPtr& x) override;

  double GetCost () const override;

private:
  std::string nodes_id_;
  std::shared_ptr<NodesVariables> nodes_;
  NodeSpline::Ptr node_spline_;
  Dx deriv_;

  double dt_;
  double weight_;

  int ee_id_;
  bool is_ee_;

  void FillJacobianBlock(std::string var_set, Jacobian&) const override;
};

} /* namespace towr */

#endif /* VEL_SMOOTH_COST_H */
