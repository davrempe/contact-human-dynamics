#ifndef DATA_COST_H
#define DATA_COST_H

#include <string>

#include <ifopt/cost_term.h>
#include <towr/variables/node_spline.h>
#include <towr/variables/phase_spline.h>

#include <towr/variables/nodes_variables.h>

namespace towr {

/**
 * @brief  Assigns a cost to a spline (the position derivative) deviating from some given initialization.
 *
 * @ingroup Costs
 */
class DataCost : public ifopt::CostTerm {
public:
  /**
   * @brief Constructs a cost term for the optimization problem.
   * @param nodes_id    The name of the node variables.
   * @param dt          The step size of the given initialization data.
   * @param spline      The spline to penalize.
   * @param data_init   An Nx3 matrix containing the data initialization.
   * @param weight      The weight to give this cost term.
   */
  DataCost (const std::string& nodes_id, double dt, NodeSpline::Ptr spline, bool is_phase_based, const Eigen::MatrixXd& data_init, double weight, int enforce_every=1);
  virtual ~DataCost () = default;

  void InitVariableDependedQuantities(const VariablesPtr& x) override;

  double GetCost () const override;

private:
  std::string nodes_id_;
  std::shared_ptr<NodesVariables> nodes_;
  NodeSpline::Ptr node_spline_;
  Eigen::MatrixXd data_init_;
  double dt_;
  double weight_;
  bool is_phase_based_;
  int steps_between_;
  int ee_id_;

  int print_every_;

  void FillJacobianBlock(std::string var_set, Jacobian&) const override;
};

} /* namespace towr */

#endif /* DATA_COST_H */
