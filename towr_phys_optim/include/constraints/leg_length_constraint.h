#ifndef LEG_LENGTH_CONSTRAINT_H
#define LEG_LENGTH_CONSTRAINT_H


#include <vector>

#include <towr/variables/spline.h>
#include <towr/variables/spline_holder.h>
#include <towr/variables/euler_converter.h>

#include <towr/models/kinematic_model.h>

#include <towr/constraints/time_discretization_constraint.h>

#include "models/humanoid.h"

namespace towr {

/** @brief Constrains an endeffector to lie within a sphere around the hip. This avoids
  * unreachable configurations when performing post-processing IK
  *
  * @ingroup Constraints
  */
class LegLengthConstraint : public TimeDiscretizationConstraint {
public:
  using EE = uint;
  using Vector3d = Eigen::Vector3d;

  /**
   * @brief Constructs a constraint instance.
   * @param robot_model   The kinematic restrictions of the robot.
   * @param T   The total duration of the optimization.
   * @param dt  the discretization intervall at which to enforce constraints.
   * @param ee            The endeffector for which to constrain the range.
   * @param spline_holder Pointer to the current variables.
   */
  LegLengthConstraint(const HumanoidKinematicModel::Ptr& robot_model,
                          double T, double dt,
                          const EE& ee,
                          const SplineHolder& spline_holder);
  virtual ~LegLengthConstraint() = default;

private:
  NodeSpline::Ptr base_linear_;     ///< the linear position of the base.
  EulerConverter base_angular_;     ///< the orientation of the base.
  NodeSpline::Ptr ee_motion_;       ///< the linear position of the endeffectors.

  double max_leg_length_;
  bool is_toe_; // whether this is a toe joint or heel joint
  std::vector<Eigen::Vector3d> ee_hip_offsets_;
  EE ee_;

  // see TimeDiscretizationConstraint for documentation
  void UpdateConstraintAtInstance (double t, int k, VectorXd& g) const override;
  void UpdateBoundsAtInstance (double t, int k, VecBound&) const override;
  void UpdateJacobianAtInstance(double t, int k, std::string, Jacobian&) const override;

};

} /* namespace towr */

#endif /* LEG_LENGTH_CONSTRAINT_H */
