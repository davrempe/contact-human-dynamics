#ifndef EE_DIST_CONSTRAINT_H
#define EE_DIST_CONSTRAINT_H


#include <vector>

#include <towr/variables/spline.h>
#include <towr/variables/spline_holder.h>
#include <towr/variables/phase_spline.h>

#include <towr/constraints/time_discretization_constraint.h>


namespace towr {

/** @brief Constraints 2 end effectors to stay the same distance apart from each other.
  *
  * @ingroup Constraints
  */
class EEDistConstraint : public TimeDiscretizationConstraint {
public:
  using EE = uint;
  using Vector3d = Eigen::Vector3d;

  /**
   * @brief Constructs a constraint instance.
   * @param T   The total duration of the optimization.
   * @param dt  the discretization intervall at which to enforce constraints.
   * @param ee1, ee2            The endeffectors to constrain
   * @param spline_holder Pointer to the current variables.
   */
  EEDistConstraint(double T, double dt,
                    const EE& ee1, const EE& ee2,
                    const SplineHolder& spline_holder,
                    double dist);
  virtual ~EEDistConstraint() = default;

private:
  PhaseSpline::Ptr ee1_motion_;     ///< the linear position of the endeffectors.
  PhaseSpline::Ptr ee2_motion_; 

  double dist_apart_;
  EE ee1_;
  EE ee2_;

  // see TimeDiscretizationConstraint for documentation
  void UpdateConstraintAtInstance (double t, int k, VectorXd& g) const override;
  void UpdateBoundsAtInstance (double t, int k, VecBound&) const override;
  void UpdateJacobianAtInstance(double t, int k, std::string, Jacobian&) const override;

};

} /* namespace towr */

#endif /* EE_DIST_CONSTRAINT_H */
