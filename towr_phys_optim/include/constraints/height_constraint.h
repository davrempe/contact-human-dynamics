#ifndef HEIGHT_CONSTRAINT_H_
#define HEIGHT_CONSTRAINT_H_

#include <towr/variables/nodes_variables.h>
#include <towr/variables/nodes_variables_phase_based.h>
#include <towr/variables/node_spline.h>
#include <towr/variables/phase_spline.h>

#include <towr/terrain/height_map.h>

#include <towr/constraints/time_discretization_constraint.h>

#include "terrain/ground_plane.h"

namespace towr {

/** @brief Constrains a position variable to be at a certain height above the terrain at multiple timesteps
  *
  * @ingroup Constraints
  */
class HeightConstraint : public TimeDiscretizationConstraint {
public:
  using EE = uint;
  using Vector3d = Eigen::Vector3d;

  //
  // Right now 0 height is enforced.
  //

  /**
   * @brief Constructs a height constraint.
   * @param T   The total duration of the optimization.
   * @param dt  the discretization interval at which to enforce constraints.
   * @param terrain  The terrain height value and slope for each position x,y.
   * @param ee_motion_id The name of the endeffector variable set.
   */
  HeightConstraint (double T, double dt, const GroundPlane::Ptr& terrain, const EE& ee, NodeSpline::Ptr spline);
  virtual ~HeightConstraint() = default;

private:
  GroundPlane::Ptr terrain_;    ///< the height map of the current terrain.

  EE ee_;
  std::shared_ptr<NodesVariables> nodes_;
  NodeSpline::Ptr node_spline_;

  bool is_phase_based_;

  // see TimeDiscretizationConstraint for documentation
  void UpdateConstraintAtInstance (double t, int k, VectorXd& g) const override;
  void UpdateBoundsAtInstance (double t, int k, VecBound&) const override;
  void UpdateJacobianAtInstance(double t, int k, std::string, Jacobian&) const override;

};

} /* namespace towr */

#endif /* HEIGHT_CONSTRAINT_H_ */
