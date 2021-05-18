#ifndef HUMANOID_MODEL_H_
#define HUMANOID_MODEL_H_

#include <vector>

#include <towr/models/kinematic_model.h>
#include <towr/models/endeffector_mappings.h>

#include <towr/variables/variable_names.h>

#include "models/humanoid_rigid_body_dynamics.h"

namespace towr {

/**
 * @brief The Kinematics of a humanoid skeleton.
 */
class HumanoidKinematicModel : public KinematicModel {
public:
  using Ptr      = std::shared_ptr<HumanoidKinematicModel>;

  HumanoidKinematicModel (int num_ee, Eigen::MatrixXd left_hip_offset, Eigen::MatrixXd right_hip_offset, double max_leg_length, double max_heel_length) 
                        : KinematicModel(num_ee)
  {

    for (int i = 0; i < left_hip_offset.rows(); i++) {
      Eigen::Vector3d cur_off;
      cur_off << left_hip_offset(i, 0), left_hip_offset(i, 1), left_hip_offset(i, 2);
      left_hip_offsets_.push_back(cur_off); 
    }
    for (int i = 0; i < right_hip_offset.rows(); i++) {
      Eigen::Vector3d cur_off;
      cur_off << right_hip_offset(i, 0), right_hip_offset(i, 1), right_hip_offset(i, 2);
      right_hip_offsets_.push_back(cur_off);
    }

    // maximum distance from hip the foot is allowed to be
    max_leg_length_ = max_leg_length;
    max_heel_length_ = max_heel_length;
  }

  double GetMaximumHeelLength() { return max_heel_length_; }
  double GetMaximumToeLength() { return max_leg_length_; }

  std::vector<Eigen::Vector3d> GetHipOffsets(int ee_id) {
      if (ee_id == 0 || ee_id == 2) return left_hip_offsets_;
      if (ee_id == 1 || ee_id == 3) return right_hip_offsets_;
  }

protected:
  double max_leg_length_;
  double max_heel_length_;
  std::vector<Eigen::Vector3d> left_hip_offsets_;
  std::vector<Eigen::Vector3d> right_hip_offsets_;
};

/**
 * @brief The Dynamics of a humanoid skeleton.
 */
class HumanoidDynamicModel : public HumanoidRigidBodyDynamics {
public:
  HumanoidDynamicModel(int num_ee, double mass, Eigen::MatrixXd& I_b, Eigen::Vector3d g, double total_time)
  : HumanoidRigidBodyDynamics(mass, I_b, num_ee, total_time) 
    {
        SetGravityDir(g);
    }
};

} /* namespace towr */

#endif /* HUMANOID_MODEL_H_ */
