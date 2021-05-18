/*
* Based on TOWR (https://github.com/ethz-adrl/towr) with modifications
* by Davis Rempe (2021).
*/

/******************************************************************************
Copyright (c) 2018, Alexander W. Winkler. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#ifndef HUMANOID_RIBID_BODY_DYNAMICS_MODEL_H_
#define HUMANOID_RIBID_BODY_DYNAMICS_MODEL_H_

#include <towr/models/dynamic_model.h>

namespace towr {

/**
 * @brief Dynamics model relating forces to base accelerations.
 *
 * This class extends the Single Rigid Body dynamics (SRBD) model, a reduced
 * dimensional model, lying in terms of accuracy between a Linear
 * Inverted Pendulum model and a full Centroidal or Rigid-body-dynamics model.
 *
 * For the derivation and all assumptions of the oriignal SRBD model, see:
 * https://doi.org/10.3929/ethz-b-000272432
 * 
 * This model extends the original to use a time-varying inertia matrix.
 * 
 */
class HumanoidRigidBodyDynamics : public DynamicModel {
public:
  using IMat = std::vector< Eigen::SparseMatrix<double, Eigen::RowMajor> >;
  using Ptr      = std::shared_ptr<HumanoidRigidBodyDynamics>;

    /**
   * @brief Constructs a specific model.
   * @param mass         The mass of the robot.
   * @param ee_count     The number of endeffectors/forces.
   * @param inertia_b    The elements of the 3x3 Inertia matrix around the CoM for each time step.
   *                       Should be num_frames x 6 where
   *                     the elements are in order Ixx, Iyy, Izz, Ixy, Ixz, Iyz
   * @param total_time   Total duration of the dyanmics sequence (seconds).
   */
  HumanoidRigidBodyDynamics (double mass, const Eigen::MatrixXd& inertia_b, int ee_count, double total_time);

  virtual ~HumanoidRigidBodyDynamics () = default;

  BaseAcc GetDynamicViolation() const override;

  Jac GetJacobianWrtBaseLin(const Jac& jac_base_lin_pos,
                            const Jac& jac_acc_base_lin) const override;
  Jac GetJacobianWrtBaseAng(const EulerConverter& base_angular,
                            double t) const override;
  Jac GetJacobianWrtForce(const Jac& jac_force, EE) const override;

  Jac GetJacobianWrtEEPos(const Jac& jac_ee_pos, EE) const override;

  void SetGravityDir(Eigen::Vector3d grav);

  void SetCurrentTime(double t);

private:
  /** Inertia of entire robot around the CoM expressed in a frame anchored
   *  in the base.
   */
  Eigen::SparseMatrix<double, Eigen::RowMajor> I_b;
  IMat I_all;

  /**
    * Direction of gravity
   */
   Eigen::Vector3d g_vec;

   double total_time_;
   double current_time_;
};


} /* namespace towr */

#endif /* HUMANOID_RIBID_BODY_DYNAMICS_MODEL_H_ */
