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

#include <iostream>

#include <towr/variables/cartesian_dimensions.h>

#include "models/humanoid_rigid_body_dynamics.h"

namespace towr {

// some Eigen helper functions
static Eigen::Matrix3d BuildInertiaTensor( double Ixx, double Iyy, double Izz,
                                           double Ixy, double Ixz, double Iyz)
{
  Eigen::Matrix3d I;
  I <<  Ixx, Ixy, Ixz,
       Ixy,  Iyy, Iyz,
       Ixz, Iyz,  Izz;
  return I;
}

// builds a cross product matrix out of "in", so in x v = X(in)*v
HumanoidRigidBodyDynamics::Jac
Cross(const Eigen::Vector3d& in)
{
  HumanoidRigidBodyDynamics::Jac out(3,3);

  out.coeffRef(0,1) = -in(2); out.coeffRef(0,2) =  in(1);
  out.coeffRef(1,0) =  in(2); out.coeffRef(1,2) = -in(0);
  out.coeffRef(2,0) = -in(1); out.coeffRef(2,1) =  in(0);

  return out;
}

HumanoidRigidBodyDynamics::HumanoidRigidBodyDynamics (double mass, const Eigen::MatrixXd& inertia_b,
                                                      int ee_count, double total_time)
    :DynamicModel(mass, ee_count)
{
  for (int i = 0; i < inertia_b.rows(); i++) {
    auto I_cur = BuildInertiaTensor(inertia_b(i, 0), inertia_b(i, 1), inertia_b(i, 2), 
                                      inertia_b(i, 3), inertia_b(i, 4), inertia_b(i, 5)).sparseView();
    I_all.push_back(I_cur);
  }

  g_vec << 0.0, 0.0, -1.0;
  total_time_ = total_time;
}

void HumanoidRigidBodyDynamics::SetCurrentTime(double t) {
  // update the current moment of inertia
  current_time_ = t;
  int cur_idx = (int)((t / total_time_) * I_all.size());
  if (cur_idx == I_all.size()) cur_idx -= 1;
  I_b = I_all.at(cur_idx);
}

HumanoidRigidBodyDynamics::BaseAcc
HumanoidRigidBodyDynamics::GetDynamicViolation () const
{
  // https://en.wikipedia.org/wiki/Newton%E2%80%93Euler_equations

  Vector3d f_sum, tau_sum;
  f_sum.setZero(); tau_sum.setZero();

  for (int ee=0; ee<ee_pos_.size(); ++ee) {
    Vector3d f = ee_force_.at(ee);
    tau_sum += f.cross(com_pos_ - ee_pos_.at(ee));
    f_sum   += f;
  }

  // express inertia matrix in world frame based on current body orientation
  Jac I_w = w_R_b_.sparseView() * I_b * w_R_b_.transpose().sparseView();

  BaseAcc acc;
  acc.segment(AX, k3D) = I_w*omega_dot_
                         + Cross(omega_)*(I_w*omega_)
                         - tau_sum;
  acc.segment(LX, k3D) = m()*com_acc_
                         - f_sum
                         - m()*g()*g_vec;

  return acc;
}

HumanoidRigidBodyDynamics::Jac
HumanoidRigidBodyDynamics::GetJacobianWrtBaseLin (const Jac& jac_pos_base_lin,
                                        const Jac& jac_acc_base_lin) const
{
  // build the com jacobian
  int n = jac_pos_base_lin.cols();

  Jac jac_tau_sum(k3D, n);
  for (const Vector3d& f : ee_force_) {
    Jac jac_tau = Cross(f)*jac_pos_base_lin;
    jac_tau_sum += jac_tau;
  }

  Jac jac(k6D, n);
  jac.middleRows(AX, k3D) = -jac_tau_sum;
  jac.middleRows(LX, k3D) = m()*jac_acc_base_lin;

  return jac;
}

HumanoidRigidBodyDynamics::Jac
HumanoidRigidBodyDynamics::GetJacobianWrtBaseAng (const EulerConverter& base_euler,
                                        double t) const
{
  Jac I_w = w_R_b_.sparseView() * I_b * w_R_b_.transpose().sparseView();

  // Derivative of R*I_b*R^T * wd
  // 1st term of product rule (derivative of R)
  Vector3d v11 = I_b*w_R_b_.transpose()*omega_dot_;
  Jac jac11 = base_euler.DerivOfRotVecMult(t, v11, false);

  // 2nd term of product rule (derivative of R^T)
  Jac jac12 = w_R_b_.sparseView()*I_b*base_euler.DerivOfRotVecMult(t, omega_dot_, true);

  // 3rd term of product rule (derivative of wd)
  Jac jac_ang_acc = base_euler.GetDerivOfAngAccWrtEulerNodes(t);
  Jac jac13 = I_w * jac_ang_acc;
  Jac jac1 = jac11 + jac12 + jac13;


  // Derivative of w x Iw
  // w x d_dn(R*I_b*R^T*w) -(I*w x d_dnw)
  // right derivative same as above, just with velocity instead acceleration
  Vector3d v21 = I_b*w_R_b_.transpose()*omega_;
  Jac jac21 = base_euler.DerivOfRotVecMult(t, v21, false);

  // 2nd term of product rule (derivative of R^T)
  Jac jac22 = w_R_b_.sparseView()*I_b*base_euler.DerivOfRotVecMult(t, omega_, true);

  // 3rd term of product rule (derivative of omega)
  Jac jac_ang_vel = base_euler.GetDerivOfAngVelWrtEulerNodes(t);
  Jac jac23 = I_w * jac_ang_vel;

  Jac jac2 = Cross(omega_)*(jac21+jac22+jac23) - Cross(I_w*omega_)*jac_ang_vel;


  // Combine the two to get sensitivity to I_w*w + w x (I_w*w)
  int n = jac_ang_vel.cols();
  Jac jac(k6D, n);
  jac.middleRows(AX, k3D) = jac1 + jac2;

  return jac;
}

HumanoidRigidBodyDynamics::Jac
HumanoidRigidBodyDynamics::GetJacobianWrtForce (const Jac& jac_force, EE ee) const
{
  Vector3d r = com_pos_ - ee_pos_.at(ee);
  Jac jac_tau = -Cross(r)*jac_force;

  int n = jac_force.cols();
  Jac jac(k6D, n);
  jac.middleRows(AX, k3D) = -jac_tau;
  jac.middleRows(LX, k3D) = -jac_force;

  return jac;
}

HumanoidRigidBodyDynamics::Jac
HumanoidRigidBodyDynamics::GetJacobianWrtEEPos (const Jac& jac_ee_pos, EE ee) const
{
  Vector3d f = ee_force_.at(ee);
  Jac jac_tau = Cross(f)*(-jac_ee_pos);

  Jac jac(k6D, jac_tau.cols());
  jac.middleRows(AX, k3D) = -jac_tau;

  // linear dynamics don't depend on endeffector position.
  return jac;
}

void
HumanoidRigidBodyDynamics::SetGravityDir(Eigen::Vector3d grav) {
    g_vec = grav / grav.norm();
}

} /* namespace towr */
