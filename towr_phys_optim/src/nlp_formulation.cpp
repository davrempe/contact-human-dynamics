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

// #include <towr/nlp_formulation.h>

#include <towr/variables/phase_durations.h>

#include <towr/constraints/base_motion_constraint.h>
#include <towr/constraints/dynamic_constraint.h>
#include <towr/constraints/force_constraint.h>
#include <towr/constraints/range_of_motion_constraint.h>
#include <towr/constraints/terrain_constraint.h>
#include <towr/constraints/spline_acc_constraint.h>

#include <towr/costs/node_cost.h>
#include <towr/variables/nodes_variables_all.h>

#include <towr/variables/variable_names.h>

#include <iostream>

#include "nlp_formulation.h"
#include "models/humanoid.h"
#include "models/humanoid_rigid_body_dynamics.h"
#include "constraints/leg_length_constraint.h"
#include "constraints/humanoid_dynamic_constraint.h"
#include "constraints/height_constraint.h"
#include "terrain/ground_plane.h"
#include "constraints/total_duration_constraint.h"
#include "constraints/ee_dist_constraint.h"

#include "variables/nodes_variables_dynamic_phase_based.h"

namespace towr {

NlpFormulation::NlpFormulation ()
{
  using namespace std;
  cout << "\n";
  cout << "************************************************************\n";
  cout << " TOWR - Trajectory Optimization for Walking Robots (v1.4)\n";
  cout << "                \u00a9 Alexander W. Winkler\n";
  cout << "           https://github.com/ethz-adrl/towr\n";
  cout << "************************************************************";
  cout << "\n\n";
}

NlpFormulation::VariablePtrVec
NlpFormulation::GetVariableSets (SplineHolder& spline_holder, bool optim_dur_)
{
  VariablePtrVec vars;

  auto base_motion = MakeBaseVariables();
  vars.insert(vars.end(), base_motion.begin(), base_motion.end());

  auto ee_motion = MakeEndeffectorVariables();
  vars.insert(vars.end(), ee_motion.begin(), ee_motion.end());

  auto ee_force = MakeForceVariables();
  vars.insert(vars.end(), ee_force.begin(), ee_force.end());

  auto contact_schedule = MakeContactScheduleVariables();

  // stores these readily constructed spline
  spline_holder = SplineHolder(base_motion.at(0), // linear
                               base_motion.at(1), // angular
                               params_.GetBasePolyDurations(),
                               ee_motion,
                               ee_force,
                               contact_schedule,
                               optim_dur_);
  return vars;
}

std::vector<NodesVariables::Ptr>
NlpFormulation::MakeBaseVariables () const
{
  std::vector<NodesVariables::Ptr> vars;

  int n_nodes = params_.GetBasePolyDurations().size() + 1;

  auto spline_lin = std::make_shared<NodesVariablesAll>(n_nodes, k3D, id::base_lin_nodes);

  double x = final_base_.lin.p().x();
  double y = final_base_.lin.p().y();
  double z = final_base_.lin.p().z();
  Vector3d final_pos(x, y, z);

  spline_lin->SetByLinearInterpolation(initial_base_.lin.p(), final_pos, params_.GetTotalTime());
  spline_lin->AddStartBound(kVel, {X,Y,Z}, initial_base_.lin.v());
  spline_lin->AddFinalBound(kVel, params_.bounds_final_lin_vel_, final_base_.lin.v());
  vars.push_back(spline_lin);

  auto spline_ang = std::make_shared<NodesVariablesAll>(n_nodes, k3D, id::base_ang_nodes);
  spline_ang->SetByLinearInterpolation(initial_base_.ang.p(), final_base_.ang.p(), params_.GetTotalTime());
  vars.push_back(spline_ang);

  return vars;
}

std::vector<NodesVariablesPhaseBased::Ptr>
NlpFormulation::MakeEndeffectorVariables () const
{
  std::vector<NodesVariablesPhaseBased::Ptr> vars;

  // Endeffector Motions
  double T = params_.GetTotalTime();
  for (int ee=0; ee<params_.GetEECount(); ee++) {
    // std::cout << ee << std::endl;
    // std::cout << params_.ee_polynomials_per_swing_phase_dynamic_.size() << std::endl;
    auto nodes = std::make_shared<NodesVariablesDynamicEEMotion>(
                                              params_.GetPhaseCount(ee),
                                              params_.ee_in_contact_at_start_.at(ee),
                                              id::EEMotionNodes(ee),
                                              params_.ee_polynomials_per_swing_phase_dynamic_.at(ee));

    // initialize towards final
    double yaw = final_base_.ang.p().z();
    Eigen::Vector3d euler(0.0, 0.0, yaw);
    Eigen::Matrix3d w_R_b = EulerConverter::GetRotationMatrixBaseToWorld(euler);
    Vector3d final_ee_pos_W = final_base_.lin.p();
    double x = final_ee_pos_W.x();
    double y = final_ee_pos_W.y();
    double z = terrain_->GetHeight(x,y);
    nodes->SetByLinearInterpolation(initial_ee_W_.at(ee), Vector3d(x,y,z), T);

    vars.push_back(nodes);
  }

  return vars;
}

std::vector<NodesVariablesPhaseBased::Ptr>
NlpFormulation::MakeForceVariables () const
{
  std::vector<NodesVariablesPhaseBased::Ptr> vars;

  double T = params_.GetTotalTime();
  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto nodes = std::make_shared<NodesVariablesDynamicEEForce>(
                                              params_.GetPhaseCount(ee),
                                              params_.ee_in_contact_at_start_.at(ee),
                                              id::EEForceNodes(ee),
                                              params_.force_polynomials_per_stance_phase_dynamic_.at(ee));

    double m = model_.dynamic_model_->m();
    double g = model_.dynamic_model_->g();

    Vector3d f_stance(0.0, 0.0, m*g/params_.GetEECount());
    nodes->SetByLinearInterpolation(f_stance, f_stance, T); // stay constant
    vars.push_back(nodes);
  }

  return vars;
}

std::vector<PhaseDurations::Ptr>
NlpFormulation::MakeContactScheduleVariables () const
{
  std::vector<PhaseDurations::Ptr> vars;

  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto var = std::make_shared<PhaseDurations>(ee,
                                                params_.ee_phase_durations_.at(ee),
                                                params_.ee_in_contact_at_start_.at(ee),
                                                params_.bound_phase_duration_.first,
                                                params_.bound_phase_duration_.second);
    vars.push_back(var);
  }

  return vars;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::GetConstraints(const SplineHolder& spline_holder) const
{
  ContraintPtrVec constraints;
  for (auto name : params_.constraints_)
    for (auto c : GetConstraint(name, spline_holder))
      constraints.push_back(c);

  return constraints;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::GetConstraint (Parameters::ConstraintName name,
                           const SplineHolder& s) const
{
  switch (name) {
    case Parameters::Dynamic:        return MakeDynamicConstraint(s);
    case Parameters::EndeffectorRom: return MakeRangeOfMotionLegConstraint(s);
    case Parameters::TotalTime:      return MakeTotalTimeConstraint();
    case Parameters::Terrain:        return MakeTerrainConstraint();
    case Parameters::Force:          return MakeForceConstraint();
    case Parameters::BaseAcc:        return MakeBaseAccConstraint(s);
    case Parameters::Height:         return MakeHeightConstraint(s);
    case Parameters::HeelDist:       return MakeHeelDistConstraints(s);
    default: throw std::runtime_error("constraint not defined!");
  }
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeDynamicConstraint(const SplineHolder& s) const
{
  auto constraint = std::make_shared<HumanoidDynamicConstraint>(std::static_pointer_cast<HumanoidRigidBodyDynamics>(model_.dynamic_model_),
                                                        params_.GetTotalTime(),
                                                        params_.dt_constraint_dynamic_,
                                                        s);
  return {constraint};
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeHeelDistConstraints (const SplineHolder& s) const
{
  ContraintPtrVec c;

  auto heeldist_left = std::make_shared<EEDistConstraint>(params_.GetTotalTime(),
                                                         params_.dt_constraint_range_of_motion_,
                                                         0, 2, s,
                                                         params_.heel_dist_);

  auto heeldist_right = std::make_shared<EEDistConstraint>(params_.GetTotalTime(),
                                                         params_.dt_constraint_range_of_motion_,
                                                         1, 3, s,
                                                         params_.heel_dist_);

  c.push_back(heeldist_left);
  c.push_back(heeldist_right);

  return c;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeRangeOfMotionLegConstraint (const SplineHolder& s) const
{
  ContraintPtrVec c;

  // only do this for toes
  for (int ee=0; ee<params_.GetEECount(); ee++) {
    // right now this assumes our model is always humanoid
    auto rom = std::make_shared<LegLengthConstraint>(std::static_pointer_cast<HumanoidKinematicModel>(model_.kinematic_model_),
                                                         params_.GetTotalTime(),
                                                         params_.dt_constraint_range_of_motion_,
                                                         ee,
                                                         s);
    c.push_back(rom);
  }

  return c;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeHeightConstraint (const SplineHolder& s) const
{
  ContraintPtrVec c;

  for (int ee=0; ee<params_.GetEECount(); ee++) {
    // right now this assumes our model is always humanoid
    auto height = std::make_shared<HeightConstraint>(    params_.GetTotalTime(),
                                                         params_.dt_constraint_height_,
                                                         std::static_pointer_cast<GroundPlane>(terrain_),
                                                         ee,
                                                         s.ee_motion_.at(ee)
                                                         );
    c.push_back(height);
  }

  return c;
}



NlpFormulation::ContraintPtrVec
NlpFormulation::MakeTotalTimeConstraint () const
{
  ContraintPtrVec c;
  double T = params_.GetTotalTime();

  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto duration_constraint = std::make_shared<ContactDurationConstraint>(T, ee, 
                                                params_.bound_phase_duration_.first,
                                                params_.bound_phase_duration_.second);
    c.push_back(duration_constraint);
  }

  return c;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeTerrainConstraint () const
{
  ContraintPtrVec constraints;

  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto c = std::make_shared<TerrainConstraint>(terrain_, id::EEMotionNodes(ee));
    constraints.push_back(c);
  }

  return constraints;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeForceConstraint () const
{
  ContraintPtrVec constraints;

  for (int ee=0; ee<params_.GetEECount(); ee++) {
    auto c = std::make_shared<ForceConstraint>(terrain_,
                                               params_.force_limit_in_normal_direction_,
                                               ee);
    constraints.push_back(c);
  }

  return constraints;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::MakeBaseAccConstraint (const SplineHolder& s) const
{
  ContraintPtrVec constraints;

  constraints.push_back(std::make_shared<SplineAccConstraint>
                        (s.base_linear_, id::base_lin_nodes));

  constraints.push_back(std::make_shared<SplineAccConstraint>
                        (s.base_angular_, id::base_ang_nodes));

  return constraints;
}

NlpFormulation::ContraintPtrVec
NlpFormulation::GetCosts() const
{
  ContraintPtrVec costs;
  for (const auto& pair : params_.costs_)
    for (auto c : GetCost(pair.first, pair.second))
      costs.push_back(c);

  return costs;
}

NlpFormulation::CostPtrVec
NlpFormulation::GetCost(const Parameters::CostName& name, double weight) const
{
  switch (name) {
    default: throw std::runtime_error("cost not defined!");
  }
}

} /* namespace towr */
