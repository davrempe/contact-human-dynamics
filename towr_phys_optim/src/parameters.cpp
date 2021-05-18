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

#include <towr/variables/cartesian_dimensions.h>

#include <algorithm>
#include <numeric>      // std::accumulate
#include <math.h>       // fabs
#include <cassert>

#include "parameters.h"

namespace towr {

Parameters::Parameters ()
{
  // constructs optimization variables
  duration_base_polynomial_ = 0.1;

  add_polys_after_dur_ = 2.0; // number of seconds allowed for a single phase before additional polynomials are added
  force_polynomials_per_stance_phase_ = 6;
  ee_polynomials_per_swing_phase_ = 6; // so step can at least lift leg

  // parameters related to specific constraints (only used when it is added as well)
  force_limit_in_normal_direction_ = 1000;
  dt_constraint_range_of_motion_ = 0.08;
  dt_constraint_height_ = 0.1;
  dt_constraint_dynamic_ = 0.1; // NOTE must be <= duration_base_polynomial
  bound_phase_duration_ = std::make_pair(0.0, 500.0);  // used only when optimizing phase durations, so gait

  // a minimal set of basic constraints
  AddInitConstraints();

  // bounds on final 6DoF base state
  bounds_final_lin_pos_ = {X,Y,Z};
  bounds_final_lin_vel_ = {X,Y,Z};
  bounds_final_ang_pos_ = {X,Y,Z};
  bounds_final_ang_vel_ = {X,Y,Z};
}

void 
Parameters::AddInitConstraints() {
    constraints_.push_back(BaseAcc); // so accelerations don't jump between polynomials
}

void 
Parameters::AddLegConstraints() {
    constraints_.push_back(Terrain); // ensures on terrain when supposed to be in contact
    constraints_.push_back(EndeffectorRom); //Ensures that the range of motion is respected at discrete times.
}

void 
Parameters::AddHeelConstraints() {
    constraints_.push_back(HeelDist);
}

void
Parameters::AddHeightConstraints() {
    constraints_.push_back(Height); // ensures always above terrain between contacts
}

void 
Parameters::AddDynamicsConstraints() {
    constraints_.push_back(Dynamic); //Ensures that the dynamic model is fullfilled at discrete times.
    constraints_.push_back(Force); // ensures unilateral forces and inside the friction cone.
}

void Parameters::ClearConstraints() {
    constraints_.clear();
}

void
Parameters::OptimizePhaseDurations ()
{
  constraints_.push_back(TotalTime);
}

Parameters::VecTimes
Parameters::GetBasePolyDurations () const
{
  std::vector<double> base_spline_timings_;
  double dt = duration_base_polynomial_;
  double t_left = GetTotalTime ();

  double eps = 1e-10; // since repeated subtraction causes inaccuracies
  while (t_left > eps) {
    double duration = t_left>dt?  dt : t_left;
    base_spline_timings_.push_back(duration);

    t_left -= dt;
  }

  return base_spline_timings_;
}

int
Parameters::GetPhaseCount(EEID ee) const
{
  return ee_phase_durations_.at(ee).size();
}

int
Parameters::GetEECount() const
{
  return ee_in_contact_at_start_.size();
}

double
Parameters::GetTotalTime () const
{
  std::vector<double> T_feet;

  for (const auto& v : ee_phase_durations_)
    T_feet.push_back(std::accumulate(v.begin(), v.end(), 0.0));

  // safety check that all feet durations sum to same value
  double T = T_feet.empty()? 0.0 : T_feet.front(); // take first foot as reference
  for (double Tf : T_feet)
    assert(fabs(Tf - T) < 1e-6);

  return T;
}

bool
Parameters::IsOptimizeTimings () const
{
  // if total time is constrained, then timings are optimized
  ConstraintName c = TotalTime;
  auto v = constraints_; // shorthand
  return std::find(v.begin(), v.end(), c) != v.end();
}

} // namespace towr
