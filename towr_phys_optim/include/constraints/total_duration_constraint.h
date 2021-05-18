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

#ifndef CONTACT_DURATION_CONSTRAINT_H_
#define CONTACT_DURATION_CONSTRAINT_H_

#include <ifopt/constraint_set.h>

#include <towr/variables/phase_durations.h>

namespace towr {

/**
 * @brief Makes sure all the phase durations sum up to the total time.
 *
 * When optimizing over the phase durations of each foot, this constraint
 * makes sure that:
 * t_phase_1 + ... + t_phase_(n-1) = T_total.
 *
 * Attention: At this point last phase duration is not an optimization variable
 * and a way should be found to optimize over all phases while setting the
 * total duration by constraint and not through hard parameterization.
 *
 * @ingroup Constraints
 */
class ContactDurationConstraint : public ifopt::ConstraintSet {
public:
  using EE = uint;

  ContactDurationConstraint(double T_total, int ee, double min_dur, double max_dur);
  ~ContactDurationConstraint() = default;

  void InitVariableDependedQuantities(const VariablesPtr& x) override;

  VectorXd GetValues() const override;
  VecBound GetBounds() const override;
  void FillJacobianBlock (std::string var_set, Jacobian&) const override;

private:
  PhaseDurations::Ptr phase_durations_;
  double T_total_;
  double min_dur_;
  double max_dur_;
  EE ee_;
};

} // namespace towr

#endif /* CONTACT_DURATION_CONSTRAINT_H_ */
