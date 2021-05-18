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

#ifndef TOWR_OPTIMIZATION_PARAMETERS_H_
#define TOWR_OPTIMIZATION_PARAMETERS_H_

#include <vector>
#include <array>
#include <utility> // std::pair, std::make_pair

namespace towr {

/**
 * @defgroup Parameters
 * @brief %Parameters to tune the optimization problem.
 *
 */
class Parameters {
public:
  /**
   * @brief Identifiers to be used to add certain constraints to the
   * optimization problem.
   */
  enum ConstraintName { Dynamic,        ///< sets DynamicConstraint
                        EndeffectorRom, ///< sets RangeOfMotionConstraint
                        TotalTime,      ///< sets TotalDurationConstraint
                        Terrain,        ///< sets TerrainConstraint
                        Force,          ///< sets ForceConstraint
                        BaseAcc,         ///< sets SplineAccConstraint
                        Height,          // HeightConstraint
                        HeelDist         // EEDistConstraint
  };
  /**
   *  @brief Indentifiers to be used to add certain costs to the optimization
   *  problem.
   */
  enum CostName       {
  };

  using CostWeights      = std::vector<std::pair<CostName, double>>;
  using UsedConstraints  = std::vector<ConstraintName>;
  using VecTimes         = std::vector<double>;
  using EEID             = unsigned int;

  /**
   * @brief Default parameters to get started.
   */
  Parameters();
  virtual ~Parameters() = default;

  void AddInitConstraints();
  void AddLegConstraints();
  void AddHeightConstraints();
  void AddDynamicsConstraints();
  void AddHeelConstraints();
  void ClearConstraints();

  /// Number and initial duration of each foot's swing and stance phases.
  std::vector<VecTimes> ee_phase_durations_;

  /// True if the foot is initially in contact with the terrain.
  std::vector<bool> ee_in_contact_at_start_;

  /// Which constraints should be used in the optimization problem.
  UsedConstraints constraints_;

  /// Which costs should be used in the optimiation problem.
  CostWeights costs_;

  /// Interval at which the dynamic constraint is enforced.
  double dt_constraint_dynamic_;

  /// Interval at which the range of motion constraint is enforced.
  double dt_constraint_range_of_motion_;

  /// Interval at which the range of motion constraint is enforced.
  double dt_constraint_height_;

  /// Fixed duration of each cubic polynomial describing the base motion.
  double duration_base_polynomial_;

  /// Number of polynomials to parameterize foot movement during swing phases.
  double add_polys_after_dur_;
  int ee_polynomials_per_swing_phase_;
  std::vector< std::vector<int> > ee_polynomials_per_swing_phase_dynamic_;

  /// Number of polynomials to parameterize each contact force during stance phase.
  int force_polynomials_per_stance_phase_;
  std::vector< std::vector<int> > force_polynomials_per_stance_phase_dynamic_;

  /// The maximum allowable force [N] in normal direction
  double force_limit_in_normal_direction_;

  /// distance form toe ot heel
  double heel_dist_;



  /// which dimensions (x,y,z) of the final base state should be bounded
  std::vector<int> bounds_final_lin_pos_,
                   bounds_final_lin_vel_,
                   bounds_final_ang_pos_,
                   bounds_final_ang_vel_;

  /** Minimum and maximum time [s] for each phase (swing,stance).
   *
   *  Only used when optimizing over phase durations.
   *  Make sure max time is less than total duration of trajectory, or segfault.
   *  limiting this range can help convergence when optimizing gait.
   */
  std::pair<double,double> bound_phase_duration_;

  /// Specifies that timings of all feet, so the gait, should be optimized.
  void OptimizePhaseDurations();

  /// The durations of each base polynomial in the spline (lin+ang).
  VecTimes GetBasePolyDurations() const;

  /// The number of phases allowed for endeffector ee.
  int GetPhaseCount(EEID ee) const;

  /// True if the phase durations should be optimized over.
  bool IsOptimizeTimings() const;

  /// The number of endeffectors.
  int GetEECount() const;

  /// Total duration [s] of the motion.
  double GetTotalTime() const;
};

} // namespace towr

#endif /* TOWR_OPTIMIZATION_PARAMETERS_H_ */
