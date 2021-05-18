#ifndef VARIABLES_DYNAMIC_PHASE_NODES_H_
#define VARIABLES_DYNAMIC_PHASE_NODES_H_

#include <towr/variables/nodes_variables_phase_based.h>

namespace towr {

/**
 * @brief Phase based nodes variables which allow for different number of polynomials
 */
class NodesVariablesDynamicPhaseBased : public NodesVariablesPhaseBased {
public:
  using Ptr         = std::shared_ptr<NodesVariablesDynamicPhaseBased>;

  /**
   * @brief Constructs a variable set of node variables.
   * @param phase_count  The number of phases (swing, stance) to represent.
   * @param first_phase_constant  Whether first node belongs to a constant phase.
   * @param var_name  The name given to this set of optimization variables.
   * @param n_polys_in_changing_phase  How many polynomials should be used to
   *                                   paramerize each non-constant phase. Must be the same 
   *                                   size as the number of non-constant phases.
   */
  NodesVariablesDynamicPhaseBased (int phase_count,
                            bool first_phase_constant,
                            const std::string& var_name,
                            std::vector<int> n_polys_in_changing_phase);

  virtual ~NodesVariablesDynamicPhaseBased() = default;

};


/**
 * @brief Variables fully defining the endeffector motion.
 *
 * @ingroup Variables
 */
class NodesVariablesDynamicEEMotion : public NodesVariablesDynamicPhaseBased {
public:
  NodesVariablesDynamicEEMotion(int phase_count,
                         bool is_in_contact_at_start,
                         const std::string& name,
                         std::vector<int> n_polys_in_changing_phase);
  virtual ~NodesVariablesDynamicEEMotion() = default;
  NodesVariablesPhaseBased::OptIndexMap GetPhaseBasedEEParameterization ();
};

/**
 * @brief Variables fully defining the endeffector forces.
 *
 * @ingroup Variables
 */
class NodesVariablesDynamicEEForce : public NodesVariablesDynamicPhaseBased {
public:
  NodesVariablesDynamicEEForce(int phase_count,
                         bool is_in_contact_at_start,
                         const std::string& name,
                         std::vector<int> n_polys_in_changing_phase);
  virtual ~NodesVariablesDynamicEEForce() = default;
  NodesVariablesPhaseBased::OptIndexMap GetPhaseBasedEEParameterization ();
};

} /* namespace towr */

#endif /* VARIABLES_DYNAMIC_PHASE_NODES_H_ */
