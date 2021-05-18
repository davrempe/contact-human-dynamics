#include "variables/nodes_variables_dynamic_phase_based.h"

#include <towr/variables/cartesian_dimensions.h>

#include <iostream>
#include <math.h>

namespace towr {

std::vector<NodesVariablesPhaseBased::PolyInfo>
BuildDynamicPolyInfos (int phase_count, bool first_phase_constant,
                std::vector<int>  n_polys_in_changing_phase)
{
  using PolyInfo = NodesVariablesPhaseBased::PolyInfo;
  std::vector<PolyInfo> polynomial_info;

  bool phase_constant = first_phase_constant;

  int const_phase_count = 0;
  for (int i=0; i<phase_count; ++i) {
    if (phase_constant)
      polynomial_info.push_back(PolyInfo(i,0,1, true));
    else {
      for (int j=0; j<n_polys_in_changing_phase.at(const_phase_count); ++j) {
         polynomial_info.push_back(PolyInfo(i,j,n_polys_in_changing_phase.at(const_phase_count), false));
      }
      const_phase_count++;
    }

    phase_constant = !phase_constant; // constant and non-constant phase alternate
  }

  return polynomial_info;
}

NodesVariablesDynamicPhaseBased::NodesVariablesDynamicPhaseBased (int phase_count,
                                                    bool first_phase_constant,
                                                    const std::string& name,
                                                    std::vector<int> n_polys_in_changing_phase)
    :NodesVariablesPhaseBased(phase_count, first_phase_constant, name, 1) // just pass constant 1, doesn't matter, will overwrite
{
  int num_change_phases;
  if (!first_phase_constant) {
      num_change_phases = (int)ceil(phase_count / 2.0);
  } else {
      num_change_phases = (int)floor(phase_count / 2.0);
  }
  assert(n_polys_in_changing_phase.size() == num_change_phases);

  // need to overwrite all the stuff NodesVariablesPhaseBased constructed with variable number of polys
  polynomial_info_ = BuildDynamicPolyInfos(phase_count, first_phase_constant, n_polys_in_changing_phase);

  n_dim_ = k3D;
  int n_nodes = polynomial_info_.size()+1;
  nodes_  = std::vector<Node>(n_nodes, Node(n_dim_));
}

NodesVariablesDynamicEEMotion::NodesVariablesDynamicEEMotion(int phase_count,
                                               bool is_in_contact_at_start,
                                               const std::string& name,
                                               std::vector<int> n_polys_in_changing_phase)
    :NodesVariablesDynamicPhaseBased(phase_count,
                              is_in_contact_at_start, // contact phase for motion is constant
                              name,
                              n_polys_in_changing_phase)
{
  index_to_node_value_info_ = GetPhaseBasedEEParameterization();
  SetNumberOfVariables(index_to_node_value_info_.size());
}

NodesVariablesPhaseBased::OptIndexMap
NodesVariablesDynamicEEMotion::GetPhaseBasedEEParameterization ()
{
  OptIndexMap index_map;

  int idx = 0; // index in variables set
  for (int node_id=0; node_id<nodes_.size(); ++node_id) {
    // swing node:
    if (!IsConstantNode(node_id)) {
      for (int dim=0; dim<GetDim(); ++dim) {
        // intermediate way-point position of swing motion are optimized
        index_map[idx++].push_back(NodeValueInfo(node_id, kPos, dim));
        // velocity in x,y,z dimension during swing fully optimized.
        index_map[idx++].push_back(NodeValueInfo(node_id, kVel, dim));
      }
    }
    // stance node (next one will also be stance, so handle that one too):
    else {
      // ensure that foot doesn't move by not even optimizing over velocities
      nodes_.at(node_id).at(kVel).setZero();
      nodes_.at(node_id+1).at(kVel).setZero();

      // position of foot is still an optimization variable used for
      // both start and end node of that polynomial
      for (int dim=0; dim<GetDim(); ++dim) {
        index_map[idx].push_back(NodeValueInfo(node_id,   kPos, dim));
        index_map[idx].push_back(NodeValueInfo(node_id+1, kPos, dim));
        idx++;
      }

      node_id += 1; // already added next constant node, so skip
    }
  }

  return index_map;
}

NodesVariablesDynamicEEForce::NodesVariablesDynamicEEForce(int phase_count,
                                              bool is_in_contact_at_start,
                                              const std::string& name,
                                              std::vector<int>  n_polys_in_changing_phase)
    :NodesVariablesDynamicPhaseBased(phase_count,
                              !is_in_contact_at_start, // contact phase for force is non-constant
                              name,
                              n_polys_in_changing_phase)
{
  index_to_node_value_info_ = GetPhaseBasedEEParameterization();
  SetNumberOfVariables(index_to_node_value_info_.size());
}

NodesVariablesPhaseBased::OptIndexMap
NodesVariablesDynamicEEForce::GetPhaseBasedEEParameterization ()
{
  OptIndexMap index_map;

  int idx = 0; // index in variables set
  for (int id=0; id<nodes_.size(); ++id) {
    // stance node:
    // forces can be created during stance, so these nodes are optimized over.
    if (!IsConstantNode(id)) {
      for (int dim=0; dim<GetDim(); ++dim) {
        index_map[idx++].push_back(NodeValueInfo(id, kPos, dim));
        index_map[idx++].push_back(NodeValueInfo(id, kVel, dim));
      }
    }
    // swing node (next one will also be swing, so handle that one too)
    else {
      // forces can't exist during swing phase, so no need to be optimized
      // -> all node values simply set to zero.
      nodes_.at(id).at(kPos).setZero();
      nodes_.at(id+1).at(kPos).setZero();

      nodes_.at(id).at(kVel).setZero();
      nodes_.at(id+1).at(kVel).setZero();

      id += 1; // already added next constant node, so skip
    }
  }

  return index_map;
}

} /* namespace towr */
