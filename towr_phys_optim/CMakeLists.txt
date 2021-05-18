cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(physical_mocap VERSION 0.0.2)

find_package(towr 1.2 REQUIRED)
find_package(ifopt 2.0.1 REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
find_package(GFlags REQUIRED)

include_directories(
    include
)

#add_executable(towr_example towr_example.cpp) # Your custom variables, costs and constraints added to TOWR
#target_link_libraries(towr_example PUBLIC towr::towr) # adds include directories and libraries
#target_link_libraries(towr_example PRIVATE ifopt::ifopt_ipopt)

add_executable(phys_optim phys_optim.cpp
                          src/terrain/ground_plane.cpp
                          src/models/humanoid_rigid_body_dynamics.cpp
                          src/costs/data_cost.cpp
                          src/nlp_formulation.cpp
                          src/parameters.cpp
                          src/constraints/leg_length_constraint.cpp
                          src/constraints/humanoid_dynamic_constraint.cpp
                          src/constraints/height_constraint.cpp
                          src/costs/duration_cost.cpp
                          src/constraints/total_duration_constraint.cpp
                          src/costs/vel_smooth_cost.cpp
                          src/constraints/ee_dist_constraint.cpp
                          src/variables/nodes_variables_dynamic_phase_based.cpp
                          )
target_link_libraries(phys_optim PUBLIC towr::towr ${GFLAGS_LIBRARY}) # adds include directories and libraries
target_link_libraries(phys_optim PRIVATE ifopt::ifopt_ipopt)
