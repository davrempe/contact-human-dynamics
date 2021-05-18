#include <cmath>
#include <iostream>
#include <fstream>

#include <gflags/gflags.h>

#include <towr/variables/variable_names.h>
#include <ifopt/ipopt_solver.h>
#include <towr/terrain/examples/height_map_examples.h>
#include <towr/terrain/height_map.h> 

#include "models/humanoid.h"
#include "terrain/ground_plane.h"
#include "costs/data_cost.h"
#include "costs/duration_cost.h"
#include "costs/vel_smooth_cost.h"
#include "constraints/ee_dist_constraint.h"

#include "nlp_formulation.h"

using namespace towr;

DEFINE_string(out_dir, "sol_out", "Directory to write out results to (must already exist).");
DEFINE_string(in_dir, "./", "directory to read in input information from (skeleton, motion, and terrain)");
DEFINE_int32(nframes, 100, "the number of frames to read in");

DEFINE_double(w_com_lin, 0.4, "Weight for COM linear position during physical optimization.");
DEFINE_double(w_com_ang, 1.7, "Weight for COM angular orientation during physical optimization.");
DEFINE_double(w_ee, 0.3, "Weight for end-effector position during physical optimization.");
DEFINE_double(w_smooth, 0.1, "Weight end-effector velocity smoothing during physical optimization.");
DEFINE_double(w_dur, 0.1, "Weight total duration cost physical optimization.");

void check_flags(int argc, char* argv[]) {
#ifdef GFLAGS_NAMESPACE
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
#else
    google::ParseCommandLineFlags(&argc, &argv, true);
#endif
    std::cout << "Out Dir: " << FLAGS_out_dir << std::endl;
    std::cout << "Input Directory: " << FLAGS_in_dir << std::endl;
    std::cout << "num frames: " << FLAGS_nframes << std::endl;

    std::cout << "Optim weights (" << FLAGS_w_com_lin << ", " << FLAGS_w_com_ang << ", " <<
              FLAGS_w_ee << ", " << FLAGS_w_smooth << ")" << std::endl;
    std::cout << "Duration cost weight " << FLAGS_w_dur << std::endl;
}

void PrintDurationResults(SplineHolder& solution, std::vector<double>& left_durations, std::vector<double>& right_durations) {
  std::cout << "OG\n";
  for (int i = 0; i < left_durations.size(); i++) {
    std::cout << left_durations.at(i) << std::endl;
  }
  std::cout << "NEW\n";
  std::cout << solution.phase_durations_.at(0)->GetValues() << std::endl;
  std::cout << "OG\n";
  for (int i = 0; i < right_durations.size(); i++) {
    std::cout << right_durations.at(i) << std::endl;
  }
  std::cout << "NEW\n";
  std::cout << solution.phase_durations_.at(1)->GetValues() << std::endl;
}

int SaveSolution(SplineHolder& solution, std::string out_path, double dt, Parameters& params) {
  std::cout << "Saving optimized solution to " << out_path << "..." << std::endl;
  std::cout.precision(5);
  std::ofstream f(out_path);
  assert(f.good());
  f.precision(10);
  double tot_time = solution.base_linear_->GetTotalTime();

  int num_frames = (int)((tot_time + 1e-5) / dt) + 1; // + 1 for time 0.0
  // first some metadata
  f << "dt" << std::endl;
  f << dt << std::endl;
  f << "num_frames" << std::endl;
  f << num_frames << std::endl;
  f << "num_feet" << std::endl;
  f << params.GetEECount() << std::endl;


  // then per frame information
  f << "base_lin" << std::endl;
  double t = 0.0;
  while (t <= tot_time + 1e-5) {
    // std::cout << t << std::endl;
    Eigen::Vector3d cur_p = solution.base_linear_->GetPoint(t).p().transpose();
    f << cur_p(X) << " " << cur_p(Y) << " " << cur_p(Z);
    t += dt;
    if (t <= tot_time + 1e-5) f << " ";
  }
  f << std::endl;

  f << "base_ang" << std::endl;
  t = 0.0;
  while (t <= tot_time + 1e-5) {
    Eigen::Vector3d rad = solution.base_angular_->GetPoint(t).p();
    Eigen::Vector3d deg = (rad/M_PI*180).transpose();
    f << deg(X) << " " << deg(Y) << " " << deg(Z);
    t += dt;
    if (t <= tot_time + 1e-5) f << " ";
  }
  f << std::endl;

  // left toe, right toe, left heel, right heel
  for (int i = 0; i < params.GetEECount(); i++) {
    f << "foot" << i << "_pos" << std::endl;
    t = 0.0;
    while (t <= tot_time + 1e-5) {
      Eigen::Vector3d cur_p = solution.ee_motion_.at(i)->GetPoint(t).p().transpose();
      f << cur_p(X) << " " << cur_p(Y) << " " << cur_p(Z);
      t += dt;
      if (t <= tot_time + 1e-5) f << " ";
    }
    f << std::endl;
  }

  // left toe, right toe, left heel, right heel
  for (int i = 0; i < params.GetEECount(); i++) {
    f << "foot" << i << "_force" << std::endl;
    t = 0.0;
    while (t <= tot_time + 1e-5) {
      Eigen::Vector3d cur_f = solution.ee_force_.at(i)->GetPoint(t).p().transpose(); 
      f << cur_f(X) << " " << cur_f(Y) << " " << cur_f(Z);
      t += dt;
      if (t <= tot_time + 1e-5) f << " ";
    }
    f << std::endl;
  }

  // left toe, right toe, left heel, right heel
  for (int i = 0; i < params.GetEECount(); i++) {
    f << "foot" << i << "_contact" << std::endl;
    t = 0.0;
    while (t <= tot_time + 1e-5) {
      bool contact = solution.phase_durations_.at(i)->IsContactPhase(t);
      f << contact;
      t += dt;
      if (t <= tot_time + 1e-5) f << " ";
    }
    f << std::endl;
  }

}

void SaveSuccessLog(std::string out_path, bool dynamics_succeed, bool durations_succeed) {
  std::ofstream f(out_path);
  assert(f.good());

  f << "dynamics " << dynamics_succeed << std::endl;
  f << "durations " << durations_succeed << std::endl;

  return;
}

void ReadNx3Line(std::ifstream& f, int num_frames, Eigen::MatrixXd& mat_out) {
  double xin, yin, zin;
  for (int i = 0; i < num_frames; i++) {
   f >> xin;
   f >> yin;
   f >> zin;
   mat_out(i, 0) = xin;
   mat_out(i, 1) = yin;
   mat_out(i, 2) = zin;
  }

  return;
}

void ReadSkeletonInfo(std::string skel_info_path, int num_frames, Eigen::MatrixXd& left_hip_offset, Eigen::MatrixXd& right_hip_offset, 
                      double& max_leg_length, double& max_heel_length, double& heel_dist, double& body_mass, Eigen::MatrixXd& inertia) {
  std::cout << "Reading skeleton info from " << skel_info_path << "..." << std::endl;

  std::ifstream f(skel_info_path);
  assert(f.good());

  ReadNx3Line(f, num_frames, left_hip_offset);
  ReadNx3Line(f, num_frames, right_hip_offset);

  f >> max_leg_length;
  f >> max_heel_length;
  f >> heel_dist;
  f >> body_mass;
  for (int i = 0; i < num_frames; i++) {
    for (int j = 0; j < 6; j++) {
      f >> inertia(i, j);
    }
  }
}

void ReadMotionInfo(std::string motion_info_path, int num_frames, double& dt, Eigen::MatrixXd& base_lin_init, Eigen::MatrixXd& base_ang_init,
                      Eigen::MatrixXd& ee0_init, Eigen::MatrixXd& ee1_init, Eigen::MatrixXd& heel0_init, Eigen::MatrixXd& heel1_init) {
  std::cout << "Reading motion info from " << motion_info_path << "..." << std::endl;

  std::ifstream f(motion_info_path);
  assert(f.good());

  f >> dt;

  ReadNx3Line(f, num_frames, base_lin_init);
  ReadNx3Line(f, num_frames, base_ang_init);
  ReadNx3Line(f, num_frames, ee0_init);
  ReadNx3Line(f, num_frames, heel0_init);
  ReadNx3Line(f, num_frames, ee1_init);
  ReadNx3Line(f, num_frames, heel1_init);

  return;

}

void ReadTerrainInfo(std::string terrain_info_path, Eigen::Vector3d& normal, Eigen::Vector3d& point) {
  std::cout << "Reading terrain info from " << terrain_info_path << "..." << std::endl;

  std::ifstream f(terrain_info_path);
  assert(f.good());

  f >> normal(0);
  f >> normal(1);
  f >> normal(2);
  f >> point(0);
  f >> point(1);
  f >> point(2);

  return;
}

void ReadContactInfo(std::string contact_info_path, 
                      bool& left_toe_start_contact, bool& left_heel_start_contact, 
                      bool& right_toe_start_contact, bool& right_heel_start_contact,
                      std::vector<double>& left_toe_durations, std::vector<double>& left_heel_durations,
                      std::vector<double>& right_toe_durations, std::vector<double>& right_heel_durations) {
  std::cout << "Reading contact info from " << contact_info_path << "..." << std::endl;

  std::ifstream f(contact_info_path);
  assert(f.good());

  f >> left_toe_start_contact;
  int num_phases;
  f >> num_phases;
  double cur_dur;
  for (int i = 0; i < num_phases; i++) {
    f >> cur_dur;
    left_toe_durations.push_back(cur_dur);
  }

  f >> left_heel_start_contact;
  f >> num_phases;
  for (int i = 0; i < num_phases; i++) {
    f >> cur_dur;
    left_heel_durations.push_back(cur_dur);
  }

  f >> right_toe_start_contact;
  f >> num_phases;
  for (int i = 0; i < num_phases; i++) {
    f >> cur_dur;
    right_toe_durations.push_back(cur_dur);
  }

  f >> right_heel_start_contact;
  f >> num_phases;
  for (int i = 0; i < num_phases; i++) {
    f >> cur_dur;
    right_heel_durations.push_back(cur_dur);
  }

  return;
}

void PrintDurations(std::vector<double>& left_toe_durations, std::vector<double>& left_heel_durations,
                    std::vector<double>& right_toe_durations, std::vector<double>& right_heel_durations) {
  for (int i = 0; i < left_toe_durations.size(); i++) {
    std::cout << left_toe_durations.at(i) << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < left_heel_durations.size(); i++) {
    std::cout << left_heel_durations.at(i) << " ";
  }
  std::cout << std::endl;
  for (int i = 0; i < right_toe_durations.size(); i++) {
    std::cout << right_toe_durations.at(i) << " "; 
  }
  std::cout << std::endl;
  for (int i = 0; i < right_heel_durations.size(); i++) {
    std::cout << right_heel_durations.at(i) << " "; 
  }
  std::cout << std::endl;
}

std::vector<int> GetPolyChangingPhase(bool start_constant, std::vector<double> durations, double max_dur, int n_polys_per_change) {
    int num_change_phases;
    if (!start_constant) {
        num_change_phases = (int)ceil(durations.size() / 2.0);
    } else {
        num_change_phases = (int)floor(durations.size() / 2.0);
    }
    std::vector<int> change_phase_polys;

    bool is_constant = start_constant;
    double num_polys_per_s = n_polys_per_change / max_dur;
    for (int i = 0; i < durations.size(); i++) {
      if (!is_constant) {
        int num_polys = n_polys_per_change;
        if (durations.at(i) > max_dur) {
          num_polys += (int)ceil((durations.at(i) - max_dur) * num_polys_per_s);
        } 
        change_phase_polys.push_back(num_polys);
      }
      is_constant = !is_constant;
    }

    return change_phase_polys;
}

void AddDataCosts(ifopt::Problem& nlp, const SplineHolder& solution, double dt, Eigen::MatrixXd& base_lin_init, Eigen::MatrixXd& base_ang_init,
                      Eigen::MatrixXd& ee0_init, Eigen::MatrixXd& ee1_init, Eigen::MatrixXd& heel0_init, Eigen::MatrixXd& heel1_init,
                      double w_base_lin, double w_base_ang, double w_ee) {
  // base
  auto base_lin_data_cost = std::make_shared<DataCost>(id::base_lin_nodes, dt, solution.base_linear_, false, base_lin_init, w_base_lin);
  nlp.AddCostSet(base_lin_data_cost);
  auto base_ang_data_cost = std::make_shared<DataCost>(id::base_ang_nodes, dt, solution.base_angular_, false, base_ang_init, w_base_ang);
  nlp.AddCostSet(base_ang_data_cost);
  // toes
  auto ee0_data_cost = std::make_shared<DataCost>(id::EEMotionNodes(0), dt, solution.ee_motion_.at(0), true, ee0_init, w_ee);
  nlp.AddCostSet(ee0_data_cost);
  auto ee1_data_cost = std::make_shared<DataCost>(id::EEMotionNodes(1), dt, solution.ee_motion_.at(1), true, ee1_init, w_ee);
  nlp.AddCostSet(ee1_data_cost);
  // heels
  auto heel0_data_cost = std::make_shared<DataCost>(id::EEMotionNodes(2), dt, solution.ee_motion_.at(2), true, heel0_init, w_ee);
  nlp.AddCostSet(heel0_data_cost);
  auto heel1_data_cost = std::make_shared<DataCost>(id::EEMotionNodes(3), dt, solution.ee_motion_.at(3), true, heel1_init, w_ee);
  nlp.AddCostSet(heel1_data_cost);
  return;
}

void AddVelocitySmoothCosts(ifopt::Problem& nlp, const SplineHolder& solution, double dt,
                            double w_base_lin, double w_base_ang, double w_ee) {
  // base
  auto base_lin_smooth_cost = std::make_shared<VelSmoothCost>(id::base_lin_nodes, dt, solution.base_linear_, kPos, w_base_lin, false);
  nlp.AddCostSet(base_lin_smooth_cost);
  auto base_ang_smooth_cost = std::make_shared<VelSmoothCost>(id::base_ang_nodes, dt, solution.base_angular_, kPos, w_base_ang, false);
  nlp.AddCostSet(base_ang_smooth_cost);
  // toes
  auto ee0_smooth_cost = std::make_shared<VelSmoothCost>(id::EEMotionNodes(0), dt, solution.ee_motion_.at(0), kPos, w_ee, true);
  nlp.AddCostSet(ee0_smooth_cost);
  auto ee1_smooth_cost = std::make_shared<VelSmoothCost>(id::EEMotionNodes(1), dt, solution.ee_motion_.at(1), kPos, w_ee, true);
  nlp.AddCostSet(ee1_smooth_cost);
  // heels
  auto heel0_smooth_cost = std::make_shared<VelSmoothCost>(id::EEMotionNodes(2), dt, solution.ee_motion_.at(2), kPos, w_ee, true);
  nlp.AddCostSet(heel0_smooth_cost);
  auto heel1_smooth_cost = std::make_shared<VelSmoothCost>(id::EEMotionNodes(3), dt, solution.ee_motion_.at(3), kPos, w_ee, true);
  nlp.AddCostSet(heel1_smooth_cost);
  return;
}

void AddAccelSmoothCosts(ifopt::Problem& nlp, const SplineHolder& solution, double dt,
                            double w_base_lin, double w_base_ang, double w_ee) {
  // base
  auto base_lin_smooth_cost = std::make_shared<VelSmoothCost>(id::base_lin_nodes, dt, solution.base_linear_, kVel, w_base_lin, false);
  nlp.AddCostSet(base_lin_smooth_cost);
  auto base_ang_smooth_cost = std::make_shared<VelSmoothCost>(id::base_ang_nodes, dt, solution.base_angular_, kVel, w_base_ang, false);
  nlp.AddCostSet(base_ang_smooth_cost);
  // toes
  auto ee0_smooth_cost = std::make_shared<VelSmoothCost>(id::EEMotionNodes(0), dt, solution.ee_motion_.at(0), kVel, w_ee, true);
  nlp.AddCostSet(ee0_smooth_cost);
  auto ee1_smooth_cost = std::make_shared<VelSmoothCost>(id::EEMotionNodes(1), dt, solution.ee_motion_.at(1), kVel, w_ee, true);
  nlp.AddCostSet(ee1_smooth_cost);
  // heels
  auto heel0_smooth_cost = std::make_shared<VelSmoothCost>(id::EEMotionNodes(2), dt, solution.ee_motion_.at(2), kVel, w_ee, true);
  nlp.AddCostSet(heel0_smooth_cost);
  auto heel1_smooth_cost = std::make_shared<VelSmoothCost>(id::EEMotionNodes(3), dt, solution.ee_motion_.at(3), kVel, w_ee, true);
  nlp.AddCostSet(heel1_smooth_cost);
  return;
}

int main(int argc, char* argv[])
{
  check_flags(argc, argv);

  // read in information to formulate the problem
  std::string skel_in_path = FLAGS_in_dir + "/skel_info.txt";
  std::string motion_in_path = FLAGS_in_dir + "/motion_info.txt";
  std::string terrain_in_path = FLAGS_in_dir + "/terrain_info.txt";
  std::string contacts_in_path = FLAGS_in_dir + "/contact_info.txt";

  // skeleton info
  Eigen::MatrixXd left_hip_offset(FLAGS_nframes, 3);
  Eigen::MatrixXd right_hip_offset(FLAGS_nframes, 3);
  double max_leg_length;
  double max_heel_length;
  double heel_dist;
  double body_mass;
  Eigen::MatrixXd I_body(FLAGS_nframes, 6);
  ReadSkeletonInfo(skel_in_path, FLAGS_nframes, left_hip_offset, right_hip_offset, max_leg_length, max_heel_length, heel_dist, body_mass, I_body);

  // motion info
  double dt;
  Eigen::MatrixXd base_lin_init(FLAGS_nframes, 3);
  Eigen::MatrixXd base_ang_init(FLAGS_nframes, 3);
  Eigen::MatrixXd ee0_init(FLAGS_nframes, 3);
  Eigen::MatrixXd ee1_init(FLAGS_nframes, 3);
  Eigen::MatrixXd heel0_init(FLAGS_nframes, 3);
  Eigen::MatrixXd heel1_init(FLAGS_nframes, 3);
  ReadMotionInfo(motion_in_path, FLAGS_nframes, dt, base_lin_init, base_ang_init, ee0_init, ee1_init, heel0_init, heel1_init);

  // terrain info
  Eigen::Vector3d floor_normal;
  Eigen::Vector3d floor_point;
  ReadTerrainInfo(terrain_in_path, floor_normal, floor_point);

  // contacts info
  bool left_toe_start_contact, left_heel_start_contact, right_toe_start_contact, right_heel_start_contact;
  std::vector<double> left_toe_durations, left_heel_durations;
  std::vector<double> right_toe_durations, right_heel_durations;
  ReadContactInfo(contacts_in_path, left_toe_start_contact, left_heel_start_contact, 
                                    right_toe_start_contact, right_heel_start_contact, 
                                    left_toe_durations, left_heel_durations,
                                    right_toe_durations, right_heel_durations);
  // PrintDurations(left_toe_durations, left_heel_durations, right_toe_durations, right_heel_durations);

  double total_time = 0.0;
  for (int i = 0; i < left_toe_durations.size(); i++) {
    total_time += left_toe_durations.at(i);
  }

  //
  // build the problem
  //
  NlpFormulation formulation;
  // terrain
  formulation.terrain_ = std::make_shared<GroundPlane>(floor_normal, floor_point);
  // heels
  int num_ee = 4;
  formulation.params_.heel_dist_ = heel_dist;

  // kinematic & dynamic parameters of human model
  RobotModel robot_model;
  robot_model.dynamic_model_   = std::make_shared<HumanoidDynamicModel>(num_ee, body_mass, I_body, -floor_normal, total_time);
  robot_model.kinematic_model_ = std::make_shared<HumanoidKinematicModel>(num_ee, left_hip_offset, right_hip_offset, max_leg_length, max_heel_length);
  formulation.model_ = robot_model;

  // get motion intialization from given data
  int vel_avg_over_num = 5;
  // directly set initial and final states
  Eigen::Vector3d base_lin_0;
  base_lin_0 << base_lin_init(0,0), base_lin_init(0,1), base_lin_init(0,2);
  auto base_vel_0 = (base_lin_init.middleRows(1, 1) - base_lin_init.middleRows(0, 1)) / dt;
  Eigen::Vector3d base_vel_0_vec;
  base_vel_0_vec << base_vel_0(0, 0), base_vel_0(0, 1), base_vel_0(0, 2);
  for (int frame_idx = 1; frame_idx < vel_avg_over_num; frame_idx++) {
    auto cur_vel = (base_lin_init.middleRows(frame_idx+1, 1) - base_lin_init.middleRows(frame_idx, 1)) / dt;
    Eigen::Vector3d cur_vel_vec;
    cur_vel_vec << cur_vel(0, 0), cur_vel(0, 1), cur_vel(0, 2);
    base_vel_0_vec += cur_vel_vec;
  }
  base_vel_0_vec /= vel_avg_over_num;
  
  formulation.initial_base_.lin.at(kPos) = base_lin_0;
  formulation.initial_base_.lin.at(kVel) = base_vel_0_vec;

  Eigen::Vector3d base_ang_0;
  base_ang_0 << base_ang_init(0,0), base_ang_init(0,1), base_ang_init(0,2);
  auto base_angvel_0 = (base_ang_init.middleRows(1, 1) - base_ang_init.middleRows(0, 1)) / dt;
  Eigen::Vector3d base_angvel_0_vec;
  base_angvel_0_vec << base_angvel_0(0, 0), base_angvel_0(0, 1), base_angvel_0(0, 2);
  formulation.initial_base_.ang.at(kPos) = base_ang_0;
  // formulation.initial_base_.ang.at(kVel) = base_angvel_0_vec;

  Eigen::Vector3d base_lin_f;
  base_lin_f << base_lin_init(FLAGS_nframes-1,0), base_lin_init(FLAGS_nframes-1,1), base_lin_init(FLAGS_nframes-1,2);
  auto base_vel_f = (base_lin_init.middleRows(FLAGS_nframes-1, 1) - base_lin_init.middleRows(FLAGS_nframes-2, 1)) / dt;
  Eigen::Vector3d base_vel_f_vec;
  base_vel_f_vec << base_vel_f(0, 0), base_vel_f(0, 1), base_vel_f(0, 2);
  for (int frame_idx = 1; frame_idx < vel_avg_over_num; frame_idx++) {
    auto cur_vel = (base_lin_init.middleRows(FLAGS_nframes-1-frame_idx, 1) - base_lin_init.middleRows(FLAGS_nframes-2-frame_idx, 1)) / dt;
    Eigen::Vector3d cur_vel_vec;
    cur_vel_vec << cur_vel(0, 0), cur_vel(0, 1), cur_vel(0, 2);
    base_vel_f_vec += cur_vel_vec;
  }
  base_vel_f_vec /= vel_avg_over_num;
  formulation.final_base_.lin.at(kPos) = base_lin_f;
  formulation.final_base_.lin.at(kVel) = base_vel_f_vec;

  Eigen::Vector3d base_ang_f;
  base_ang_f << base_ang_init(FLAGS_nframes-1,0), base_ang_init(FLAGS_nframes-1,1), base_ang_init(FLAGS_nframes-1,2);
  auto base_angvel_f = (base_ang_init.middleRows(FLAGS_nframes-1, 1) - base_ang_init.middleRows(FLAGS_nframes-2, 1)) / dt;
  Eigen::Vector3d base_angvel_f_vec;
  base_angvel_f_vec << base_angvel_f(0, 0), base_angvel_f(0, 1), base_angvel_f(0, 2);
  formulation.final_base_.ang.at(kPos) = base_ang_f;
  // formulation.final_base_.ang.at(kVel) = base_angvel_f_vec;

  Eigen::Vector3d ee0_pos;
  ee0_pos << ee0_init(0,0), ee0_init(0,1), ee0_init(0,2);
  formulation.initial_ee_W_.push_back(ee0_pos);
  Eigen::Vector3d ee1_pos;
  ee1_pos << ee1_init(0,0), ee1_init(0,1), ee1_init(0,2);
  formulation.initial_ee_W_.push_back(ee1_pos);

  Eigen::Vector3d heel0_pos;
  heel0_pos << heel0_init(0,0), heel0_init(0,1), heel0_init(0,2);
  formulation.initial_ee_W_.push_back(heel0_pos);
  Eigen::Vector3d heel1_pos;
  heel1_pos << heel1_init(0,0), heel1_init(0,1), heel1_init(0,2);
  formulation.initial_ee_W_.push_back(heel1_pos);

  formulation.params_.ee_phase_durations_.push_back(left_toe_durations);
  formulation.params_.ee_in_contact_at_start_.push_back(left_toe_start_contact);
  formulation.params_.ee_phase_durations_.push_back(right_toe_durations);
  formulation.params_.ee_in_contact_at_start_.push_back(right_toe_start_contact);  

  formulation.params_.ee_phase_durations_.push_back(left_heel_durations);
  formulation.params_.ee_in_contact_at_start_.push_back(left_heel_start_contact);
  formulation.params_.ee_phase_durations_.push_back(right_heel_durations);
  formulation.params_.ee_in_contact_at_start_.push_back(right_heel_start_contact); 

  // go through durations and calulate how many polynomials we want to use for each non-contact phase
  for (int ee_idx = 0; ee_idx < formulation.params_.ee_phase_durations_.size(); ee_idx++) {
    // std::cout << "ee" << ee_idx << std::endl;
    formulation.params_.ee_polynomials_per_swing_phase_dynamic_.push_back(
              GetPolyChangingPhase(formulation.params_.ee_in_contact_at_start_.at(ee_idx), 
                                    formulation.params_.ee_phase_durations_.at(ee_idx), 
                                    formulation.params_.add_polys_after_dur_, 
                                    formulation.params_.ee_polynomials_per_swing_phase_)
    );
    //  std::cout << "swing" << std::endl;
    //  for (int i = 0; i < formulation.params_.ee_polynomials_per_swing_phase_dynamic_.at(ee_idx).size(); i++) {
      //  std::cout << formulation.params_.ee_polynomials_per_swing_phase_dynamic_.at(ee_idx).at(i) << " ";
    //  }
    //  std::cout << std::endl;
    formulation.params_.force_polynomials_per_stance_phase_dynamic_.push_back(
              GetPolyChangingPhase(!formulation.params_.ee_in_contact_at_start_.at(ee_idx), 
                                    formulation.params_.ee_phase_durations_.at(ee_idx), 
                                    formulation.params_.add_polys_after_dur_, 
                                    formulation.params_.force_polynomials_per_stance_phase_)
    );
    //  std::cout << "force" << std::endl;
    //  for (int i = 0; i < formulation.params_.force_polynomials_per_stance_phase_dynamic_.at(ee_idx).size(); i++) {
    //    std::cout << formulation.params_.force_polynomials_per_stance_phase_dynamic_.at(ee_idx).at(i) << " ";
    //  }
    //  std::cout << std::endl;
  }

  // Initialize the nonlinear-programming problem with the variables,
  // constraints and costs.
  ifopt::Problem nlp;
  SplineHolder solution;
  auto variable_set = formulation.GetVariableSets(solution, true);
  for (auto c : variable_set)
    nlp.AddVariableSet(c);
  for (auto c : formulation.GetConstraints(solution))
    nlp.AddConstraintSet(c);
  for (auto c : formulation.GetCosts())
    nlp.AddCostSet(c);

  //
  // STAGE 1.1: Initialization - fit spline representation as close as possible to input data.
  //          Use only data and velocity smoothing terms.
  //

  // data term
  AddDataCosts(nlp, solution, dt, base_lin_init, base_ang_init,
                      ee0_init, ee1_init, heel0_init, heel1_init,
                      1.0, 1.0, 1.0);
  // Velocity smoothing
  AddVelocitySmoothCosts(nlp, solution, dt, 0.1, 0.1, 0.1);

  // Set up solver
  auto solver = std::make_shared<ifopt::IpoptSolver>();
  solver->SetOption("jacobian_approximation", "exact");
  solver->SetOption("hessian_approximation", "limited-memory");
  solver->SetOption("max_cpu_time", 100000000.0);
  solver->SetOption("max_iter", 7000);
  // solver->SetOption("linear_solver", "mumps");
  solver->SetOption("linear_solver", "MA57");
  solver->SetOption("print_level", 5);
  solver->SetOption("print_frequency_iter", 20);
  solver->SetOption("print_timing_statistics", "yes");
  solver->SetOption("print_user_options", "yes");
  solver->SetOption("tol", 0.001);
  std::cout << "STAGE 1.1: Starting initialization optimization (fitting spline to data)...\n";
  solver->Solve(nlp);
  std::cout << "RETURN STATUS: " << solver->GetReturnStatus() << std::endl;

  nlp.PrintCurrent(); // view variable-set, constraint violations, indices,...

  using namespace std;
  // cout.precision(2);

  //
  // STAGE 1.2: Now add leg and foot kinematics and fine-tune. 
  //
  formulation.params_.ClearConstraints();
  formulation.params_.AddLegConstraints();
  formulation.params_.AddHeelConstraints();
  for (auto c : formulation.GetConstraints(solution))
    nlp.AddConstraintSet(c);

  std::cout << "STAGE 1.2: Adding in kinematic constraints...\n";
  solver->Solve(nlp);
  nlp.PrintCurrent();

  std::cout << "Saving solution before optimizing dynamics..." << std::endl;
  std::string out_file_no_dynamics = FLAGS_out_dir + "/sol_out_no_dynamics.txt";
  SaveSolution(solution, out_file_no_dynamics, dt, formulation.params_);


  //
  // STAGE 2.1: Add in physical dynamics constraints
  //
  ifopt::Problem dynamics_nlp;
  formulation.params_.ClearConstraints();
  formulation.params_.AddInitConstraints();
  formulation.params_.AddLegConstraints();
  formulation.params_.AddDynamicsConstraints();
  formulation.params_.AddHeelConstraints();

  int var_idx = 0;
  for (auto c : variable_set) {
    dynamics_nlp.AddVariableSet(c);
    var_idx++;
  }
  for (auto c : formulation.GetConstraints(solution))
    dynamics_nlp.AddConstraintSet(c);
  for (auto c : formulation.GetCosts())
    dynamics_nlp.AddCostSet(c);


  double cost_weights[] = {FLAGS_w_com_lin, FLAGS_w_com_ang, FLAGS_w_ee, FLAGS_w_smooth, 1.0};
  double duration_cost_weight_dynamics = FLAGS_w_dur;

  // data term
  AddDataCosts(dynamics_nlp, solution, dt, base_lin_init, base_ang_init,
                      ee0_init, ee1_init, heel0_init, heel1_init,
                      cost_weights[0], cost_weights[1], cost_weights[2]);
  // Velocity smoothing
  AddVelocitySmoothCosts(dynamics_nlp, solution, dt, 0.001, 0.001, cost_weights[3]);
  // Acceleration smoothing
  AddAccelSmoothCosts(dynamics_nlp, solution, dt, 0.0001, 0.0001, 0.0001);

  // solve
  solver->SetOption("max_iter", 7000);
  std::cout << "STAGE 2.1: Adding in dynamics constraints...\n";
  solver->Solve(dynamics_nlp);
  dynamics_nlp.PrintCurrent();

  //
  // Stage 2.2: Make sure floor height constraints are met through fine-tuning.
  //
  formulation.params_.ClearConstraints();
  formulation.params_.AddHeightConstraints();
  for (auto c : formulation.GetConstraints(solution))
    dynamics_nlp.AddConstraintSet(c);
  solver->SetOption("max_iter", 2500);
  std::cout << "STAGE 2.2: Adding in floor height constraint...\n";
  solver->Solve(dynamics_nlp);
  bool dynamics_succeed = (solver->GetReturnStatus() == 0);
  dynamics_nlp.PrintCurrent();


  std::cout << "Saving solution before optimizing durations..." << std::endl;
  std::string out_file_dynamics = FLAGS_out_dir + "/sol_out_dynamics.txt";
  SaveSolution(solution, out_file_dynamics, dt, formulation.params_);

  //
  // STAGE 3: Fine-tune foot contact durations.
  //
  ifopt::Problem durations_nlp;
  formulation.params_.ClearConstraints();
  formulation.params_.AddInitConstraints();
  formulation.params_.AddLegConstraints();
  formulation.params_.AddDynamicsConstraints();
  formulation.params_.AddHeightConstraints();
  formulation.params_.AddHeelConstraints();

  var_idx = 0;
  for (auto c : variable_set) {
    durations_nlp.AddVariableSet(c);
    var_idx++;
  }
  formulation.params_.OptimizePhaseDurations();
  for (auto c : solution.phase_durations_)
    durations_nlp.AddVariableSet(c);
  for (auto c : formulation.GetConstraints(solution))
    durations_nlp.AddConstraintSet(c);
  for (auto c : formulation.GetCosts())
    durations_nlp.AddCostSet(c);

  // data term
  AddDataCosts(durations_nlp, solution, dt, base_lin_init, base_ang_init,
                      ee0_init, ee1_init, heel0_init, heel1_init,
                      cost_weights[0], cost_weights[1], cost_weights[2]);
  // Velocity smoothing
  AddVelocitySmoothCosts(durations_nlp, solution, dt, 0.001, 0.001, cost_weights[3]);
  // NOTE: using acceleration smoothing while optimizing contact durations is currently not supported.

  // durations costs (keep close to initialization)
  auto left_duration_cost_dynamics = std::make_shared<DurationCost>(0, left_toe_durations, duration_cost_weight_dynamics);
  durations_nlp.AddCostSet(left_duration_cost_dynamics);
  auto right_duration_cost_dynamics = std::make_shared<DurationCost>(1, right_toe_durations, duration_cost_weight_dynamics);
  durations_nlp.AddCostSet(right_duration_cost_dynamics);
  auto left_heel_duration_cost_dynamics = std::make_shared<DurationCost>(2, left_heel_durations, duration_cost_weight_dynamics);
  durations_nlp.AddCostSet(left_heel_duration_cost_dynamics);
  auto right_heel_duration_cost_dynamics = std::make_shared<DurationCost>(3, right_heel_durations, duration_cost_weight_dynamics);
  durations_nlp.AddCostSet(right_heel_duration_cost_dynamics);

  // solve
  solver->SetOption("max_iter", 2000);
  std::cout << "STAGE 3: Adding in contact durations...\n";
  solver->Solve(durations_nlp);
  bool durations_succeed = (solver->GetReturnStatus() == 0);
  std::cout << "Return status: " << solver->GetReturnStatus() << std::endl;
  durations_nlp.PrintCurrent();

  // if durations fails to meet constraints, fix the new contact durations and optimize just dynamics again.
  if (!durations_succeed) {
    ifopt::Problem post_durations_nlp;
    formulation.params_.ClearConstraints();
    formulation.params_.AddInitConstraints();
    formulation.params_.AddLegConstraints();
    formulation.params_.AddDynamicsConstraints();
    formulation.params_.AddHeightConstraints();
    formulation.params_.AddHeelConstraints();

    var_idx = 0;
    for (auto c : variable_set) {
      post_durations_nlp.AddVariableSet(c);
      var_idx++;
    }
    for (auto c : formulation.GetConstraints(solution))
      post_durations_nlp.AddConstraintSet(c);
    for (auto c : formulation.GetCosts())
      post_durations_nlp.AddCostSet(c);

    // data term
    AddDataCosts(post_durations_nlp, solution, dt, base_lin_init, base_ang_init,
                        ee0_init, ee1_init, heel0_init, heel1_init,
                        cost_weights[0], cost_weights[1], cost_weights[2]);
    // Velocity smoothing
    AddVelocitySmoothCosts(post_durations_nlp, solution, dt, 0.001, 0.001, cost_weights[3]);
    // Acceleration smoothing
    AddAccelSmoothCosts(post_durations_nlp, solution, dt, 0.0001, 0.0001, 0.0001);

    // solve
    solver->SetOption("max_iter", 7000);
    std::cout << "STAGE 4: Durations failed, returning to dynamics optim with current duration variables...\n";
    solver->Solve(post_durations_nlp);
    durations_succeed = (solver->GetReturnStatus() == 0);
    std::cout << "Return status: " << solver->GetReturnStatus() << std::endl;
    post_durations_nlp.PrintCurrent();
  } 

  // std::cout << "TOE DURATIONS: " << std::endl;
  // PrintDurationResults(solution, left_toe_durations, right_toe_durations);
  // std::cout << "HEEL DURATIONS: " << std::endl;
  // PrintDurationResults(solution, left_heel_durations, right_heel_durations);

  std::cout << "Saving final solution..." << std::endl;
  std::string out_file_durations = FLAGS_out_dir + "/sol_out_durations.txt";
  SaveSolution(solution, out_file_durations, dt, formulation.params_);

  std::string success_file_out = FLAGS_out_dir + "/success_log.txt";
  SaveSuccessLog(success_file_out, dynamics_succeed, durations_succeed);
}
