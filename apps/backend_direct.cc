
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include "so3.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "imu_error.h"
#include "pre_int_imu_error.h"
#include "reprojection_error.h"



// TODO: initialized from config files
Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      
Eigen::Vector3d gyro_bias = Eigen::Vector3d(-0.003196, 0.021298, 0.078430);
Eigen::Vector3d accel_bias = Eigen::Vector3d(-0.026176, 0.137568, 0.076295);


double ConverStrTime(std::string time_str) {
  std::string time_str_sec = time_str.substr(7,3);       // second
  std::string time_str_nano_sec = time_str.substr(10);   // nano-second

  return std::stoi(time_str_sec) + std::stoi(time_str_nano_sec)*1e-9;
}

struct IMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUData(std::string imu_data_str) {
    std::stringstream str_stream(imu_data_str);          // Create a stringstream of the current line

    if (str_stream.good()) {
        
      std::string data_str;
      std::getline(str_stream, data_str, ',');           // get first string delimited by comma
      timestamp_ = ConverStrTime(data_str);

      for (int i=0; i<3; ++i) {                          // gyrometer measurement
        std::getline(str_stream, data_str, ','); 
        gyr_(i) = std::stod(data_str);
      }

      for (int i=0; i<3; ++i) {                    
        std::getline(str_stream, data_str, ',');         // accelerometer measurement 
        acc_(i) = std::stod(data_str);
      }
    }
  }

  double timestamp_;
  Eigen::Vector3d gyr_;
  Eigen::Vector3d acc_; 
};


struct ObservationData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObservationData(std::string observation_data_str) {
    std::stringstream str_stream(observation_data_str);          // Create a stringstream of the current line

    if (str_stream.good()) {
        
      std::string data_str;
      std::getline(str_stream, data_str, ',');   // get first string delimited by comma
      timestamp_ = ConverStrTime(data_str);

      std::getline(str_stream, data_str, ','); 
      index_ = std::stoi(data_str);

      for (int i=0; i<2; ++i) {                    
        std::getline(str_stream, data_str, ','); 
        feature_pos_(i) = std::stod(data_str);
      }

      std::getline(str_stream, data_str, ','); 
      size_ = std::stod(data_str);
    }
  }

  double timestamp_;
  size_t index_;
  Eigen::Vector2d feature_pos_; 
  double size_;
};

class State {

 public:
  State(double timestamp) {
    timestamp_ = timestamp;

    rotation_block_ptr_ = new QuatParameterBlock();
    velocity_block_ptr_ = new Vec3dParameterBlock();
    position_block_ptr_ = new Vec3dParameterBlock();
  }

  ~State() {
    delete [] rotation_block_ptr_;
    delete [] velocity_block_ptr_;
    delete [] position_block_ptr_;
  }

  double GetTimestamp() {
    return timestamp_;
  }

  QuatParameterBlock* GetRotationBlock() {
    return rotation_block_ptr_;
  }

  Vec3dParameterBlock* GetVelocityBlock() {
    return velocity_block_ptr_;
  }

  Vec3dParameterBlock* GetPositionBlock() {
    return position_block_ptr_;
  }

 private:
  double timestamp_;
  QuatParameterBlock* rotation_block_ptr_;
  Vec3dParameterBlock* velocity_block_ptr_;
  Vec3dParameterBlock* position_block_ptr_;
};






class ExpLandmarkOptSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkOptSLAM(std::string config_folder_path) {
    ReadConfigurationFiles(config_folder_path);

    quat_parameterization_ptr_ = new ceres::QuaternionParameterization();
  }

  bool ReadConfigurationFiles(std::string config_folder_path) {

    // test configuration file
    cv::FileStorage test_config_file(config_folder_path + "test.yaml", cv::FileStorage::READ);
    time_begin_ = ConverStrTime(test_config_file["time_window"][0]);  
    time_end_ = ConverStrTime(test_config_file["time_window"][1]);  

    tri_max_num_iterations_ = (int)(test_config_file["backend"]["tri_max_num_iterations"]);

    // experiment configuration file
    cv::FileStorage experiment_config_file(config_folder_path + "config_fpga_p2_euroc.yaml", cv::FileStorage::READ);

    cv::FileNode T_BC_node = experiment_config_file["cameras"][0]["T_SC"];            // from camera frame to body frame

    // Eigen::Matrix4d T_BC;
    T_bc_  <<  T_BC_node[0],  T_BC_node[1],  T_BC_node[2],  T_BC_node[3], 
               T_BC_node[4],  T_BC_node[5],  T_BC_node[6],  T_BC_node[7], 
               T_BC_node[8],  T_BC_node[9], T_BC_node[10], T_BC_node[11], 
              T_BC_node[12], T_BC_node[13], T_BC_node[14], T_BC_node[15];

    //T_bc_ = T_BC;

    fu_ = experiment_config_file["cameras"][0]["focal_length"][0];
    fv_ = experiment_config_file["cameras"][0]["focal_length"][1];

    cu_ = experiment_config_file["cameras"][0]["principal_point"][0];
    cv_ = experiment_config_file["cameras"][0]["principal_point"][1];

    sigma_g_c_ = experiment_config_file["imu_params"]["sigma_g_c"];
    sigma_a_c_ = experiment_config_file["imu_params"]["sigma_a_c"];

    
    return true;
  }

  bool ReadInitialCondition(std::string ground_truth_file_path) {

    std::cout << "Read ground truth data at " << ground_truth_file_path << std::endl;

    std::ifstream input_file(ground_truth_file_path);
    
    if(!input_file.is_open()) 
      throw std::runtime_error("Could not open file");

    // Read the column names
    // Extract the first line in the file
    std::string line;
    std::getline(input_file, line);

    while (std::getline(input_file, line)) {
      std::stringstream s_stream(line);                // Create a stringstream of the current line

      if (s_stream.good()) {
        std::string time_stamp_str;
        std::getline(s_stream, time_stamp_str, ',');   // get first string delimited by comma
      
        if (time_begin_ <= ConverStrTime(time_stamp_str)) {

          // state
          state_parameter_.push_back(new State(ConverStrTime(time_stamp_str)));


          // position
          std::string initial_position_str[3];
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, initial_position_str[i], ','); 
          }

          Eigen::Vector3d initial_position(std::stod(initial_position_str[0]), std::stod(initial_position_str[1]), std::stod(initial_position_str[2]));
          state_parameter_.at(0)->GetPositionBlock()->setEstimate(initial_position);
          optimization_problem_.AddParameterBlock(state_parameter_.at(0)->GetPositionBlock()->parameters(), 3);

          // rotation
          std::string initial_rotation_str[4];
          for (int i=0; i<4; ++i) {                    
            std::getline(s_stream, initial_rotation_str[i], ','); 
          }

          Eigen::Quaterniond initial_rotation(std::stod(initial_rotation_str[0]), std::stod(initial_rotation_str[1]), std::stod(initial_rotation_str[2]), std::stod(initial_rotation_str[3]));
          state_parameter_.at(0)->GetRotationBlock()->setEstimate(initial_rotation);
          optimization_problem_.AddParameterBlock(state_parameter_.at(0)->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);

          // velocity
          std::string initial_velocity_str[3];
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, initial_velocity_str[i], ','); 
          }

          Eigen::Vector3d initial_velocity(std::stod(initial_velocity_str[0]), std::stod(initial_velocity_str[1]), std::stod(initial_velocity_str[2]));
          state_parameter_.at(0)->GetVelocityBlock()->setEstimate(initial_velocity);
          optimization_problem_.AddParameterBlock(state_parameter_.at(0)->GetVelocityBlock()->parameters(), 3);

          // set initial condition
          optimization_problem_.SetParameterBlockConstant(state_parameter_.at(0)->GetRotationBlock()->parameters());
          optimization_problem_.SetParameterBlockConstant(state_parameter_.at(0)->GetVelocityBlock()->parameters());
          optimization_problem_.SetParameterBlockConstant(state_parameter_.at(0)->GetPositionBlock()->parameters());


          std::cout << "Finished initialization from the ground truth file." << std::endl;

          input_file.close();
          return true;
        }
      }
    }

    input_file.close();
    std::cout << "Initialization fails!" << std::endl;
    return false;
  }  


  bool ProcessGroundTruth(std::string ground_truth_file_path) {

    std::cout << "Read ground truth data at " << ground_truth_file_path << std::endl;

    std::ifstream input_file(ground_truth_file_path);
    if(!input_file.is_open()) 
      throw std::runtime_error("Could not open file");

    // Read the column names
    // Extract the first line in the file
    std::string line;
    std::getline(input_file, line);

    std::ofstream output_file("ground_truth.csv");
    output_file << "timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,b_w_x,b_w_y,b_w_z,b_a_x,b_a_y,b_a_z\n";

    while (std::getline(input_file, line)) {
      std::stringstream s_stream(line);                // Create a stringstream of the current line

      if (s_stream.good()) {
        std::string time_stamp_str;
        std::getline(s_stream, time_stamp_str, ',');   // get first string delimited by comma
      
        if (time_begin_ <= ConverStrTime(time_stamp_str) && ConverStrTime(time_stamp_str) <= time_end_) {

          std::string data;
          output_file << std::to_string(ConverStrTime(time_stamp_str));
          for (int i=0; i<7; ++i) {                    
            std::getline(s_stream, data, ',');

            output_file << ",";
            output_file << data;
          }

          // velocity
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, data, ',');

            output_file << ",";
            output_file << data;
          }

          for (int i=0; i<6; ++i) {                    
            std::getline(s_stream, data, ',');

            output_file << ",";
            output_file << data;
          }

          output_file << std::endl;
        }
      }
    }

    input_file.close();
    output_file.close();

    return true;
  }  

  bool ReadIMUData(std::string imu_file_path) {
  
    std::cout << "Read IMU data at " << imu_file_path << std::endl;

    std::ifstream input_file(imu_file_path);
    
    if(!input_file.is_open()) 
      throw std::runtime_error("Could not open file");

    // Read the column names
    // Extract the first line in the file
    std::string first_line_data_str;
    std::getline(input_file, first_line_data_str);

    // storage of IMU data
    std::vector<IMUData> imu_data_vec;

    std::string imu_data_str;
    while (std::getline(input_file, imu_data_str)) {

      IMUData imu_data(imu_data_str);

      if (time_begin_ <= imu_data.timestamp_ && imu_data.timestamp_ <= time_end_) {

        imu_data_vec.push_back(imu_data);

        if (imu_data_vec.size()>1) {

          state_parameter_.push_back(new State(imu_data.timestamp_));

          optimization_problem_.AddParameterBlock(state_parameter_.back()->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
          optimization_problem_.AddParameterBlock(state_parameter_.back()->GetVelocityBlock()->parameters(), 3);
          optimization_problem_.AddParameterBlock(state_parameter_.back()->GetPositionBlock()->parameters(), 3);
        }
      }
    }

    // dead-reckoning to initialize 
    for (size_t i=0; i<imu_data_vec.size()-1; ++i) {
        
      double time_diff = state_parameter_.at(i+1)->GetTimestamp() - state_parameter_.at(i)->GetTimestamp();

      Eigen::Quaterniond rotation_t = state_parameter_.at(i)->GetRotationBlock()->estimate().normalized();
      Eigen::Vector3d velocity_t = state_parameter_.at(i)->GetVelocityBlock()->estimate();
      Eigen::Vector3d position_t = state_parameter_.at(i)->GetPositionBlock()->estimate();

      Eigen::Vector3d accel_measurement = imu_data_vec.at(i).acc_;
      Eigen::Vector3d gyro_measurement = imu_data_vec.at(i).gyr_;      
      Eigen::Vector3d accel_plus_gravity = rotation_t.toRotationMatrix()*(accel_measurement - accel_bias) + gravity;
      

      Eigen::Quaterniond rotation_t1 = rotation_t * Exp_q(time_diff*(gyro_measurement-gyro_bias));
      Eigen::Vector3d velocity_t1 = velocity_t + time_diff*accel_plus_gravity;
      Eigen::Vector3d position_t1 = position_t + time_diff*velocity_t + (0.5*time_diff*time_diff)*accel_plus_gravity;




      state_parameter_.at(i+1)->GetRotationBlock()->setEstimate(rotation_t1);
      state_parameter_.at(i+1)->GetVelocityBlock()->setEstimate(velocity_t1); 
      state_parameter_.at(i+1)->GetPositionBlock()->setEstimate(position_t1); 

      
      // add constraints
      ceres::CostFunction* cost_function = new ImuError(imu_data_vec.at(i).gyr_,
                                                        imu_data_vec.at(i).acc_,
                                                        time_diff,
                                                        Eigen::Vector3d(-0.003196, 0.021298, 0.078430),  // gyr bias
                                                        Eigen::Vector3d(-0.026176, 0.137568, 0.076295),  // acc bias
                                                        sigma_g_c_,
                                                        sigma_a_c_);


      optimization_problem_.AddResidualBlock(cost_function,
                                                 NULL,
                                                 state_parameter_.at(i+1)->GetRotationBlock()->parameters(),
                                                 state_parameter_.at(i+1)->GetVelocityBlock()->parameters(),
                                                 state_parameter_.at(i+1)->GetPositionBlock()->parameters(),
                                                 state_parameter_.at(i)->GetRotationBlock()->parameters(),
                                                 state_parameter_.at(i)->GetVelocityBlock()->parameters(),
                                                 state_parameter_.at(i)->GetPositionBlock()->parameters());   

      /***
      optimization_problem_.SetParameterLowerBound(state_parameter_.at(i+1)->GetPositionBlock()->parameters(), 0, -2.8);
      optimization_problem_.SetParameterLowerBound(state_parameter_.at(i+1)->GetPositionBlock()->parameters(), 1,  4.2);
      optimization_problem_.SetParameterLowerBound(state_parameter_.at(i+1)->GetPositionBlock()->parameters(), 2, -1.8);

      optimization_problem_.SetParameterUpperBound(state_parameter_.at(i+1)->GetPositionBlock()->parameters(), 0, 1.8);
      optimization_problem_.SetParameterUpperBound(state_parameter_.at(i+1)->GetPositionBlock()->parameters(), 1, 8.8);
      optimization_problem_.SetParameterUpperBound(state_parameter_.at(i+1)->GetPositionBlock()->parameters(), 2, 2.8);
      ***/
    }

    input_file.close();
    std::cout << "Finished reading IMU data." << std::endl;
    return true;
  }


  bool ReadObservationData(std::string observation_file_path) {
  
    std::cout << "Read observation data at " << observation_file_path << std::endl;

    std::ifstream input_file(observation_file_path);
    
    if(!input_file.is_open())
      throw std::runtime_error("Could not open file");

    std::string first_line_data_str;
    std::getline(input_file, first_line_data_str);

    std::string observation_data_str;
    while (std::getline(input_file, observation_data_str)) {
      ObservationData observation_data(observation_data_str);

      // add observation constraints
      size_t pose_id;
      size_t landmark_id = observation_data.index_-1;

      for (size_t i=0; i<state_parameter_.size(); ++i) {
        if (observation_data.timestamp_ <= state_parameter_.at(i)->GetTimestamp()) {
          pose_id = i;
          break;
        }
      }

      if (landmark_id >= landmark_parameter_.size()) {
        landmark_parameter_.resize(landmark_id+1);
      }

      // landmark_parameter_.at(landmark_id) = new Vec3dParameterBlock(Eigen::Vector3d(0, 0, 0));
      landmark_parameter_.at(landmark_id) = new Vec3dParameterBlock(Eigen::Vector3d()+0.5*Eigen::Vector3d::Random());

      ceres::CostFunction* cost_function = new ReprojectionError(observation_data.feature_pos_,
                                                                 T_bc_,
                                                                 fu_, fv_,
                                                                 cu_, cv_,
                                                                 (observation_data.size_*observation_data.size_/64.0)*Eigen::Matrix2d::Identity());

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             state_parameter_.at(pose_id)->GetRotationBlock()->parameters(),
                                             state_parameter_.at(pose_id)->GetPositionBlock()->parameters(),
                                             landmark_parameter_.at(landmark_id)->parameters()); 
    }

    for (size_t i=0; i<landmark_parameter_.size(); ++i) {

      optimization_problem_.SetParameterLowerBound(landmark_parameter_.at(i)->parameters(), 0, -30);
      optimization_problem_.SetParameterLowerBound(landmark_parameter_.at(i)->parameters(), 1, -40);
      optimization_problem_.SetParameterLowerBound(landmark_parameter_.at(i)->parameters(), 2, -20);

      optimization_problem_.SetParameterUpperBound(landmark_parameter_.at(i)->parameters(), 0, 30);
      optimization_problem_.SetParameterUpperBound(landmark_parameter_.at(i)->parameters(), 1, 20);
      optimization_problem_.SetParameterUpperBound(landmark_parameter_.at(i)->parameters(), 2, 20);
    }

    input_file.close();
    std::cout << "Finished reading observation data." << std::endl;
    return true;
  }


  bool SolveOptimizationProblem() {

    std::cout << "Begin solving the optimization problem." << std::endl;

    // Step 1: triangularization (set trajectory constant)
    optimization_options_.linear_solver_type = ceres::DENSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 6;
    optimization_options_.max_num_iterations = tri_max_num_iterations_;
    optimization_options_.function_tolerance = 1e-9;

    for (size_t i=1; i<state_parameter_.size(); ++i) {
      optimization_problem_.SetParameterBlockConstant(state_parameter_.at(i)->GetRotationBlock()->parameters());
      optimization_problem_.SetParameterBlockConstant(state_parameter_.at(i)->GetVelocityBlock()->parameters());
      optimization_problem_.SetParameterBlockConstant(state_parameter_.at(i)->GetPositionBlock()->parameters());
    }


    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";


    // Step 2: optimize trajectory
    optimization_options_.max_num_iterations = 30;

    for (size_t i=1; i<state_parameter_.size(); ++i) {
      optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetRotationBlock()->parameters());
      optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetVelocityBlock()->parameters());
      optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetPositionBlock()->parameters());
    }

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }

  bool OutputOptimizationResult(std::string output_file_name) {

    std::ofstream output_file(output_file_name);

    output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=0; i<state_parameter_.size(); ++i) {
      output_file << std::to_string(state_parameter_.at(i)->GetTimestamp()) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetPositionBlock()->estimate()(0)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetPositionBlock()->estimate()(1)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetPositionBlock()->estimate()(2)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetVelocityBlock()->estimate()(0)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetVelocityBlock()->estimate()(1)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetVelocityBlock()->estimate()(2)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetRotationBlock()->estimate().w()) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetRotationBlock()->estimate().x()) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetRotationBlock()->estimate().y()) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetRotationBlock()->estimate().z()) << std::endl;
    }

    output_file.close();

    std::ofstream output_file_landmark("landmark.csv");

    output_file_landmark << "id,p_x,p_y,p_z\n";

    for (size_t i=0; i<landmark_parameter_.size(); ++i) {
      output_file_landmark << std::to_string(i+1) << ",";
      output_file_landmark << std::to_string(landmark_parameter_.at(i)->estimate()(0)) << ",";
      output_file_landmark << std::to_string(landmark_parameter_.at(i)->estimate()(1)) << ",";
      output_file_landmark << std::to_string(landmark_parameter_.at(i)->estimate()(2)) << std::endl;
    }

    output_file_landmark.close();

    return true;
  }


 private:
  double time_begin_;
  double time_end_;
  int tri_max_num_iterations_;

  // Eigen::Transform<double, 3, Eigen::Affine> T_bc_;
  Eigen::Matrix4d T_bc_;
  double fu_;   // focal length u
  double fv_;
  double cu_;   // image center u
  double cv_;   // image center v

  // parameter containers
  std::vector<State*>                state_parameter_;
  std::vector<Vec3dParameterBlock*>  landmark_parameter_;

  double accel_bias_parameter_[3];
  double gyro_bias_parameter_[3];

  double sigma_g_c_;   // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;   //  accelerometer noise density [m/s^2/sqrt(Hz)]

  // ceres parameter
  ceres::LocalParameterization* quat_parameterization_ptr_;
  ceres::Problem optimization_problem_;
  ceres::Solver::Options optimization_options_;
  ceres::Solver::Summary optimization_summary_;
};


int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  std::string config_folder_path("../config/");
  ExpLandmarkOptSLAM slam_problem(config_folder_path);

  std::string euroc_dataset_path = "../../../dataset/mav0/";
  std::string ground_truth_file_path = euroc_dataset_path + "state_groundtruth_estimate0/data.csv";
  slam_problem.ReadInitialCondition(ground_truth_file_path);
  slam_problem.ProcessGroundTruth(ground_truth_file_path);

  std::string imu_file_path = euroc_dataset_path + "imu0/data.csv";
  slam_problem.ReadIMUData(imu_file_path);
  slam_problem.OutputOptimizationResult("trajectory_dr.csv");

  std::string observation_file_path = "feature_observation.csv";
  slam_problem.ReadObservationData(observation_file_path);

  slam_problem.SolveOptimizationProblem();
  slam_problem.OutputOptimizationResult("trajectory.csv");

  return 0;
}