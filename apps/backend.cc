
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

#include "landmark_parameter_block.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "imu_error.h"
#include "pre_int_imu_error.h"
#include "reprojection_error.h"



// TODO: initialized from config files
Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      
Eigen::Vector3d gyro_bias = Eigen::Vector3d(-0.003196, 0.021298, 0.078430);
Eigen::Vector3d accel_bias = Eigen::Vector3d(-0.026176, 0.137568, 0.076295);

// TODO: avoid data conversion
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
        gyro_(i) = std::stod(data_str);
      }

      for (int i=0; i<3; ++i) {                    
        std::getline(str_stream, data_str, ',');         // accelerometer measurement 
        accel_(i) = std::stod(data_str);
      }
    }
  }

  double timestamp_;
  Eigen::Vector3d gyro_;
  Eigen::Vector3d accel_; 
};

struct PreIntIMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreIntIMUData(double dt, 
                Eigen::Matrix3d d_rotation, 
                Eigen::Vector3d d_velocity, 
                Eigen::Vector3d d_position) {
    dt_ = dt;
    d_rotation_ = d_rotation;
    d_velocity_ = d_velocity;
    d_position_ = d_position;
  }

  double dt_;
  Eigen::Matrix3d d_rotation_;  
  Eigen::Vector3d d_velocity_;
  Eigen::Vector3d d_position_; 
};

class ObservationData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
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
    }
  }

  double GetTimestamp() {
    return timestamp_;
  }

  double GetId() {
    return index_;
  }

  Eigen::Vector2d GetFeaturePosition() {
    return feature_pos_;
  }

 private:
  double timestamp_;
  size_t index_;
  Eigen::Vector2d feature_pos_; 
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

  void SetRotationBlock(QuatParameterBlock* rotation_block) {
    rotation_block_ptr_ = rotation_block;
  }

  void SetVelocityBlock(Vec3dParameterBlock* velocity_block) {
    velocity_block_ptr_ = velocity_block;
  }

  void SetPositionBlock(Vec3dParameterBlock* position_block) {
    position_block_ptr_ = position_block;
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


Eigen::Matrix3d Hat(Eigen::Vector3d vec) {
  Eigen::Matrix3d hatted_matrix;
  hatted_matrix <<      0 , -vec(2),  vec(1),
                    vec(2),      0 , -vec(0),
                   -vec(1),  vec(0),      0;

  return hatted_matrix;
}

Eigen::Matrix3d Exp(Eigen::Vector3d omega) {
  Eigen::Matrix3d hatted_omega = Hat(omega);
  double bar_omega = acos(0.5*(hatted_omega.trace()-1));

  return Eigen::Matrix3d::Identity() + (sin(bar_omega)/bar_omega) * hatted_omega + ((1-cos(bar_omega))/(bar_omega*bar_omega)) * hatted_omega*hatted_omega;
}


class ExpLandmarkOptSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkOptSLAM(std::string config_folder_path) {
    ReadConfigurationFiles(config_folder_path);
  }

  bool ReadConfigurationFiles(std::string config_folder_path) {

    // test configuration file
    cv::FileStorage test_config_file(config_folder_path + "test.yaml", cv::FileStorage::READ);
    time_begin_ = ConverStrTime(test_config_file["time_window"][0]);  
    time_end_ = ConverStrTime(test_config_file["time_window"][1]);  

    tri_max_num_iterations_ = (int)(test_config_file["backend"]["tri_max_num_iterations"]);

    // experiment configuration file
    cv::FileStorage experiment_config_file(config_folder_path + "config_fpga_p2_euroc.yaml", cv::FileStorage::READ);

    imu_dt_ = 1.0 / (double) experiment_config_file["imu_params"]["imu_rate"]; 

    cv::FileNode T_BC_node = experiment_config_file["cameras"][0]["T_SC"];            // from camera frame to body frame

    Eigen::Matrix4d T_BC;
    T_BC  <<  T_BC_node[0],  T_BC_node[1],  T_BC_node[2],  T_BC_node[3], 
              T_BC_node[4],  T_BC_node[5],  T_BC_node[6],  T_BC_node[7], 
              T_BC_node[8],  T_BC_node[9], T_BC_node[10], T_BC_node[11], 
             T_BC_node[12], T_BC_node[13], T_BC_node[14], T_BC_node[15];

    T_bc_ = T_BC;

    double focal_length_0 = experiment_config_file["cameras"][0]["focal_length"][0];  // i don't know the unit!!!!
    double focal_length_1 = experiment_config_file["cameras"][0]["focal_length"][1];
    focal_length_ = 0.5*focal_length_0 + 0.5*focal_length_1;

    principal_point_[0] = experiment_config_file["cameras"][0]["principal_point"][0];
    principal_point_[1] = experiment_config_file["cameras"][0]["principal_point"][1];
    
    return true;
  }

  bool ReadInitialCondition(std::string ground_truth_file_path) {

    std::cout << "Read ground truth data at " << ground_truth_file_path << std::endl;

    std::ifstream input_file(ground_truth_file_path);

    assert(("Could not open ground truth file.", input_file.is_open()));

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
          Vec3dParameterBlock* position_parameter = new Vec3dParameterBlock(initial_position);
          state_parameter_.at(0)->SetPositionBlock(position_parameter);
          optimization_problem_.AddParameterBlock(state_parameter_.at(0)->GetPositionBlock()->parameters(), 3);

          // rotation
          std::string initial_rotation_str[4];
          for (int i=0; i<4; ++i) {                    
            std::getline(s_stream, initial_rotation_str[i], ','); 
          }

          Eigen::Quaterniond initial_rotation(std::stod(initial_rotation_str[0]), std::stod(initial_rotation_str[1]), std::stod(initial_rotation_str[2]), std::stod(initial_rotation_str[3]));
          QuatParameterBlock* rotation_parameter = new QuatParameterBlock(initial_rotation);
          state_parameter_.at(0)->SetRotationBlock(rotation_parameter);
          optimization_problem_.AddParameterBlock(state_parameter_.at(0)->GetRotationBlock()->parameters(), 4, new ceres::QuaternionParameterization());

          // velocity
          std::string initial_velocity_str[3];
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, initial_velocity_str[i], ','); 
          }

          Eigen::Vector3d initial_velocity(std::stod(initial_velocity_str[0]), std::stod(initial_velocity_str[1]), std::stod(initial_velocity_str[2]));
          Vec3dParameterBlock* velocity_parameter = new Vec3dParameterBlock(initial_velocity);
          state_parameter_.at(0)->SetVelocityBlock(velocity_parameter);
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


  bool ReadObservationData(std::string observation_file_path) {
  
    assert(("state_parameter_ should have been initialized.", !state_parameter_.empty()));

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
      // size_t pose_id;
      size_t landmark_id = observation_data.GetId()-1;

      if (state_parameter_.back()->GetTimestamp() < observation_data.GetTimestamp()) {
          state_parameter_.push_back(new State(observation_data.GetTimestamp()));


          optimization_problem_.AddParameterBlock(state_parameter_.back()->GetRotationBlock()->parameters(), 4, new ceres::QuaternionParameterization());

      }
      else if (state_parameter_.back()->GetTimestamp() == observation_data.GetTimestamp()) {
      }
      else if (state_parameter_.back()->GetTimestamp() > observation_data.GetTimestamp()) {
          std::cout << "error!";
      }

      if (landmark_id >= landmark_parameter_.size()) {
        landmark_parameter_.resize(landmark_id+1);
      }


      landmark_parameter_.at(landmark_id) = new LandmarkParameterBlock(Eigen::Vector3d(0, 0, 0));
      // landmark_parameter_.at(landmark_id) = new LandmarkParameterBlock(Eigen::Vector3d()+0.5*Eigen::Vector3d::Random());      

      ceres::CostFunction* cost_function = new ReprojectionError(observation_data.GetFeaturePosition(),
                                                                 T_bc_,
                                                                 focal_length_,
                                                                 principal_point_);
      

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             state_parameter_.back()->GetRotationBlock()->parameters(),
                                             state_parameter_.back()->GetPositionBlock()->parameters(),
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

    /***
    for (size_t i=0; i<state_parameter_.size(); ++i) {
      std::cout << state_parameter_.at(i)->GetTimestamp() << std::endl;
    }
    ***/


    input_file.close();
    std::cout << "Finished reading observation data." << std::endl;
    return true;
  }

  bool ProcessGroundTruth(std::string ground_truth_file_path) {

    std::cout << "Read ground truth data at " << ground_truth_file_path << std::endl;

    std::ifstream input_file(ground_truth_file_path);
    
    assert(("Could not open ground truth file.", input_file.is_open()));

    // Read the column names
    // Extract the first line in the file
    std::string line;
    std::getline(input_file, line);

    std::ofstream output_file("ground_truth.csv");
    output_file << "timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,v_x,v_y,v_z,b_w_x,b_w_y,b_w_z,b_a_x,b_a_y,b_a_z\n";

    size_t state_idx = 0;


    while (std::getline(input_file, line)) {
      std::stringstream s_stream(line);                // Create a stringstream of the current line

      if (s_stream.good()) {
        std::string time_stamp_str;
        std::getline(s_stream, time_stamp_str, ',');   // get first string delimited by comma
      
        double ground_truth_timestamp = ConverStrTime(time_stamp_str);
        if (time_begin_ <= ground_truth_timestamp && ground_truth_timestamp <= time_end_) {

          if ((state_idx + 1) == state_parameter_.size()) {
          }
          else if (ground_truth_timestamp < state_parameter_.at(state_idx+1)->GetTimestamp()) {
          }
          else {
            // output 
            std::string data;
            output_file << std::to_string(ground_truth_timestamp);
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
            
            // bias
            for (int i=0; i<6; ++i) {                    
              std::getline(s_stream, data, ',');

              output_file << ",";
              output_file << data;
            }

            output_file << std::endl;

            state_idx++;
          }
        }
      }
    }

    input_file.close();
    output_file.close();

    return true;
  }    

  PreIntIMUData Preintegrate(std::vector<IMUData> imu_data_vec) {
    
    Eigen::Matrix3d Delta_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Delta_V = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d Delta_P = Eigen::Vector3d(0, 0, 0);

    for (size_t i=0; i<imu_data_vec.size(); i++) {
      Delta_P = Delta_P + imu_dt_*Delta_V + 0.5*(imu_dt_*imu_dt_)*Delta_R*(imu_data_vec.at(i).accel_ - accel_bias);
      Delta_V = Delta_V + imu_dt_ * Delta_R*(imu_data_vec.at(i).accel_ - accel_bias);
      Delta_R = Delta_R * Exp(imu_dt_ * (imu_data_vec.at(i).gyro_ - gyro_bias));
    }

    return PreIntIMUData(imu_data_vec.size()*imu_dt_, Delta_R, Delta_V, Delta_P);
  }

  bool ReadIMUData(std::string imu_file_path) {
  
    std::cout << "Read IMU data at " << imu_file_path << std::endl;

    std::ifstream input_file(imu_file_path);

    assert(("Could not open IMU file.", input_file.is_open()));

    // Read the column names
    // Extract the first line in the file
    std::string first_line_data_str;
    std::getline(input_file, first_line_data_str);

    // storage of IMU data
    std::vector<IMUData> imu_data_vec;
    size_t state_idx = 0;

    // dead-reckoning for initialization
    Eigen::Quaterniond rotation_dr = state_parameter_.at(0)->GetRotationBlock()->estimate();
    Eigen::Vector3d velocity_dr = state_parameter_.at(0)->GetVelocityBlock()->estimate();
    Eigen::Vector3d position_dr = state_parameter_.at(0)->GetPositionBlock()->estimate();


    std::string imu_data_str;
    while (std::getline(input_file, imu_data_str)) {

      IMUData imu_data(imu_data_str);

      if (time_begin_ <= imu_data.timestamp_ && imu_data.timestamp_ <= time_end_) {


        Eigen::Vector3d accel_measurement = imu_data.accel_;
        Eigen::Vector3d gyro_measurement = imu_data.gyro_;      
        Eigen::Vector3d accel_plus_gravity = rotation_dr.toRotationMatrix()*(imu_data.accel_ - accel_bias) + gravity;
        
        position_dr = position_dr + imu_dt_*velocity_dr + 0.5*(imu_dt_*imu_dt_)*accel_plus_gravity;
        velocity_dr = velocity_dr + imu_dt_*accel_plus_gravity;
        rotation_dr = rotation_dr * Eigen::Quaterniond(1, 0.5*imu_dt_*(imu_data.gyro_(0)-gyro_bias(0)), 
                                                          0.5*imu_dt_*(imu_data.gyro_(1)-gyro_bias(1)), 
                                                          0.5*imu_dt_*(imu_data.gyro_(2)-gyro_bias(2)));
      
        if ((state_idx + 1) == state_parameter_.size()) {
          imu_data_vec.push_back(imu_data);

        }
        else if (imu_data.timestamp_ < state_parameter_.at(state_idx+1)->GetTimestamp()) {
          imu_data_vec.push_back(imu_data);

        }
        else {
          // preintegrate imu_data_vec_small
          PreIntIMUData pre_int_imu_data = Preintegrate(imu_data_vec);
          double dT = pre_int_imu_data.dt_;

          // current state
          Eigen::Quaterniond current_rotation = state_parameter_.at(state_idx)->GetRotationBlock()->estimate();
          Eigen::Vector3d current_velocity = state_parameter_.at(state_idx)->GetVelocityBlock()->estimate();
          Eigen::Vector3d current_position = state_parameter_.at(state_idx)->GetPositionBlock()->estimate();

          // add imu constraint
          ceres::CostFunction* cost_function = new PreIntImuError(pre_int_imu_data.d_rotation_,
                                                                  pre_int_imu_data.d_velocity_,
                                                                  pre_int_imu_data.d_position_,
                                                                  pre_int_imu_data.dt_);

          optimization_problem_.AddResidualBlock(cost_function,
                                                 NULL,
                                                 state_parameter_.at(state_idx+1)->GetRotationBlock()->parameters(),
                                                 state_parameter_.at(state_idx+1)->GetVelocityBlock()->parameters(),
                                                 state_parameter_.at(state_idx+1)->GetPositionBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetRotationBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetVelocityBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetPositionBlock()->parameters());   

          state_idx++;
          
          // dead-reckoning to initialize 
          state_parameter_.at(state_idx)->GetRotationBlock()->setEstimate(rotation_dr);
          state_parameter_.at(state_idx)->GetVelocityBlock()->setEstimate(velocity_dr);
          state_parameter_.at(state_idx)->GetPositionBlock()->setEstimate(position_dr);

          // empty imu_data_vec
          imu_data_vec.clear();

          // add current imu data in
          imu_data_vec.push_back(imu_data);

        }
      }
    }

    input_file.close();
    std::cout << "Finished reading IMU data." << std::endl;
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
    optimization_options_.max_num_iterations = 200;


    for (size_t i=1; i<state_parameter_.size(); ++i) {
      // optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetRotationBlock()->parameters());
      optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetVelocityBlock()->parameters());
      optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetPositionBlock()->parameters());
    }

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }


  bool OutputOptimizationResult() {

    std::ofstream output_file("trajectory.csv");

    output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=1; i<state_parameter_.size(); ++i) {
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
  // testing parameters
  double time_begin_;
  double time_end_;
  int tri_max_num_iterations_;

  // experiment parameters
  double imu_dt_;

  Eigen::Transform<double, 3, Eigen::Affine> T_bc_;
  double focal_length_;
  double principal_point_[2];

  // parameter containers
  std::vector<State*>                   state_parameter_;
  std::vector<LandmarkParameterBlock*>  landmark_parameter_;

  double accel_bias_parameter_[3];
  double gyro_bias_parameter_[3];

  // ceres parameter
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

  std::string observation_file_path = "feature_observation.csv";
  slam_problem.ReadObservationData(observation_file_path);

  slam_problem.ProcessGroundTruth(ground_truth_file_path);

  std::string imu_file_path = euroc_dataset_path + "imu0/data.csv";
  slam_problem.ReadIMUData(imu_file_path);

  slam_problem.SolveOptimizationProblem();
  slam_problem.OutputOptimizationResult();

  return 0;
}