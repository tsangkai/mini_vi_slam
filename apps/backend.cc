
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include "SizedParameterBlock.h"
#include "LandmarkParameterBlock.h"
#include "Timed3dParameterBlock.h"
#include "TimedQuatParameterBlock.h"

// TODO: avoid data conversion
double ConverStrTime(std::string time_str) {
  std::string time_str_sec = time_str.substr(7,3);       // second
  std::string time_str_nano_sec = time_str.substr(10);   // nano-second

  return std::stoi(time_str_sec) + std::stoi(time_str_nano_sec)*1e-9;
}

class ExpLandmarkOptSLAM {

 public:

  bool readConfigurationFiles(std::string config_folder_path) {

    // test configuration file
    cv::FileStorage test_config_file(config_folder_path + "test.yaml", cv::FileStorage::READ);
    time_begin_ = ConverStrTime(test_config_file["time_window"][0]);  
    time_end_ = ConverStrTime(test_config_file["time_window"][1]);  


    // experiment configuration file
    cv::FileStorage experiment_config_file(config_folder_path + "config_fpga_p2_euroc.yaml", cv::FileStorage::READ);

    cv::FileNode T_BC_node = experiment_config_file["cameras"][0]["T_SC"];  // from camera frame to body frame

    T_BC_ << T_BC_node[0],  T_BC_node[1],  T_BC_node[2],  T_BC_node[3], 
             T_BC_node[4],  T_BC_node[5],  T_BC_node[6],  T_BC_node[7], 
             T_BC_node[8],  T_BC_node[9],  T_BC_node[10], T_BC_node[11], 
             T_BC_node[12], T_BC_node[13], T_BC_node[14], T_BC_node[15];

    double focal_length_0 = experiment_config_file["cameras"][0]["focal_length"][0];  // i don't know the unit!!!!
    double focal_length_1 = experiment_config_file["cameras"][0]["focal_length"][1];
    focal_length_ = 0.5*focal_length_0 + 0.5*focal_length_1;

    principal_point_[0] = experiment_config_file["cameras"][0]["principal_point"][0];
    principal_point_[1] = experiment_config_file["cameras"][0]["principal_point"][1];
    
    return 1;
  }

  bool readInitialCondition(std::string ground_truth_file_path) {

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

          // position
          std::string initial_position_str[3];
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, initial_position_str[i], ','); 
            std::cout << std::stod(initial_position_str[i]) << std::endl;
          }

          Eigen::Vector3d initial_position(std::stod(initial_position_str[0]), std::stod(initial_position_str[1]), std::stod(initial_position_str[2]));
          position_parameter_.push_back(Timed3dParameterBlock(initial_position, 0, std::stod(time_stamp_str)));
          optimization_problem_.AddParameterBlock(position_parameter_.at(0).parameters(), 3);
          optimization_problem_.SetParameterBlockConstant(position_parameter_.at(0).parameters());

          // rotation
          std::string initial_rotation_str[4];
          for (int i=0; i<4; ++i) {                    
            std::getline(s_stream, initial_rotation_str[i], ','); 
            std::cout << std::stod(initial_rotation_str[i]) << std::endl;
          }

          Eigen::Quaterniond initial_rotation(std::stod(initial_rotation_str[0]), std::stod(initial_rotation_str[1]), std::stod(initial_rotation_str[2]), std::stod(initial_rotation_str[3]));
          rotation_parameter_.push_back(TimedQuatParameterBlock(initial_rotation, 0, std::stod(time_stamp_str)));
          optimization_problem_.AddParameterBlock(rotation_parameter_.at(0).parameters(), 4);
          optimization_problem_.SetParameterBlockConstant(rotation_parameter_.at(0).parameters());

          // velocity
          std::string initial_velocity_str[3];
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, initial_velocity_str[i], ','); 
            std::cout << std::stod(initial_velocity_str[i]) << std::endl;
          }

          Eigen::Vector3d initial_velocity(std::stod(initial_velocity_str[0]), std::stod(initial_velocity_str[1]), std::stod(initial_velocity_str[2]));
          velocity_parameter_.push_back(Timed3dParameterBlock(initial_velocity, 0, std::stod(time_stamp_str)));
          optimization_problem_.AddParameterBlock(velocity_parameter_.at(0).parameters(), 3);
          optimization_problem_.SetParameterBlockConstant(velocity_parameter_.at(0).parameters());

          return 1;
        }
      }
    }

    std::cout << "Initialization fails!" << std::endl;
    return 0;
  }  


  bool readIMUData(std::string imu_file_path) {
  
    return 1;
  }

 private:
  double time_begin_;
  double time_end_;

  // camera parameters
  Eigen::Matrix4d T_BC_; // from camera frame to body frame
  double focal_length_;
  double principal_point_[2];

  // data storage
  std::vector<TimedQuatParameterBlock> rotation_parameter_;
  std::vector<Timed3dParameterBlock> position_parameter_;
  std::vector<Timed3dParameterBlock> velocity_parameter_;
  double accel_bias_parameter_[3];
  double gyro_bias_parameter_[3];

  std::vector<LandmarkParameterBlock> landmark_parameter_;

  // ceres parameter
  ceres::Problem optimization_problem_;
  ceres::Solver::Options optimization_options_;
  ceres::Solver::Summary optimization_summary_;


};



int main(int argc, char **argv) {

  /*** Step 0. Configuration files ***/

  ExpLandmarkOptSLAM slam_problem;

  std::string config_folder_path = "../config/";
  slam_problem.readConfigurationFiles(config_folder_path);


  /*** Step 1. Datasets ***/
  std::string euroc_dataset_path = "../../../dataset/mav0/";
  std::string ground_truth_file_path = euroc_dataset_path + "state_groundtruth_estimate0/data.csv";
  slam_problem.readInitialCondition(ground_truth_file_path);

  std::string imu_file_path = euroc_dataset_path + "imu0/data.csv";
  slam_problem.readIMUData(imu_file_path);

  std::string observation_file_path = "feature_observation.csv";
  // slam_problem.readObservationData(observation_file_path);


  return 0;
}