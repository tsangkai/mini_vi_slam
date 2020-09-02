
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include "LandmarkParameterBlock.h"
#include "Timed3dParameterBlock.h"
#include "TimedQuatParameterBlock.h"
#include "ImuError.h"
#include "ReprojectionError.h"


// TODO: avoid data conversion
double ConverStrTime(std::string time_str) {
  std::string time_str_sec = time_str.substr(7,3);       // second
  std::string time_str_nano_sec = time_str.substr(10);   // nano-second

  return std::stoi(time_str_sec) + std::stoi(time_str_nano_sec)*1e-9;
}

class IMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  IMUData(std::string imu_data_str) {
    std::stringstream str_stream(imu_data_str);          // Create a stringstream of the current line

    if (str_stream.good()) {
        
      std::string data_str;
      std::getline(str_stream, data_str, ',');   // get first string delimited by comma
      timestamp_ = ConverStrTime(data_str);

      for (int i=0; i<3; ++i) {                    
        std::getline(str_stream, data_str, ','); 
        gyro_(i) = std::stod(data_str);
      }

      for (int i=0; i<3; ++i) {                    
        std::getline(str_stream, data_str, ','); 
        accel_(i) = std::stod(data_str);
      }
    }
  }

  double getTimeStamp() {
    return timestamp_;
  }

  Eigen::Vector3d getGyroMeasurement() {
    return gyro_;
  }

  Eigen::Vector3d getAccelMeasurement() {
    return accel_;
  }

 private:
  double timestamp_;
  Eigen::Vector3d gyro_;
  Eigen::Vector3d accel_; 
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

  double getTimeStamp() {
    return timestamp_;
  }

  double getId() {
    return index_;
  }

  Eigen::Vector2d getObservation() {
    return feature_pos_;
  }

 private:
  double timestamp_;
  size_t index_;
  Eigen::Vector2d feature_pos_; 
};



class ExpLandmarkOptSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
          }

          Eigen::Vector3d initial_position(std::stod(initial_position_str[0]), std::stod(initial_position_str[1]), std::stod(initial_position_str[2]));
          position_parameter_.push_back(Timed3dParameterBlock(initial_position, 0, ConverStrTime(time_stamp_str)));
          optimization_problem_.AddParameterBlock(position_parameter_.at(0).parameters(), 3);
          optimization_problem_.SetParameterBlockConstant(position_parameter_.at(0).parameters());

          // rotation
          std::string initial_rotation_str[4];
          for (int i=0; i<4; ++i) {                    
            std::getline(s_stream, initial_rotation_str[i], ','); 
          }

          Eigen::Quaterniond initial_rotation(std::stod(initial_rotation_str[0]), std::stod(initial_rotation_str[1]), std::stod(initial_rotation_str[2]), std::stod(initial_rotation_str[3]));
          rotation_parameter_.push_back(TimedQuatParameterBlock(Eigen::Quaterniond(), 0, ConverStrTime(time_stamp_str)));
          optimization_problem_.AddParameterBlock(rotation_parameter_.at(0).parameters(), 4);
          optimization_problem_.SetParameterBlockConstant(rotation_parameter_.at(0).parameters());

          // velocity
          std::string initial_velocity_str[3];
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, initial_velocity_str[i], ','); 
          }

          Eigen::Vector3d initial_velocity(std::stod(initial_velocity_str[0]), std::stod(initial_velocity_str[1]), std::stod(initial_velocity_str[2]));
          velocity_parameter_.push_back(Timed3dParameterBlock(initial_velocity, 0, ConverStrTime(time_stamp_str)));
          optimization_problem_.AddParameterBlock(velocity_parameter_.at(0).parameters(), 3);
          optimization_problem_.SetParameterBlockConstant(velocity_parameter_.at(0).parameters());

          return true;
        }
      }
    }

    std::cout << "Initialization fails!" << std::endl;
    return false;
  }  


  bool readIMUData(std::string imu_file_path) {
  

    std::ifstream input_file(imu_file_path);
    
    if(!input_file.is_open()) 
      throw std::runtime_error("Could not open file");

    // Read the column names
    // Extract the first line in the file
    std::string first_line_data_str;
    std::getline(input_file, first_line_data_str);


    std::vector<IMUData> imu_data_vec;

    std::string imu_data_str;
    while (std::getline(input_file, imu_data_str)) {

      IMUData imu_data(imu_data_str);

      if (time_begin_ <= imu_data.getTimeStamp() && imu_data.getTimeStamp() <= time_end_) {

        imu_data_vec.push_back(imu_data);

        if (imu_data_vec.size()>1) {
          position_parameter_.push_back(Timed3dParameterBlock(Eigen::Vector3d(), imu_data_vec.size()-1, imu_data.getTimeStamp()));
          velocity_parameter_.push_back(Timed3dParameterBlock(Eigen::Vector3d(), imu_data_vec.size()-1, imu_data.getTimeStamp()));
          rotation_parameter_.push_back(TimedQuatParameterBlock(Eigen::Quaterniond(), imu_data_vec.size()-1, imu_data.getTimeStamp()));
        }

      }
    }

    // dead-reckoning to initialize 
    for (size_t i=0; i<imu_data_vec.size()-1; ++i) {
        
      double time_diff = position_parameter_.at(i+1).timestamp() - position_parameter_.at(i).timestamp();

      Eigen::Vector3d accel_measurement = imu_data_vec.at(i).getAccelMeasurement();
      Eigen::Vector3d gyro_measurement = imu_data_vec.at(i).getGyroMeasurement();      
      Eigen::Vector3d accel_plus_gravity = rotation_parameter_.at(i).estimate().normalized().toRotationMatrix()*accel_measurement + Eigen::Vector3d(0, 0, -9.81007);
      Eigen::Vector3d position_t1 = position_parameter_.at(i).estimate() + time_diff*velocity_parameter_.at(i).estimate() + (0.5*time_diff*time_diff) * accel_plus_gravity;
      Eigen::Vector3d velocity_t1 = velocity_parameter_.at(i).estimate() + time_diff*time_diff * accel_plus_gravity;
      Eigen::Quaterniond rotation_t1 = rotation_parameter_.at(i).estimate().normalized() * Eigen::Quaterniond(1, 0.5*time_diff*gyro_measurement(0), 0.5*time_diff*gyro_measurement(1), 0.5*time_diff*gyro_measurement(2));

      position_parameter_.at(i+1).setEstimate(position_t1);
      velocity_parameter_.at(i+1).setEstimate(velocity_t1);
      rotation_parameter_.at(i+1).setEstimate(rotation_t1);

      // add constraints
      ceres::CostFunction* cost_function = new ImuError(imu_data_vec.at(i).getGyroMeasurement(),
                                                        imu_data_vec.at(i).getAccelMeasurement(),
                                                        time_diff);

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             position_parameter_.at(i+1).parameters(),
                                             velocity_parameter_.at(i+1).parameters(),
                                             rotation_parameter_.at(i+1).parameters(),
                                             position_parameter_.at(i).parameters(),
                                             velocity_parameter_.at(i).parameters(),
                                             rotation_parameter_.at(i).parameters());    
    }

    return true;
  }


  bool readObservationData(std::string observation_file_path) {
  

    std::ifstream input_file(observation_file_path);
    
    if(!input_file.is_open())
      throw std::runtime_error("Could not open file");

    std::string first_line_data_str;
    std::getline(input_file, first_line_data_str);

    std::string observation_data_str;
    while (std::getline(input_file, observation_data_str)) {
      ObservationData observation_data(observation_data_str);

      if (observation_data.getId() > landmark_parameter_.size()) {

        landmark_parameter_.push_back(LandmarkParameterBlock(Eigen::Vector3d(), observation_data.getId()-1));
      }

      // add observation constraints
      size_t pose_id;
      size_t landmark_id = observation_data.getId()-1;

      for (size_t i=0; i<position_parameter_.size(); ++i) {
        if (observation_data.getTimeStamp() <= position_parameter_.at(i).timestamp()) {
          pose_id = i;
          break;
        }
      }

      std::cout << pose_id << " " << landmark_id << std::endl;

      ceres::CostFunction* cost_function = new ReprojectionError(observation_data.getObservation(),
                                                                 focal_length_,
                                                                 principal_point_);

      std::cout << pose_id << " position prt: " << position_parameter_.at(pose_id).parameters() << std::endl;
      for  (size_t i=0; i<rotation_parameter_.size(); ++i) {
        std::cout << i << " rotation prt: " << rotation_parameter_.at(i).parameters() << std::endl;
      }
      std::cout << landmark_id << " landmark prt: " << landmark_parameter_.at(landmark_id).parameters() << std::endl;

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             position_parameter_.at(pose_id).parameters(),
                                             rotation_parameter_.at(pose_id).parameters(),
                                             landmark_parameter_.at(landmark_id).parameters());     



    }

    std::cout << "Finish reading observation data." << std::endl;
    return true;
  }


  bool solveOptimizationProblem() {
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.

    optimization_options_.linear_solver_type = ceres::DENSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 10;

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }


 private:
  double time_begin_;
  double time_end_;

  // camera parameters
  Eigen::Matrix4d T_BC_;    // from camera frame to body frame
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
  slam_problem.readObservationData(observation_file_path);

  // slam_problem.solveOptimizationProblem();


  return 0;
}