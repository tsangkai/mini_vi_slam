
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>

#include "landmark_parameter_block.h"
#include "timed_3d_parameter_block.h"
#include "timed_quat_parameter_block.h"
#include "imu_error.h"
#include "reprojection_error.h"


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

  double GetTimestamp() {
    return timestamp_;
  }

  Eigen::Vector3d GetGyroMeasurement() {
    return gyro_;
  }

  Eigen::Vector3d GetAccelMeasurement() {
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

    cv::FileNode T_BC_node = experiment_config_file["cameras"][0]["T_SC"];            // from camera frame to body frame

    Eigen::Matrix4d T_BC;
    T_BC  << T_BC_node[0],  T_BC_node[1],  T_BC_node[2],  T_BC_node[3], 
             T_BC_node[4],  T_BC_node[5],  T_BC_node[6],  T_BC_node[7], 
             T_BC_node[8],  T_BC_node[9],  T_BC_node[10], T_BC_node[11], 
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
          position_parameter_.push_back(new Timed3dParameterBlock(initial_position, ConverStrTime(time_stamp_str)));
          optimization_problem_.AddParameterBlock(position_parameter_.at(0)->parameters(), 3);
          optimization_problem_.SetParameterBlockConstant(position_parameter_.at(0)->parameters());

          // rotation
          std::string initial_rotation_str[4];
          for (int i=0; i<4; ++i) {                    
            std::getline(s_stream, initial_rotation_str[i], ','); 
          }

          Eigen::Quaterniond initial_rotation(std::stod(initial_rotation_str[0]), std::stod(initial_rotation_str[1]), std::stod(initial_rotation_str[2]), std::stod(initial_rotation_str[3]));
          rotation_parameter_.push_back(new TimedQuatParameterBlock(initial_rotation, ConverStrTime(time_stamp_str)));
          optimization_problem_.AddParameterBlock(rotation_parameter_.at(0)->parameters(), 4);
          optimization_problem_.SetParameterBlockConstant(rotation_parameter_.at(0)->parameters());

          // velocity
          std::string initial_velocity_str[3];
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, initial_velocity_str[i], ','); 
          }

          Eigen::Vector3d initial_velocity(std::stod(initial_velocity_str[0]), std::stod(initial_velocity_str[1]), std::stod(initial_velocity_str[2]));
          velocity_parameter_.push_back(new Timed3dParameterBlock(initial_velocity, ConverStrTime(time_stamp_str)));
          optimization_problem_.AddParameterBlock(velocity_parameter_.at(0)->parameters(), 3);
          optimization_problem_.SetParameterBlockConstant(velocity_parameter_.at(0)->parameters());

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
    output_file << "timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z,b_w_x,b_w_y,b_w_z,b_a_x,b_a_y,b_a_z\n";

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

          // ignore velocity part
          for (int i=0; i<3; ++i) {                    
            std::getline(s_stream, data, ',');
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

      if (time_begin_ <= imu_data.GetTimestamp() && imu_data.GetTimestamp() <= time_end_) {

        imu_data_vec.push_back(imu_data);

        if (imu_data_vec.size()>1) {
          rotation_parameter_.push_back(new TimedQuatParameterBlock(Eigen::Quaterniond(), imu_data.GetTimestamp()));
          position_parameter_.push_back(new Timed3dParameterBlock(Eigen::Vector3d(), imu_data.GetTimestamp()));
          velocity_parameter_.push_back(new Timed3dParameterBlock(Eigen::Vector3d(), imu_data.GetTimestamp()));
        }
      }
    }

    // dead-reckoning to initialize 
    for (size_t i=0; i<imu_data_vec.size()-1; ++i) {
        
      double time_diff = position_parameter_.at(i+1)->timestamp() - position_parameter_.at(i)->timestamp();

      // for debugging now
      // parameters are from ground truth data currently, and can be estimatd in the optimization problem later
      Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      
      Eigen::Vector3d gyro_bias = Eigen::Vector3d(-0.003196, 0.021298, 0.078430);
      Eigen::Vector3d accel_bias = Eigen::Vector3d(-0.026176, 0.137568, 0.076295);

      Eigen::Vector3d accel_measurement = imu_data_vec.at(i).GetAccelMeasurement();
      Eigen::Vector3d gyro_measurement = imu_data_vec.at(i).GetGyroMeasurement();      
      Eigen::Vector3d accel_plus_gravity = rotation_parameter_.at(i)->estimate().normalized().toRotationMatrix()*(accel_measurement - accel_bias) + gravity;
      
      Eigen::Quaterniond rotation_t_plus_1 = rotation_parameter_.at(i)->estimate().normalized() * Eigen::Quaterniond(1, 0.5*time_diff*(gyro_measurement(0)-gyro_bias(0)), 
                                                                                                                        0.5*time_diff*(gyro_measurement(1)-gyro_bias(1)), 
                                                                                                                        0.5*time_diff*(gyro_measurement(2)-gyro_bias(2)));
      Eigen::Vector3d position_t_plus_1 = position_parameter_.at(i)->estimate() + time_diff*velocity_parameter_.at(i)->estimate() + (0.5*time_diff*time_diff)*accel_plus_gravity;
      Eigen::Vector3d velocity_t_plus_1 = velocity_parameter_.at(i)->estimate() + time_diff*accel_plus_gravity;



      rotation_parameter_.at(i+1)->setEstimate(rotation_t_plus_1);
      position_parameter_.at(i+1)->setEstimate(position_t_plus_1);
      velocity_parameter_.at(i+1)->setEstimate(velocity_t_plus_1);
      
      // add constraints
      ceres::CostFunction* cost_function = new ImuError(imu_data_vec.at(i).GetGyroMeasurement(),
                                                        imu_data_vec.at(i).GetAccelMeasurement(),
                                                        time_diff);

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             rotation_parameter_.at(i+1)->parameters(),
                                             position_parameter_.at(i+1)->parameters(),
                                             velocity_parameter_.at(i+1)->parameters(),
                                             rotation_parameter_.at(i)->parameters(),
                                             position_parameter_.at(i)->parameters(),
                                             velocity_parameter_.at(i)->parameters());    

      optimization_problem_.SetParameterLowerBound(position_parameter_.at(i+1)->parameters(), 0, -2.8);
      optimization_problem_.SetParameterLowerBound(position_parameter_.at(i+1)->parameters(), 1,  4.2);
      optimization_problem_.SetParameterLowerBound(position_parameter_.at(i+1)->parameters(), 2, -1.8);

      optimization_problem_.SetParameterUpperBound(position_parameter_.at(i+1)->parameters(), 0, 1.8);
      optimization_problem_.SetParameterUpperBound(position_parameter_.at(i+1)->parameters(), 1, 8.8);
      optimization_problem_.SetParameterUpperBound(position_parameter_.at(i+1)->parameters(), 2, 2.8);
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
      size_t landmark_id = observation_data.GetId()-1;

      for (size_t i=0; i<position_parameter_.size(); ++i) {
        if (observation_data.GetTimestamp() <= position_parameter_.at(i)->timestamp()) {
          pose_id = i;
          break;
        }
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
                                             rotation_parameter_.at(pose_id)->parameters(),
                                             position_parameter_.at(pose_id)->parameters(),
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

    for (size_t i=1; i<position_parameter_.size(); ++i) {
      optimization_problem_.SetParameterBlockConstant(rotation_parameter_.at(i)->parameters());
      optimization_problem_.SetParameterBlockConstant(position_parameter_.at(i)->parameters());
      optimization_problem_.SetParameterBlockConstant(velocity_parameter_.at(i)->parameters());
    }

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";


    // Step 2: optimize trajectory
    optimization_options_.max_num_iterations = 30;

    for (size_t i=1; i<position_parameter_.size(); ++i) {
      optimization_problem_.SetParameterBlockVariable(rotation_parameter_.at(i)->parameters());
      optimization_problem_.SetParameterBlockVariable(position_parameter_.at(i)->parameters());
      optimization_problem_.SetParameterBlockVariable(velocity_parameter_.at(i)->parameters());
    }

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }

  bool OutputOptimizationResult() {

    std::ofstream output_file("trajectory.csv");

    output_file << "timestamp,p_x,p_y,p_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=0; i<rotation_parameter_.size(); ++i) {
      output_file << std::to_string(rotation_parameter_.at(i)->timestamp()) << ",";
      output_file << std::to_string(position_parameter_.at(i)->estimate()(0)) << ",";
      output_file << std::to_string(position_parameter_.at(i)->estimate()(1)) << ",";
      output_file << std::to_string(position_parameter_.at(i)->estimate()(2)) << ",";
      output_file << std::to_string(rotation_parameter_.at(i)->estimate().w()) << ",";
      output_file << std::to_string(rotation_parameter_.at(i)->estimate().x()) << ",";
      output_file << std::to_string(rotation_parameter_.at(i)->estimate().y()) << ",";
      output_file << std::to_string(rotation_parameter_.at(i)->estimate().z()) << std::endl;
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

  // camera intrinsic parameters
  Eigen::Transform<double, 3, Eigen::Affine> T_bc_;
  double focal_length_;
  double principal_point_[2];

  // data storage (parameters to be optimized)
  std::vector<TimedQuatParameterBlock*> rotation_parameter_;
  std::vector<Timed3dParameterBlock*>   position_parameter_;
  std::vector<Timed3dParameterBlock*>   velocity_parameter_;
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
  slam_problem.ProcessGroundTruth(ground_truth_file_path);

  std::string imu_file_path = euroc_dataset_path + "imu0/data.csv";
  slam_problem.ReadIMUData(imu_file_path);

  std::string observation_file_path = "feature_observation.csv";
  slam_problem.ReadObservationData(observation_file_path);

  slam_problem.SolveOptimizationProblem();

  slam_problem.OutputOptimizationResult();

  return 0;
}