
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


// TODO: move this term to somewhere else
Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      

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

struct PreIntIMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreIntIMUData(Eigen::Vector3d bias_gyr,
                Eigen::Vector3d bias_acc,
                double sigma_g_c, 
                double sigma_a_c) {

    bias_gyr_ = bias_gyr;
    bias_acc_ = bias_acc;

    sigma_g_c_ = sigma_g_c;
    sigma_a_c_ = sigma_a_c;

    dt_ = 0;

    dR_ = Eigen::Matrix3d::Identity();
    dv_ = Eigen::Vector3d(0, 0, 0);
    dp_ = Eigen::Vector3d(0, 0, 0);

    cov_.setZero();
  }
  
  // the imu_data is measured at imu_data.dt_
  // and this imu_data lasts for imu_dt
  // assume this imu_data is constant over the interval
  bool IntegrateSingleIMU(IMUData imu_data, double imu_dt) {

    Eigen::Vector3d gyr = imu_data.gyr_ - bias_gyr_;
    Eigen::Vector3d acc = imu_data.acc_ - bias_acc_;



    // covariance update
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    F.block<3,3>(3,0) = (-1) * dR_ * Hat(acc) * imu_dt;
    F.block<3,3>(6,0) = (-0.5) * dR_ * Hat(acc)*imu_dt*imu_dt;
    F.block<3,3>(6,3) = imu_dt * Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 9, 6> G;
    G.setZero();
    G.block<3,3>(0,0) = dR_.transpose()*LeftJacobian(gyr*imu_dt)*imu_dt;
    G.block<3,3>(3,3) = dR_.transpose()*imu_dt;
    G.block<3,3>(6,3) = 0.5*dR_.transpose()*imu_dt*imu_dt;

    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Identity();
    Q.block<3,3>(0,0) = (sigma_g_c_ * sigma_g_c_ / imu_dt) * Eigen::Matrix3d::Identity();
    Q.block<3,3>(3,3) = (sigma_a_c_ * sigma_a_c_ / imu_dt) * Eigen::Matrix3d::Identity();

    cov_ = F * cov_ * F.transpose() + G * Q * G.transpose();

    // deviation update
    dp_ = dp_ + imu_dt * dv_ + 0.5 * (imu_dt * imu_dt) * dR_ * acc;
    dv_ = dv_ + imu_dt * dR_ * acc;
    dR_ = dR_ * Exp(imu_dt * gyr);

    dt_ = dt_ + imu_dt;

    return true;
  }

  Eigen::Vector3d bias_gyr_;
  Eigen::Vector3d bias_acc_;

  double sigma_g_c_;
  double sigma_a_c_;

  double dt_;
  Eigen::Matrix3d dR_;  
  Eigen::Vector3d dv_;
  Eigen::Vector3d dp_; 

  Eigen::Matrix<double, 9, 9> cov_;
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

  Eigen::Matrix2d cov() {
    double sigma_2 = size_ * size_ / 64.0;
    return sigma_2 * Eigen::Matrix2d::Identity();
  }

  double timestamp_;
  size_t index_;
  Eigen::Vector2d feature_pos_; 
  double size_;
};


// This class only handles the memory. The underlying estimate is handled by each parameter blocks.
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


struct TriangularData {
  TriangularData(size_t state_idx, Eigen::Vector2d keypoint) {
    state_idx_ = state_idx;
    keypoint_ = keypoint;
  }

  size_t state_idx_;
  Eigen::Vector2d keypoint_;
};

Eigen::Vector3d EpipolarInitialize(Eigen::Vector2d keypoint1, Eigen::Quaterniond q1, Eigen::Vector3d p1_n1,
                                   Eigen::Vector2d keypoint2, Eigen::Quaterniond q2, Eigen::Vector3d p2_n2,
                                   Eigen::Matrix4d T_bc, 
                                   double fu, double fv,
                                   double cu, double cv) {

  Eigen::Matrix3d K;
  K << fu,  0, 0,
        0, fv, 0,
        0,  0, 1;

  Eigen::Matrix<double, 3, 4> projection;
  projection << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0;

  Eigen::Vector2d kp1 = keypoint1 - Eigen::Vector2d(cu, cv);
  Eigen::Vector2d kp2 = keypoint2 - Eigen::Vector2d(cu, cv);

  // from body frame to camera frame
  Eigen::Matrix3d R_bc = T_bc.topLeftCorner<3,3>();
  Eigen::Matrix3d R_cb = R_bc.transpose();
  Eigen::Vector3d t_bc = T_bc.topRightCorner<3,1>();

  Eigen::Matrix4d T_cb = Eigen::Matrix4d::Identity();
  T_cb.topLeftCorner<3,3>() = R_cb;
  T_cb.topRightCorner<3,1>() = -R_cb * t_bc;

  // from nagivation frame to body frame
  Eigen::Matrix3d R_nb_1 = q1.toRotationMatrix();
  Eigen::Matrix3d R_bn_1 = R_nb_1.transpose();

  Eigen::Matrix4d T_bn_1 = Eigen::Matrix4d::Identity();
  T_bn_1.topLeftCorner<3,3>() = R_bn_1;
  T_bn_1.topRightCorner<3,1>() = -R_bn_1 * p1_n1;

  Eigen::Matrix3d R_nb_2 = q2.toRotationMatrix();
  Eigen::Matrix3d R_bn_2 = R_nb_2.transpose();

  Eigen::Matrix4d T_bn_2 = Eigen::Matrix4d::Identity();
  T_bn_2.topLeftCorner<3,3>() = R_bn_2;
  T_bn_2.topRightCorner<3,1>() = -R_bn_2 * p2_n2;


  // 
  Eigen::Matrix<double, 3, 4> P1 = K * projection * T_cb * T_bn_1;
  Eigen::Matrix<double, 3, 4> P2 = K * projection * T_cb * T_bn_2;

  //
  Eigen::Matrix4d A;
  A.row(0) = kp1(0) * P1.row(2) - P1.row(0);
  A.row(1) = kp1(1) * P1.row(2) - P1.row(1);
  A.row(2) = kp2(0) * P2.row(2) - P2.row(0);
  A.row(3) = kp2(1) * P2.row(2) - P2.row(1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4d vec = svd.matrixV().col(3);

  return vec.head(3) / vec(3);
}


class ExpLandmarkOptSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkOptSLAM(std::string config_folder_path) {
    ReadConfigurationFiles(config_folder_path);

    quat_parameterization_ptr_ = new ceres::QuaternionParameterization();

    bias_gyr_.setZero();
    bias_acc_.setZero();

  }

  bool ReadConfigurationFiles(std::string config_folder_path) {

    // test configuration file
    cv::FileStorage test_config_file(config_folder_path + "test.yaml", cv::FileStorage::READ);
    time_begin_ = ConverStrTime(test_config_file["time_window"][0]);  
    time_end_ = ConverStrTime(test_config_file["time_window"][1]);  

    // experiment configuration file
    cv::FileStorage experiment_config_file(config_folder_path + "config_fpga_p2_euroc.yaml", cv::FileStorage::READ);

    imu_dt_ = 1.0 / (double) experiment_config_file["imu_params"]["imu_rate"]; 

    cv::FileNode T_BC_node = experiment_config_file["cameras"][0]["T_SC"];            // from camera frame to body frame

    T_bc_  <<  T_BC_node[0],  T_BC_node[1],  T_BC_node[2],  T_BC_node[3], 
               T_BC_node[4],  T_BC_node[5],  T_BC_node[6],  T_BC_node[7], 
               T_BC_node[8],  T_BC_node[9], T_BC_node[10], T_BC_node[11], 
              T_BC_node[12], T_BC_node[13], T_BC_node[14], T_BC_node[15];

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
      observation_data_.push_back(observation_data);


      // add observation constraints
      // size_t pose_id;

      if (state_parameter_.back()->GetTimestamp() < observation_data.timestamp_) {
          state_parameter_.push_back(new State(observation_data.timestamp_));

          optimization_problem_.AddParameterBlock(state_parameter_.back()->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
          optimization_problem_.AddParameterBlock(state_parameter_.back()->GetVelocityBlock()->parameters(), 3);
          optimization_problem_.AddParameterBlock(state_parameter_.back()->GetPositionBlock()->parameters(), 3);
      }
      else if (state_parameter_.back()->GetTimestamp() == observation_data.timestamp_) {
      }
      else if (state_parameter_.back()->GetTimestamp() > observation_data.timestamp_) {
          std::cout << "error!";
      }

      size_t landmark_id = observation_data.index_-1;

      if (landmark_id >= landmark_parameter_.size()) {
        landmark_parameter_.resize(landmark_id+1);
      }
      // should know how many landmarks in total!


    }

    for (size_t i=0; i<landmark_parameter_.size(); ++i) {
      landmark_parameter_.at(i) = new Vec3dParameterBlock();
    }


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
    double num = 0.0;


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
            num = num + 1;

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
            
            // gyro bias
            std::string b_gyr[3];
            for (int i=0; i<3; ++i) {                    
              std::getline(s_stream, b_gyr[i], ',');

              output_file << ",";
              output_file << b_gyr[i];
            }
            bias_gyr_ = (1 - 1/num) * bias_gyr_ + (1/num) * Eigen::Vector3d(std::stod(b_gyr[0]), std::stod(b_gyr[1]), std::stod(b_gyr[2]));


            // acce bias
            std::string b_acc[3];
            for (int i=0; i<3; ++i) {                    
              std::getline(s_stream, b_acc[i], ',');

              output_file << ",";
              output_file << b_acc[i];
            }
            bias_acc_ = (1 - 1/num) * bias_acc_ + (1/num) * Eigen::Vector3d(std::stod(b_acc[0]), std::stod(b_acc[1]), std::stod(b_acc[2]));


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
    size_t state_idx = 0;                 // the index of the last element

    PreIntIMUData int_imu_data(bias_gyr_,
                               bias_acc_,
                               sigma_g_c_,
                               sigma_a_c_);
    
    // dead-reckoning for initialization
    Eigen::Quaterniond rotation_dr = state_parameter_.at(0)->GetRotationBlock()->estimate();
    Eigen::Vector3d velocity_dr = state_parameter_.at(0)->GetVelocityBlock()->estimate();
    Eigen::Vector3d position_dr = state_parameter_.at(0)->GetPositionBlock()->estimate();


    std::string imu_data_str;
    while (std::getline(input_file, imu_data_str)) {

      IMUData imu_data(imu_data_str);

      if (time_begin_ <= imu_data.timestamp_ && imu_data.timestamp_ <= time_end_) {

        Eigen::Vector3d acc_measurement = imu_data.acc_;
        Eigen::Vector3d gyr_measurement = imu_data.gyr_;      
        Eigen::Vector3d accel_plus_gravity = rotation_dr.toRotationMatrix()*(imu_data.acc_ - bias_acc_) + gravity;
        
        position_dr = position_dr + imu_dt_*velocity_dr + 0.5*(imu_dt_*imu_dt_)*accel_plus_gravity;
        velocity_dr = velocity_dr + imu_dt_*accel_plus_gravity;
        rotation_dr = rotation_dr * Exp_q(imu_dt_*(gyr_measurement-bias_gyr_));


        // starting to put imu data in the previously established state_parameter_
        // case 1: the time stamp of the imu data is after the last state
        if ((state_idx + 1) == state_parameter_.size()) {
          imu_data_vec.push_back(imu_data);

          int_imu_data.IntegrateSingleIMU(imu_data, imu_dt_);
        }
        // case 2: the time stamp of the imu data is between two consecutive states
        else if (imu_data.timestamp_ < state_parameter_.at(state_idx+1)->GetTimestamp()) {
          imu_data_vec.push_back(imu_data);

          int_imu_data.IntegrateSingleIMU(imu_data, imu_dt_);
        }
        // case 3: the imu data just enter the new interval of integration
        else {

          // add imu constraint
          ceres::CostFunction* cost_function = new PreIntImuError(int_imu_data.dt_,
                                                                  int_imu_data.dR_,
                                                                  int_imu_data.dv_,
                                                                  int_imu_data.dp_);

          optimization_problem_.AddResidualBlock(cost_function,
                                                 NULL,
                                                 state_parameter_.at(state_idx+1)->GetRotationBlock()->parameters(),
                                                 state_parameter_.at(state_idx+1)->GetVelocityBlock()->parameters(),
                                                 state_parameter_.at(state_idx+1)->GetPositionBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetRotationBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetVelocityBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetPositionBlock()->parameters());   

          state_idx++;
          
          // dead-reckoning to initialize states
          state_parameter_.at(state_idx)->GetRotationBlock()->setEstimate(rotation_dr);
          state_parameter_.at(state_idx)->GetVelocityBlock()->setEstimate(velocity_dr);
          state_parameter_.at(state_idx)->GetPositionBlock()->setEstimate(position_dr);

          // prepare for next iteration
          int_imu_data = PreIntIMUData(bias_gyr_,
                                       bias_acc_,
                                       sigma_g_c_,
                                       sigma_a_c_);

          int_imu_data.IntegrateSingleIMU(imu_data, imu_dt_);

        }
      }
    }

    input_file.close();
    std::cout << "Finished reading IMU data." << std::endl;
    return true;
  }

  bool Triangulate() {

    std::cout << "Begin triangularization to initialize landmark estimates." << std::endl;

    std::vector<std::vector<TriangularData>> tri_data;
    tri_data.resize(observation_data_.size());

    for (size_t i=0; i<observation_data_.size(); ++i) {

      ObservationData observation_data = observation_data_.at(i);


      // determine the two nodes of the bipartite graph
      size_t state_idx = 0;
      for (size_t j=0; j<state_parameter_.size(); ++j) {
        if (state_parameter_.at(j)->GetTimestamp() == observation_data.timestamp_) {
          state_idx = j;
          break;
        }
      }

      size_t landmark_idx = observation_data.index_-1;  

      if (tri_data.at(landmark_idx).empty()) {
        TriangularData tri_data_instance(state_idx, observation_data.feature_pos_);
        tri_data.at(landmark_idx).push_back(tri_data_instance);
      }
      else if (tri_data.at(landmark_idx).size() == 1 && tri_data.at(landmark_idx).at(0).state_idx_!=state_idx) {
        TriangularData tri_data_instance(state_idx, observation_data.feature_pos_);
        tri_data.at(landmark_idx).push_back(tri_data_instance);
      }

      ceres::CostFunction* cost_function = new ReprojectionError(observation_data.feature_pos_,
                                                                 T_bc_,
                                                                 fu_, fv_,
                                                                 cu_, cv_,
                                                                 observation_data.cov());

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             state_parameter_.at(state_idx)->GetRotationBlock()->parameters(),
                                             state_parameter_.at(state_idx)->GetPositionBlock()->parameters(),
                                             landmark_parameter_.at(landmark_idx)->parameters());
    }

    // 
    for (size_t i=0; i<landmark_parameter_.size(); ++i) {

      Eigen::Vector2d keypoint_0 = tri_data.at(i).at(0).keypoint_;
      Eigen::Vector2d keypoint_1 = tri_data.at(i).at(1).keypoint_;

      size_t state_idx_0 = tri_data.at(i).at(0).state_idx_;
      size_t state_idx_1 = tri_data.at(i).at(1).state_idx_;

      Eigen::Vector3d init_landmark_pos = EpipolarInitialize(keypoint_0, state_parameter_.at(state_idx_0)->GetRotationBlock()->estimate(), state_parameter_.at(state_idx_0)->GetPositionBlock()->estimate(),
                                                             keypoint_1, state_parameter_.at(state_idx_1)->GetRotationBlock()->estimate(), state_parameter_.at(state_idx_1)->GetPositionBlock()->estimate(),
                                                             T_bc_,
                                                             fu_, fv_,
                                                             cu_, cv_);

      landmark_parameter_.at(i)->setEstimate(init_landmark_pos);

    }

    return true;
  }

  bool SolveOptimizationProblem() {

    std::cout << "Begin solving the optimization problem." << std::endl;

    optimization_options_.linear_solver_type = ceres::DENSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 6;
    optimization_options_.function_tolerance = 1e-9;
    optimization_options_.max_num_iterations = 100;

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }


  bool OutputOptimizationResult(std::string output_file_name) {

    std::ofstream output_file(output_file_name);

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

  // experiment parameters
  double imu_dt_;

  Eigen::Matrix4d T_bc_;
  double fu_;   // focal length u
  double fv_;
  double cu_;   // image center u
  double cv_;   // image center v

  // parameter containers
  std::vector<State*>                state_parameter_;
  std::vector<Vec3dParameterBlock*>  landmark_parameter_;

  std::vector<ObservationData>       observation_data_;

  double sigma_g_c_;   // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;   // accelerometer noise density [m/s^2/sqrt(Hz)]

  Eigen::Vector3d bias_gyr_;
  Eigen::Vector3d bias_acc_;

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

  std::string observation_file_path = "feature_observation.csv";
  slam_problem.ReadObservationData(observation_file_path);

  slam_problem.ProcessGroundTruth(ground_truth_file_path);

  std::string imu_file_path = euroc_dataset_path + "imu0/data.csv";
  slam_problem.ReadIMUData(imu_file_path);
  slam_problem.OutputOptimizationResult("trajectory_dr.csv");

  slam_problem.Triangulate();

  slam_problem.SolveOptimizationProblem();
  slam_problem.OutputOptimizationResult("trajectory.csv");

  return 0;
}