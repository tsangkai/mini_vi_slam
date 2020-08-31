
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>


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


 private:


  double time_begin_;
  double time_end_;

  // camera parameters
  Eigen::Matrix4d T_BC_; // from camera frame to body frame
  double focal_length_;
  double principal_point_[2];

  // data storage

  // ceres parameter


};



int main(int argc, char **argv) {

  /*** Step 0. Configuration files ***/

  ExpLandmarkOptSLAM slam_problem;

  std::string config_folder_path("../config/");
  slam_problem.readConfigurationFiles(config_folder_path);


  /*** Step 1. Datasets ***/


  /*** Step 2. Construct state ***/



  /*** Step 3. Convert to the optmization problem ***/

  // ceres

  return 0;
}