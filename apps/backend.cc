
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <Eigen/Core>


#define TIME_WINDOW_BEGIN  "1403636649313555456"
#define TIME_WINDOW_END    "1403636658963555584"




size_t read_csv(std::string filename){

  size_t line_count = 0;

  // Create an input filestream
  std::ifstream input_file(filename);

  // Make sure the file is open
  if(!input_file.is_open()) 
    throw std::runtime_error("Could not open file");

  // Read the column names
  // Extract the first line in the file
  std::string line;
  std::getline(input_file, line);

  // Read data, line by line
  while(std::getline(input_file, line)) {
    // Create a stringstream of the current line
    std::stringstream s_stream(line);

    if(s_stream.good()) {
      std::string time_stamp_str;
      std::getline(s_stream, time_stamp_str, ','); //get first string delimited by comma
      
      if(TIME_WINDOW_BEGIN <= time_stamp_str && time_stamp_str <= TIME_WINDOW_END) {
        ++line_count;
      }
    }
  }

  // Close file
  input_file.close();

  return line_count;
}

int main(int argc, char **argv) {

  /*** Step 1. Read files ***/

  std::string ground_truth_path("../../../dataset/mav0/state_groundtruth_estimate0/data.csv");
  size_t line_count_ground_truth = read_csv(ground_truth_path);
  
  std::cout << "ground truth: " << line_count_ground_truth << std::endl;


  std::string imu_path("../../../dataset/mav0/imu0/data.csv");
  size_t line_count_imu = read_csv(imu_path);
  
  std::cout << "imu: " << line_count_imu << std::endl;


  std::string observation_path("feature_observation.csv");
  size_t line_count_observation = read_csv(observation_path);
  
  std::cout << "feature observation: " << line_count_observation << std::endl;


  // for yaml file
  std::string camera_calib_path("../config/config_fpga_p2_euroc.yaml");
  cv::FileStorage camera_calib_file(camera_calib_path, cv::FileStorage::READ);

  cv::FileNode T_BS_ = camera_calib_file["cameras"][0]["T_SC"];  // from camera frame to body frame

  Eigen::Matrix4d T_BS_e;
  T_BS_e << T_BS_[0], T_BS_[1], T_BS_[2], T_BS_[3], T_BS_[4], T_BS_[5], T_BS_[6], T_BS_[7], T_BS_[8], T_BS_[9], T_BS_[10], T_BS_[11], T_BS_[12], T_BS_[13], T_BS_[14], T_BS_[15];

  std::cout << T_BS_e << std::endl;

  /*** Step 2. Construct state ***/



  /*** Step 3. Convert to the optmization problem ***/

  // ceres

  return 0;
}