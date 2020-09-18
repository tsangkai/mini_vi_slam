// Since we do not consider the entire trajectory at the first place,
// this file helps to determine the start and the end of the trajectory

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <opencv2/core/core.hpp>


bool ProcessGroundTruth(std::string ground_truth_file_path, std::string begin_time, std::string end_time) {

  std::cout << "Read ground truth data at " << ground_truth_file_path << std::endl;
  std::cout << "Between time " << begin_time << " and " << end_time << std::endl;

  std::ifstream input_file(ground_truth_file_path);
  if(!input_file.is_open()) 
    throw std::runtime_error("Could not open file");

  // Read the column names
  // Extract the first line in the file
  std::string line;
  std::getline(input_file, line);

  std::ofstream gt_file("ground_truth.csv");

  gt_file << "timestamp,p_x,p_y,p_z,in_interval\n";

  size_t total_count = 0;

  while (std::getline(input_file, line)) {

    ++ total_count;

    if (total_count % 15 == 0) {                           // downsample images for testing

      std::stringstream s_stream(line);                    // Create a stringstream of the current line

      if (s_stream.good()) {
        std::string time_stamp_str;
        std::getline(s_stream, time_stamp_str, ',');       // get first string delimited by comma
              
        std::string data;
        gt_file << time_stamp_str;

        for (int i=0; i<3; ++i) {                    
          std::getline(s_stream, data, ',');

          gt_file << ",";
          gt_file << data;
        } 
            
        if (time_stamp_str >= begin_time && time_stamp_str < end_time) {
          gt_file << "," << "1";
        }
        else {
          gt_file << "," << "0";
        }

        gt_file << std::endl;

      }
    }
  }

  input_file.close();
  gt_file.close();

  return true;
}  


int main(int argc, char **argv) {

  std::string config_folder_path = "../config/";
  cv::FileStorage test_config_file(config_folder_path + "test.yaml", cv::FileStorage::READ);

  std::string euroc_dataset_path = "../../../dataset/mav0/";
  std::string ground_truth_file_path = euroc_dataset_path + "state_groundtruth_estimate0/data.csv";
  ProcessGroundTruth(ground_truth_file_path, test_config_file["time_window"][0], test_config_file["time_window"][1]);

  return 0;
}