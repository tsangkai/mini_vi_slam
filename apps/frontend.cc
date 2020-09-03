
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


class CameraData {
 public:
  CameraData(std::string timestamp_str, std::string data_file_path) {
    time_ = timestamp_str;
    image_ = cv::imread(data_file_path, cv::IMREAD_GRAYSCALE);
  }

  std::string getTime() { 
    return time_; 
  }
  
  cv::Mat getImage() { 
    return image_; 
  }

 private:
  std::string time_;   // we don't have to process time at this moment
  cv::Mat image_;
};

// This class wraps cv::KeyPoint in order to use std::map
class CVKeypoint {
 public:
 	CVKeypoint(cv::KeyPoint keypoint) {
 		keypoint_ = keypoint;
 		hash_value_ = keypoint.hash();
 	}

  float getU() {
  	return keypoint_.pt.x;
  }

  float getV() {
  	return keypoint_.pt.y;
  }

 	bool operator==(const CVKeypoint& kp) const{
    return hash_value_ == kp.hash_value_;
 	}

 	bool operator<(const CVKeypoint& kp) const{
    return hash_value_ < kp.hash_value_;
 	}

 private:
  cv::KeyPoint keypoint_;
  size_t hash_value_;
};

int main(int argc, char **argv) {

  /*** Step 0. Read configuration file ***/

  // for yaml file
  std::string config_file_path("../config/test.yaml");
  cv::FileStorage config_file(config_file_path, cv::FileStorage::READ);

  std::string time_window_begin(config_file["time_window"][0]);
  std::string time_window_end(config_file["time_window"][1]);

  size_t downsample_rate = (size_t)(int)(config_file["frontend"]["downsample_rate"]);

  std::cout << "Consider from " << time_window_begin << " to " << time_window_end << ": " << std::endl;

  /*** Step 1. Read image files ***/

  // the folder path
  // std::string path(argv[1]);
  std::string path("../../../dataset/mav0/");
  std::string camera_data_folder("cam0/data/");

  std::vector<std::string> image_names;

  // boost allows us to work on image files directly
  for (auto iter = boost::filesystem::directory_iterator(path + camera_data_folder);
        iter != boost::filesystem::directory_iterator(); iter++) {

    if (!boost::filesystem::is_directory(iter->path())) {          // we eliminate directories
      image_names.push_back(iter->path().filename().string());
    } 
    else
      continue;
  }

  std::sort(image_names.begin(), image_names.end());

  std::vector<CameraData> camera_observation_data;   // image and timestep

  size_t counter = 0;
  for (auto& image_names_iter: image_names) {	
  
    if (counter % downsample_rate == 0) {            // downsample images for testing
      std::string time_stamp_str = image_names_iter.substr(0,19);  // remove ".png"

      if(time_window_begin <= time_stamp_str && time_stamp_str <= time_window_end) {
        std::string dataFilePath = path + camera_data_folder + image_names_iter;
        camera_observation_data.push_back(CameraData(time_stamp_str, dataFilePath));

        // cv::imshow(time_stamp_str, camera_observation_data.back().getImage());
        // cv::waitKey(100);
      }
    }

    counter++;
  }

  size_t num_of_cam_observations = camera_observation_data.size();


  /*** Step 2. Extract features ***/

  std::shared_ptr<cv::FeatureDetector> brisk_detector =
    cv::BRISK::create(60, 0, 1.0f);


  // you can try to use ORB feature as well
  // std::shared_ptr<cv::FeatureDetector> orb_detector = cv::ORB::create();

  std::vector<std::vector<cv::KeyPoint>> image_keypoints(num_of_cam_observations);
  std::vector<cv::Mat> image_descriptions(num_of_cam_observations);

  for (size_t i=0; i<num_of_cam_observations; i++) {	

    brisk_detector->detect(camera_observation_data.at(i).getImage(), image_keypoints.at(i));

    brisk_detector->compute(camera_observation_data.at(i).getImage(), 
      image_keypoints.at(i), 
      image_descriptions.at(i));
  }

  /*** Step 3. Match features ***/

  std::shared_ptr<cv::DescriptorMatcher> matcher = 
    cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

  std::vector<std::vector<cv::DMatch>> image_matches(num_of_cam_observations-1);
  std::vector<std::vector<cv::DMatch>> image_good_matches(num_of_cam_observations-1);

  for (size_t i=0; i<num_of_cam_observations-1; i++) {

    matcher->match(image_descriptions.at(i), image_descriptions.at(i+1), image_matches.at(i));

    cv::Mat img_w_matches;
    for (size_t k=0; k<image_matches.at(i).size(); k++) {
      if (image_matches.at(i)[k].distance < 60) {
        image_good_matches.at(i).push_back(image_matches.at(i)[k]);
      }
    }

    cv::drawMatches(camera_observation_data.at(i).getImage(), image_keypoints.at(i),
                    camera_observation_data.at(i+1).getImage(), image_keypoints.at(i+1),
                    image_good_matches.at(i), img_w_matches);

    cv::imshow("Matches between " + std::to_string(i) + " and " + std::to_string(i+1), img_w_matches);
    cv::waitKey();
  }


  /*** Step 4. Obtain feature observation ***/

  std::map<CVKeypoint, size_t> pre_landmark_lookup_table;       // keypoint and landmark id
  std::map<CVKeypoint, size_t> next_landmark_lookup_table;

  std::vector<std::string> output_feature_observation;

  size_t landmakr_id_count = 0;
  size_t landmark_id = 0;

  for (size_t i=0; i<image_good_matches.size(); i++) {
    for (size_t m=0; m<image_good_matches.at(i).size(); m++) {
      
      size_t pre_keypoint_id = image_good_matches.at(i)[m].queryIdx;
      size_t next_keypoint_id = image_good_matches.at(i)[m].trainIdx;

      CVKeypoint pre_keypoint = CVKeypoint(image_keypoints.at(i)[pre_keypoint_id]);
      CVKeypoint next_keypoint = CVKeypoint(image_keypoints.at(i+1)[next_keypoint_id]);

      auto iterr = pre_landmark_lookup_table.find(pre_keypoint);
      if (iterr == pre_landmark_lookup_table.end()) {

        landmark_id = landmakr_id_count;

        pre_landmark_lookup_table.insert(std::pair<CVKeypoint, size_t>(pre_keypoint, landmark_id));
        ++landmakr_id_count;
      }
      else {
        landmark_id = iterr->second;
      }      	

    // output
    // timestamp [ns], landmark id, u [pixel], v [pixel]
    std::string output_str = camera_observation_data.at(i).getTime() + "," + std::to_string(landmark_id+1) + ","
                              + std::to_string(pre_keypoint.getU()) + "," + std::to_string(pre_keypoint.getV()) + "\n";
    output_feature_observation.push_back(output_str);
    // output_file << output_str;

      next_landmark_lookup_table.insert(std::pair<CVKeypoint, size_t>(next_keypoint, landmark_id));
    }

    std::swap(pre_landmark_lookup_table, next_landmark_lookup_table);
    next_landmark_lookup_table.clear();
  }


  /*** Step 5. Output observation ***/

  std::ofstream output_file;
  output_file.open ("feature_observation.csv");
  output_file << "timestamp [ns], landmark id, u [pixel], v [pixel]\n";
  
  for (auto& output_str: output_feature_observation) { 
    // timestamp [ns], landmark id, u [pixel], v [pixel]
    output_file << output_str;
  }
  
  output_file.close();
  return 0;
}
