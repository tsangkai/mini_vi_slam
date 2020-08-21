
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <brisk/brisk.h>


// parameters
// the following parameters allow us to start with small-scope project
#define DOWNSAMPLE_RATE    10
#define TIME_WINDOW_BEGIN  "1403636649313555456"
#define TIME_WINDOW_END    "1403636658963555584"

#define BRISK_DETECTION_THRESHOLD             50.0
#define BRISK_DETECTION_OCTAVES               0
#define BRISK_DETECTION_ABSOLUTE_THRESHOLD    800.0
#define BRISK_DETECTION_MAX_KEYPOINTS         450

#define BRISK_DESCRIPTION_ROTATION_INVARIANCE true
#define BRISK_DESCRIPTION_SCALE_INVARIANCE    false



class CameraData {
 public:
  CameraData(std::string timeStampStr, std::string dataFilePath) {
    time_ = timeStampStr;
    image_ = cv::imread(dataFilePath,cv::IMREAD_GRAYSCALE);
  }

  std::string getTime() { return time_; }
  cv::Mat getImage() { return image_; }

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

  /*** Step 1. Reading image files ***/

  // the folder path
  // std::string path(argv[1]);
  std::string path("../../../dataset/mav0/");
  std::string camera_data_folder("cam0/data/");

  std::vector<std::string> image_names;

  // boost allows us to work on image files directly
  for (auto iter = boost::filesystem::directory_iterator(path + camera_data_folder);
        iter != boost::filesystem::directory_iterator(); iter++) {

    if (!boost::filesystem::is_directory(iter->path())) {          //we eliminate directories
      image_names.push_back(iter->path().filename().string());
    } 
    else
      continue;
  }

  std::sort(image_names.begin(), image_names.end());

  size_t downsample_rate = DOWNSAMPLE_RATE;
  std::string time_window_begin = TIME_WINDOW_BEGIN;
  std::string time_window_end = TIME_WINDOW_END;

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

  size_t num_of_observations = camera_observation_data.size();


  /*** Step 2. Extracting features ***/

  std::shared_ptr<cv::FeatureDetector> brisk_detector = 
    std::shared_ptr<cv::FeatureDetector>(
      new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(
        BRISK_DETECTION_THRESHOLD, 
        BRISK_DETECTION_OCTAVES, 
        BRISK_DETECTION_ABSOLUTE_THRESHOLD, 
        BRISK_DETECTION_MAX_KEYPOINTS));

  std::shared_ptr<cv::FeatureDetector> brisk_extractor = 
    std::shared_ptr<cv::DescriptorExtractor>(
      new brisk::BriskDescriptorExtractor(
        BRISK_DESCRIPTION_ROTATION_INVARIANCE,
        BRISK_DESCRIPTION_SCALE_INVARIANCE));

  // you can try to use ORB feature as well
  // std::shared_ptr<cv::FeatureDetector> orb_detector = cv::ORB::create();

  std::vector<std::vector<cv::KeyPoint>> image_keypoints(num_of_observations);
  std::vector<cv::Mat> image_descriptions(num_of_observations);

  for (size_t i=0; i<num_of_observations; i++) {	

    brisk_detector->detect(camera_observation_data.at(i).getImage(), image_keypoints.at(i));

    // orb_detector->detect(camera_observation_data.at(i).getImage(), image_keypoints.at(i));
    // cv::drawKeypoints( , );

    brisk_extractor->compute(camera_observation_data.at(i).getImage(), 
      image_keypoints.at(i), 
      image_descriptions.at(i));
  }

  /*** Step 3. Matching features ***/

  std::shared_ptr<cv::DescriptorMatcher> matcher = 
    std::shared_ptr<cv::DescriptorMatcher>(
      new cv::BFMatcher(cv::NORM_HAMMING));   // normType goes with the descriptor, BRISK should use NORM_HAMMING

  std::vector<std::vector<cv::DMatch>> image_matches(num_of_observations-1);
  std::vector<std::vector<cv::DMatch>> image_good_matches(num_of_observations-1);

  for (size_t i=0; i<num_of_observations-1; i++) {

    matcher->match(image_descriptions.at(i), image_descriptions.at(i+1), image_matches.at(i));

    cv::Mat img_w_matches;
    for (size_t k=0; k<image_matches.at(i).size(); k++) {
      if (image_matches.at(i)[k].distance < 60) {   // 60
        image_good_matches.at(i).push_back(image_matches.at(i)[k]);
      }
    }

    cv::drawMatches(camera_observation_data.at(i).getImage(), image_keypoints.at(i),
                    camera_observation_data.at(i+1).getImage(), image_keypoints.at(i+1),
                    image_good_matches.at(i), img_w_matches);

    cv::imshow("Matches between " + std::to_string(i) + " and " + std::to_string(i+1), img_w_matches);
    cv::waitKey();
  }


  /*** Step 4. Output observation ***/

  std::ofstream output_file;
  output_file.open ("observation.txt");
  output_file << "timestamp [ns], landmark id, u [pixel], v [pixel]\n";

  std::map<CVKeypoint, size_t> pre_landmark_lookup_table;       // keypoint and landmark id
  std::map<CVKeypoint, size_t> next_landmark_lookup_table;

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
    std::string output_str = camera_observation_data.at(i).getTime() + ", " + std::to_string(landmark_id+1) + ", "
                              + std::to_string(pre_keypoint.getU()) + ", " + std::to_string(pre_keypoint.getV()) + "\n";
    output_file << output_str;

      next_landmark_lookup_table.insert(std::pair<CVKeypoint, size_t>(next_keypoint, landmark_id));
    }

    std::swap(pre_landmark_lookup_table, next_landmark_lookup_table);
    next_landmark_lookup_table.clear();
  }

  output_file.close();

  std::getchar();
  return 0;
}
