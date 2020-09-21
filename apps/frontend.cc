
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


class TimedImageData {
 public:
  TimedImageData(std::string timestamp_str, std::string data_file_path) {
    timestamp_ = timestamp_str;
    image_ = cv::imread(data_file_path, cv::IMREAD_GRAYSCALE);
  }

  std::string GetTimestamp() { 
    return timestamp_; 
  }
  
  cv::Mat GetImage() { 
    return image_; 
  }

 private:
  std::string timestamp_;   // we don't have to process time at this moment
  cv::Mat image_;
};

// This class wraps cv::KeyPoint in order to use std::map
class CVKeypoint {
 public:
 	CVKeypoint(cv::KeyPoint keypoint) {
 		keypoint_ = keypoint;
 		hash_value_ = keypoint.hash();
 	}

  float GetU() {
  	return keypoint_.pt.x;
  }

  float GetV() {
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

class Frontend {

 public:
  Frontend(std::string config_file_path) {
  
    cv::FileStorage config_file(config_file_path, cv::FileStorage::READ);

    time_window_begin_ = std::string(config_file["time_window"][0]);
    time_window_end_ = std::string(config_file["time_window"][1]);
    downsample_rate_ = (size_t)(int)(config_file["frontend"]["downsample_rate"]);

    std::cout << "Consider from " << time_window_begin_ << " to " << time_window_end_ << ": " << std::endl;
  }

  bool ReadImages(std::string image_folder_path) {

    // boost allows us to work on image files directly
    for (auto iter = boost::filesystem::directory_iterator(image_folder_path);
          iter != boost::filesystem::directory_iterator(); iter++) {

      if (!boost::filesystem::is_directory(iter->path())) {           // we eliminate directories
        image_names_.push_back(iter->path().filename().string());
      } 
      else
        continue;
    }

    std::sort(image_names_.begin(), image_names_.end());

    size_t counter = 0;
    size_t selected_counter = 0;

    for (auto& image_names_iter: image_names_) { 
    
      if (counter % downsample_rate_ == 0) {                           // downsample images for testing
        std::string time_stamp_str = image_names_iter.substr(0,19);    // remove ".png"

        if(time_window_begin_ <= time_stamp_str && time_stamp_str <= time_window_end_) {
          std::string image_file_path = image_folder_path + image_names_iter;
          image_data_.push_back(TimedImageData(time_stamp_str, image_file_path));

          // cv::imshow(time_stamp_str, image_data_.back().GetImage());
          // cv::waitKey(100);

          selected_counter++;
        }
      }

      counter++;
    }

    std::cout << "number of processed images: " << selected_counter << std::endl;

    // for visual debugging
    // loop closure observed
    /***
    for (size_t i=0; i<image_data_.size(); ++i) {
      if (i % 5 == 0) {
        cv::imshow(std::to_string(i) + " " + image_data_.at(i).GetTimestamp(), image_data_.at(i).GetImage());
      }
    }

    cv::waitKey();
    ***/

    return true;
  }

  bool ExtractFeatures(std::shared_ptr<cv::FeatureDetector> detector) {

    size_t num_of_images = image_data_.size();

    image_keypoints_.resize(num_of_images);
    image_descriptions_.resize(num_of_images);


    for (size_t i=0; i<num_of_images; i++) {  

      detector->detect(image_data_.at(i).GetImage(), image_keypoints_.at(i));

      detector->compute(image_data_.at(i).GetImage(), 
        image_keypoints_.at(i), 
        image_descriptions_.at(i));

        /***
        cv::Mat img_w_keypoints;
        cv::drawKeypoints(image_data_.at(i).GetImage(), image_keypoints_.at(i), img_w_keypoints);

        cv::imshow("image with keypoints " + std::to_string(i) + "/" + std::to_string(num_of_images) , img_w_keypoints);
        cv::waitKey();
        ***/
    }

    return true;
  }


  bool MatchFeatures(std::shared_ptr<cv::DescriptorMatcher> matcher) {

    size_t num_of_images = image_data_.size();

    

    std::vector<std::vector<cv::DMatch>> image_keypoint_temp_matches(num_of_images-1);
    image_keypoint_matches_.resize(num_of_images-1);

    for (size_t i=0; i<num_of_images-1; i++) {

      matcher->match(image_descriptions_.at(i), image_descriptions_.at(i+1), image_keypoint_temp_matches.at(i));

      for (size_t k=0; k<image_keypoint_temp_matches.at(i).size(); k++) {
        if (image_keypoint_temp_matches.at(i)[k].distance < 50) {
          image_keypoint_matches_.at(i).push_back(image_keypoint_temp_matches.at(i)[k]);
        }
      }

      /***
      cv::Mat img_w_matches;
      cv::drawMatches(image_data_.at(i).GetImage(), image_keypoints_.at(i),
                      image_data_.at(i+1).GetImage(), image_keypoints_.at(i+1),
                      image_keypoint_matches_.at(i), img_w_matches);

      cv::imshow("Matches between " + std::to_string(i) + " and " + std::to_string(i+1), img_w_matches);
      cv::waitKey();
      ***/
    }


    //===============================================//


    std::map<CVKeypoint, size_t> pre_landmark_lookup_table;       // keypoint and landmark id
    std::map<CVKeypoint, size_t> next_landmark_lookup_table;


    size_t landmakr_id_count = 0;
    size_t landmark_id = 0;

    for (size_t i=0; i<image_keypoint_matches_.size(); i++) {
      for (size_t m=0; m<image_keypoint_matches_.at(i).size(); m++) {
      
        size_t pre_keypoint_id = image_keypoint_matches_.at(i)[m].queryIdx;
        size_t next_keypoint_id = image_keypoint_matches_.at(i)[m].trainIdx;

        CVKeypoint pre_keypoint = CVKeypoint(image_keypoints_.at(i)[pre_keypoint_id]);
        CVKeypoint next_keypoint = CVKeypoint(image_keypoints_.at(i+1)[next_keypoint_id]);

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
        std::string output_str = image_data_.at(i).GetTimestamp() + "," + std::to_string(landmark_id+1) + ","
                              + std::to_string(pre_keypoint.GetU()) + "," + std::to_string(pre_keypoint.GetV()) + "\n";
        output_feature_observation_.push_back(output_str);
        // output_file << output_str;

        next_landmark_lookup_table.insert(std::pair<CVKeypoint, size_t>(next_keypoint, landmark_id));
      }

      std::swap(pre_landmark_lookup_table, next_landmark_lookup_table);
      next_landmark_lookup_table.clear();
    }

    //===============================================//

    return true;
  }


  bool OutputFeatureObservation(std::string output_file_str) {


    std::ofstream output_file;
    output_file.open(output_file_str);
    output_file << "timestamp [ns], landmark id, u [pixel], v [pixel]\n";
  
    for (auto& output_str: output_feature_observation_) { 
      // timestamp [ns], landmark id, u [pixel], v [pixel]
      output_file << output_str;
    }
  
    output_file.close();

    return true;
  }

 private: 
  std::string time_window_begin_;
  std::string time_window_end_;
  size_t downsample_rate_;

  std::vector<std::string> image_names_;
  std::vector<TimedImageData> image_data_;       

  std::vector<std::vector<cv::KeyPoint>> image_keypoints_;
  std::vector<cv::Mat> image_descriptions_;           

  std::vector<std::vector<cv::DMatch>> image_keypoint_matches_;

  std::vector<std::string> output_feature_observation_;
};

int main(int argc, char **argv) {

  /*** Step 0. Read configuration file ***/

  std::string config_file_path("../config/test.yaml");
  Frontend frontend(config_file_path);                     // read configuration file


  /*** Step 1. Read image files ***/

  // std::string path(argv[1]);
  std::string dataset_path("../../../dataset/mav0/");
  std::string camera_data_folder("cam0/data/");

  frontend.ReadImages(dataset_path + camera_data_folder);


  /*** Step 2. Extract features ***/
  std::shared_ptr<cv::FeatureDetector> brisk_detector = cv::BRISK::create(60, 0, 1.0f);
  std::shared_ptr<cv::FeatureDetector> orb_detector = cv::ORB::create();

  frontend.ExtractFeatures(brisk_detector);


  /*** Step 3. Match features ***/
  std::shared_ptr<cv::DescriptorMatcher> bf_hamming_matcher = 
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);  
  frontend.MatchFeatures(bf_hamming_matcher);


  /*** Step 4. Obtain feature observation ***/

  /*** Step 5. Output observation ***/

  std::string output_file_str("feature_observation.csv");
  frontend.OutputFeatureObservation(output_file_str);


  return 0;
}
