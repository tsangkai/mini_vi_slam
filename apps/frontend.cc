
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/LU>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

class Camera {

 public:
  Camera(double fu, double fv, double pu, double pv,
         double dis_para_0, double dis_para_1, double dis_para_2, double dis_para_3) {
    fu_ = fu;
    fv_ = fv;

    pu_ = pu;
    pv_ = pv;

    k1_ = dis_para_0;
    k2_ = dis_para_1;
    p1_ = dis_para_2;
    p2_ = dis_para_3;
  }

  Eigen::Vector2d ScaleAndShift(Eigen::Vector2d point) {
    Eigen::Vector2d return_point;

    return_point[0] = fu_ * point[0] + pu_;
    return_point[1] = fv_ * point[1] + pv_;

    return return_point;
  }

  Eigen::Vector2d UnScaleAndShift(Eigen::Vector2d point) {
    Eigen::Vector2d return_point;

    return_point[0] = (point[0] - pu_) / fu_;
    return_point[1] = (point[1] - pv_) / fv_;

    return return_point;
  }

  bool Distort(const Eigen::Vector2d & pointUndistorted, Eigen::Vector2d * pointDistorted,
    Eigen::Matrix2d * pointJacobian) const {

    // first compute the distorted point
    const double u0 = pointUndistorted[0];
    const double u1 = pointUndistorted[1];
    const double mx_u = u0 * u0;
    const double my_u = u1 * u1;
    const double mxy_u = u0 * u1;
    const double rho_u = mx_u + my_u;
    const double rad_dist_u = k1_ * rho_u + k2_ * rho_u * rho_u;
    (*pointDistorted)[0] = u0 + u0 * rad_dist_u + 2.0 * p1_ * mxy_u
      + p2_ * (rho_u + 2.0 * mx_u);
    (*pointDistorted)[1] = u1 + u1 * rad_dist_u + 2.0 * p2_ * mxy_u
      + p1_ * (rho_u + 2.0 * my_u);

    // next the Jacobian w.r.t. changes on the undistorted point
    Eigen::Matrix2d & J = *pointJacobian;
    J(0, 0) = 1 + rad_dist_u + k1_ * 2.0 * mx_u + k2_ * rho_u * 4 * mx_u
      + 2.0 * p1_ * u1 + 6 * p2_ * u0;
    J(1, 0) = k1_ * 2.0 * u0 * u1 + k2_ * 4 * rho_u * u0 * u1 + p1_ * 2.0 * u0
      + 2.0 * p2_ * u1;
    J(0, 1) = J(1, 0);
    J(1, 1) = 1 + rad_dist_u + k1_ * 2.0 * my_u + k2_ * rho_u * 4 * my_u
      + 6 * p1_ * u1 + 2.0 * p2_ * u0;


    return true;
  
  }

  bool UnDistort(const Eigen::Vector2d & pointDistorted,
                     Eigen::Vector2d * pointUndistorted) const {

    // this is expensive: we solve with Gauss-Newton...
    Eigen::Vector2d x_bar = pointDistorted; // initialise at distorted point
    const int n = 15;  // just 5 iterations max.
    Eigen::Matrix2d E;  // error Jacobian

    bool success = false;
    for (int i = 0; i < n; i++) {

      Eigen::Vector2d x_tmp;

      Distort(x_bar, &x_tmp, &E);

      Eigen::Vector2d e(pointDistorted - x_tmp);
      Eigen::Matrix2d E2 = (E.transpose() * E);
      Eigen::Vector2d du = E2.inverse() * E.transpose() * e;

      x_bar += du;

      const double chi2 = e.dot(e);
      if (chi2 < 1e-4) {
        success = true;
      }
      if (chi2 < 1e-15) {
        success = true;
        break;
      }

    }
    *pointUndistorted = x_bar;

    if(!success){
      std::cout<<(E.transpose() * E)<<std::endl;
    }

    return success;
  }

 private:
  double fu_;
  double fv_;
  double pu_;
  double pv_;

  double k1_;
  double k2_;
  double p1_;
  double p2_;

};

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

class FeatureNode {
 public: 
  FeatureNode() {
    landmark_id_ = 0;
  }

  bool AddNeighbor(FeatureNode* feature_node_ptr) {
    neighbors_.push_back(feature_node_ptr);
    return true;
  }

  bool IsNeighborEmpty() {
    return neighbors_.empty();
  }

  size_t GetLandmarkId() {
    return landmark_id_;
  }

  void SetLandmarkId(const size_t new_landmark_id) {
    landmark_id_ = new_landmark_id;
  }  

  bool AssignLandmarkId(const size_t input_landmark_id) {
    if (input_landmark_id == 0) {
      std::cout << "invalid landmark id." << std::endl;
      return false;
    }
    else if (input_landmark_id == landmark_id_) {
      return true;
    }    
    else if (input_landmark_id != landmark_id_ && landmark_id_ == 0) {
      landmark_id_ = input_landmark_id;
      
      for (auto neighbor_ptr: neighbors_) {
        neighbor_ptr->AssignLandmarkId(input_landmark_id);
      }

      return true;
    }
    else {   // input_landmark_id != landmark_id_ && landmark_id_ != 0
      std::cout << "The same landmark is assigned 2 different id." << std::endl;
      return false;
    }
  }

  bool ResetLandmarkId() {
    if (landmark_id_ == 0) {
      return true;
    }
    else {
      landmark_id_ = 0;

      for (auto neighbor_ptr: neighbors_) {
        neighbor_ptr->ResetLandmarkId();
      }
      return true;
    }
  }  

 private:
  size_t landmark_id_;
  std::vector<FeatureNode *> neighbors_;   
};

class Frontend {

 public:
  Frontend(std::string config_folder_path) {
  
    std::string config_file_path = config_folder_path + "test.yaml";
    cv::FileStorage test_config_file(config_file_path, cv::FileStorage::READ);

    time_window_begin_ = std::string(test_config_file["time_window"][0]);
    time_window_end_ = std::string(test_config_file["time_window"][1]);
    downsample_rate_ = (size_t)(int)(test_config_file["frontend"]["downsample_rate"]);

    std::cout << "Consider from " << time_window_begin_ << " to " << time_window_end_ << ": " << std::endl;


    cv::FileStorage experiment_config_file(config_folder_path + "config_fpga_p2_euroc.yaml", cv::FileStorage::READ);

    camera_ptr = new Camera((double) experiment_config_file["cameras"][0]["focal_length"][0], 
                            (double) experiment_config_file["cameras"][0]["focal_length"][1],
                            (double) experiment_config_file["cameras"][0]["principal_point"][0],
                            (double) experiment_config_file["cameras"][0]["principal_point"][1],
                            (double) experiment_config_file["cameras"][0]["distortion_coefficients"][0],
                            (double) experiment_config_file["cameras"][0]["distortion_coefficients"][1],
                            (double) experiment_config_file["cameras"][0]["distortion_coefficients"][2],
                            (double) experiment_config_file["cameras"][0]["distortion_coefficients"][3]);

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

    landmark_id_table_.resize(num_of_images);
    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        landmark_id_table_.at(i).push_back(new FeatureNode());
      }
    }


    // call opencv matcher
    for (size_t i=0; i<num_of_images; i++) {
      for (size_t j=i+1; j<num_of_images; j++) {

        std::vector<cv::DMatch> image_keypoint_temp_matches;
        std::vector<cv::DMatch> image_keypoint_matches;
  
        matcher->match(image_descriptions_.at(i), image_descriptions_.at(j), image_keypoint_temp_matches);

        // keep the matches that have smaller distance
        for (size_t k=0; k<image_keypoint_temp_matches.size(); k++) {
          if (image_keypoint_temp_matches[k].distance < 40) {   // 60

            image_keypoint_matches.push_back(image_keypoint_temp_matches[k]);

            // add edge to the graph
            size_t query_idx = image_keypoint_temp_matches[k].queryIdx;
            size_t train_idx = image_keypoint_temp_matches[k].trainIdx;

            landmark_id_table_.at(i).at(query_idx)->AddNeighbor(landmark_id_table_.at(j).at(train_idx));
            landmark_id_table_.at(j).at(train_idx)->AddNeighbor(landmark_id_table_.at(i).at(query_idx));
          }
        }  

        /***
        cv::Mat img_w_matches;
        cv::drawMatches(image_data_.at(i).GetImage(), image_keypoints_.at(i),
                      image_data_.at(j).GetImage(), image_keypoints_.at(j),
                      image_keypoint_matches, img_w_matches);

        cv::imshow("Matches between " + std::to_string(i) + " and " + std::to_string(j), img_w_matches);
        cv::waitKey();

        cv::imwrite("extraction.jpg", img_w_keypoints);

        ***/
      }
    }


    // assign landmark id to each matched features
    size_t landmark_count = 0;

    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        if (!landmark_id_table_.at(i).at(k)->IsNeighborEmpty() && landmark_id_table_.at(i).at(k)->GetLandmarkId()==0) {
          landmark_count++;
          landmark_id_table_.at(i).at(k)->AssignLandmarkId(landmark_count);
        }
      }
    }

    // remove landmark id that appears twice in a single image
    for (size_t i=0; i<num_of_images; i++) {
      std::vector<size_t> landmark_count_vec(landmark_count, 0);

      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        size_t landmark_id_temp = landmark_id_table_.at(i).at(k)->GetLandmarkId();
        if (landmark_id_temp > 0) {
          landmark_count_vec.at(landmark_id_temp-1)++;
        }
      }

      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        size_t landmark_id_temp = landmark_id_table_.at(i).at(k)->GetLandmarkId();
        if (landmark_id_temp > 0 && landmark_count_vec.at(landmark_id_temp-1) > 1) {

          landmark_id_table_.at(i).at(k)->ResetLandmarkId();
          landmark_count_vec.at(landmark_id_temp-1) = 0;
        }
      }
    }

    // assign new id to those landmarks
    std::vector<size_t> landmark_id_2_id_table(landmark_count, 0);
    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        
        size_t temp_landmark_id = landmark_id_table_.at(i).at(k)->GetLandmarkId();
        if (temp_landmark_id > 0) {
          landmark_id_2_id_table.at(temp_landmark_id-1) = 1;
        }
      }
    }

    size_t landmark_count_after_threshold = 0;
    for (size_t i=0; i<landmark_count; i++) {
      if (landmark_id_2_id_table.at(i) > 0) {
        landmark_count_after_threshold++;
        landmark_id_2_id_table.at(i) = landmark_count_after_threshold;
      }
    }    

    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {

        size_t temp_landmark_id = landmark_id_table_.at(i).at(k)->GetLandmarkId();
        if (temp_landmark_id > 0) {
          landmark_id_table_.at(i).at(k)->SetLandmarkId(landmark_id_2_id_table.at(temp_landmark_id-1));
        }
      }
    }


    // CHECK: assign the same landmark id to different feature points in a single image
    for (size_t i=0; i<num_of_images; i++) {
      std::vector<size_t> landmark_count_vec(landmark_count, 0);

      for (size_t k=0; k<image_keypoints_.at(i).size(); k++) {
        size_t landmark_id_temp = landmark_id_table_.at(i).at(k)->GetLandmarkId();
        if (landmark_id_temp > 0) {
          landmark_count_vec.at(landmark_id_temp-1)++;
        }
      }

      for (size_t j=0; j<landmark_count; ++j) {
        if (landmark_count_vec.at(j) > 1) {
          std::cout << "Image " << i << ": " << "landmark id " << j+1 << " is observed " << landmark_count_vec.at(j) << "times" << std::endl;
        }
      }
    }


    std::cout << "total landmark counts: " << landmark_count << std::endl;
    std::cout << "total landmark counts after thresholding observed number: " << landmark_count_after_threshold << std::endl;

    return true;
  }


  bool OutputFeatureObservation(std::string output_file_str) {

    std::ofstream output_file;
    output_file.open(output_file_str);
    output_file << "timestamp [ns], landmark id, u [pixel], v [pixel], size\n";
  
    
    size_t num_of_images = image_data_.size();

    for (size_t i=0; i<num_of_images; i++) {
      for (size_t k=0; k<image_keypoints_.at(i).size(); ++k) {

        if (landmark_id_table_.at(i).at(k)->GetLandmarkId()!=0) {

          // 

          Eigen::Vector2d distorted_point(image_keypoints_.at(i).at(k).pt.x, image_keypoints_.at(i).at(k).pt.y);
          Eigen::Vector2d undistorted_point;
          camera_ptr->UnDistort(camera_ptr->UnScaleAndShift(distorted_point), &undistorted_point);
          Eigen::Vector2d corrected_point = camera_ptr->ScaleAndShift(undistorted_point);

          // std::cout << distorted_point << std::endl;
          // std::cout << corrected_point << std::endl << std::endl;

          // 


          /***
          std::string output_str = image_data_.at(i).GetTimestamp() + "," 
                                   + std::to_string(landmark_id_table_.at(i).at(k)->GetLandmarkId()) + ","
                                   + std::to_string(image_keypoints_.at(i).at(k).pt.x) + ","
                                   + std::to_string(image_keypoints_.at(i).at(k).pt.y) + ","
                                   + std::to_string(image_keypoints_.at(i).at(k).size) + "\n";
          ***/

          std::string output_str = image_data_.at(i).GetTimestamp() + "," 
                                   + std::to_string(landmark_id_table_.at(i).at(k)->GetLandmarkId()) + ","
                                   + std::to_string(corrected_point[0]) + ","
                                   + std::to_string(corrected_point[1]) + ","
                                   + std::to_string(image_keypoints_.at(i).at(k).size) + "\n";

          output_file << output_str;
        }
      }
    }
  
    output_file.close();

    return true;
  }

 private: 
  std::string time_window_begin_;
  std::string time_window_end_;
  size_t downsample_rate_;

  Camera* camera_ptr;

  std::vector<std::string>                  image_names_;
  std::vector<TimedImageData>               image_data_;       

  std::vector<std::vector<cv::KeyPoint>>    image_keypoints_;
  std::vector<cv::Mat>                      image_descriptions_;           

  std::vector<std::vector<FeatureNode*>>    landmark_id_table_;       // [image id][keypoint id]
};

int main(int argc, char **argv) {

  /*** Step 0. Read configuration file ***/

  std::string config_folder_path("../config/");
  Frontend frontend(config_folder_path);                              // read configuration file


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


  /*** Step 4. Output observation ***/
  std::string output_file_str("feature_observation.csv");
  frontend.OutputFeatureObservation(output_file_str);


  return 0;
}
