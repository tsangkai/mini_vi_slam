
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

  cv::Mat K() {
    
    cv::Mat ret_K(3, 3, CV_64F, cv::Scalar(0));
    
    ret_K.at<double>(0,0) = fu_;
    ret_K.at<double>(1,1) = fv_;
    ret_K.at<double>(0,2) = pu_;
    ret_K.at<double>(1,2) = pv_;
    ret_K.at<double>(2,2) = 1;

    return ret_K;
  }

  cv::Mat GetDistortionCoeff() {
    cv::Mat distortion_coeff(4, 1, CV_64F, cv::Scalar(0));

    distortion_coeff.at<double>(0) = k1_;
    distortion_coeff.at<double>(1) = k2_;
    distortion_coeff.at<double>(2) = p1_;
    distortion_coeff.at<double>(3) = p2_;

    return distortion_coeff;
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


  cv::Point2f UnScaleAndShift(cv::Point2f point) {
    cv::Point2f return_point;

    return_point.x = (point.x - pu_) / fu_;
    return_point.y = (point.y - pv_) / fv_;

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

    std::ofstream output_file;
    std::string output_file_str("epi_init.csv");
    output_file.open(output_file_str);
    output_file << "timestamp [ns], R, t\n";
    output_file << image_data_.at(0).GetTimestamp() << std::endl;

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

    

        
        // TEST: initialization from 8-point algorithm
        if (j == i+1) {
          std::cout << "matches between " << i << " and " << j << " = " << image_keypoint_matches.size() << std::endl;

          size_t point_count = image_keypoint_matches.size();
          std::vector<cv::Point2d> points_1(point_count);
          std::vector<cv::Point2d> points_2(point_count);

          for (size_t k=0; k<point_count; ++k) {
            cv::Mat_<cv::Point2d> distorted_points(1,2);
            distorted_points(0) = image_keypoints_.at(i)[image_keypoint_matches[k].queryIdx].pt;
            distorted_points(1) = image_keypoints_.at(j)[image_keypoint_matches[k].trainIdx].pt;

            cv::Mat un_distorted_points;

            cv::undistortPoints(distorted_points, un_distorted_points, camera_ptr->K(), camera_ptr->GetDistortionCoeff(), cv::noArray(), cv::noArray());

            points_1[k].x = un_distorted_points.at<double>(0,0);
            points_1[k].y = un_distorted_points.at<double>(0,1);
            points_2[k].x = un_distorted_points.at<double>(1,0);
            points_2[k].y = un_distorted_points.at<double>(1,1);

          }

          cv::Mat essential_mat = cv::findEssentialMat(points_1, points_2, cv::Mat::eye(3,3, CV_64F), cv::RANSAC, 0.99, 0.9);


          cv::Mat R1, R2, t;
          DecomposeE(essential_mat, R1, R2, t);
          cv::Mat t1 = t;
          cv::Mat t2 = -t;

          int nGood1 = CheckRT(R1, t1, points_1, points_2);
          int nGood2 = CheckRT(R1, t2, points_1, points_2);
          int nGood3 = CheckRT(R2, t1, points_1, points_2);
          int nGood4 = CheckRT(R2, t2, points_1, points_2);

          std::cout << nGood1 << "\t" << nGood2 << "\t" << nGood3 << "\t" << nGood4 << "\t" << std::endl;

          int max_nGood = std::max(nGood1, std::max(nGood2, std::max(nGood3, nGood4)));

          cv::Mat min_R, min_t;
          if(nGood1 == max_nGood) {
            min_R = R1;
            min_t = t1;
          }
          else if (nGood2 == max_nGood) {
            min_R = R1;
            min_t = t2;
          }
          else if (nGood3 == max_nGood) {
            min_R = R2;
            min_t = t1;
          }
          else {
            min_R = R2;
            min_t = t2;
          }

          // std::cout << min_R << std::endl;
          // std::cout << min_t << std::endl;


          output_file << image_data_.at(j).GetTimestamp() << ",";
          output_file << min_R.at<double>(0,0) << "," << min_R.at<double>(0,1) << "," << min_R.at<double>(0,2) << ","
                      << min_R.at<double>(1,0) << "," << min_R.at<double>(1,1) << "," << min_R.at<double>(1,2) << ","
                      << min_R.at<double>(2,0) << "," << min_R.at<double>(2,1) << "," << min_R.at<double>(2,2) << ",";
          
          output_file << min_t.at<double>(0) << "," << min_t.at<double>(1) << "," << min_t.at<double>(2) << "\n";


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
    
    output_file.close();


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
int CheckRT(const cv::Mat &R, const cv::Mat &t, 
                                   const std::vector<cv::Point2d> &vKeys1, 
                                   const std::vector<cv::Point2d> &vKeys2)
{
    double parallax;
    double th2 = 4.0;

    // Calibration parameters
    // const float fx = K.at<float>(0,0);
    // const float fy = K.at<float>(1,1);
    // const float cx = K.at<float>(0,2);
    // const float cy = K.at<float>(1,2);

    // vbGood = vector<bool>(vKeys1.size(),false);
    std::vector<cv::Point3d> vP3D;
    vP3D.resize(vKeys1.size());

    std::vector<double> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_64F,cv::Scalar(0));
    // K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_64F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_64F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    // P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i=0, iend=vKeys1.size();i<iend;i++)
    {

      const cv::Point2d &kp1 = vKeys1[i];
      const cv::Point2d &kp2 = vKeys2[i];

        // const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        // const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        Triangulate(kp1,kp2,P1,P2,p3dC1);

        if(!isfinite(p3dC1.at<double>(0)) || !isfinite(p3dC1.at<double>(1)) || !isfinite(p3dC1.at<double>(2)))
        {
            std::cout << "infinite number" << std::endl;
            //vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        double dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        double dist2 = cv::norm(normal2);

        double cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if(p3dC1.at<double>(2)<=0 && cosParallax<0.99998) {
          continue;

        }

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<double>(2)<=0 && cosParallax<0.99998) {
          continue;

        }

        // Check reprojection error in first image
        double im1x, im1y;
        double invZ1 = 1.0/p3dC1.at<double>(2);
        im1x = p3dC1.at<double>(0)*invZ1;
        im1y = p3dC1.at<double>(1)*invZ1;

        double squareError1 = (im1x-kp1.x)*(im1x-kp1.x)+(im1y-kp1.y)*(im1y-kp1.y);

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image
        double im2x, im2y;
        double invZ2 = 1.0/p3dC2.at<double>(2);
        im2x = p3dC2.at<double>(0)*invZ2;
        im2y = p3dC2.at<double>(1)*invZ2;

        double squareError2 = (im2x-kp2.x)*(im2x-kp2.x)+(im2y-kp2.y)*(im2y-kp2.y);

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);
        // vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        //if(cosParallax<0.99998)
        //    vbGood[vMatches12[i].first]=true;
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = std::min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}
 
 
  // from ORB SLAM
  void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t) {
    cv::Mat u,w,vt;
    cv::SVD::compute(E, w, u, vt);

    u.col(2).copyTo(t);
    
    t = t/cv::norm(t);

    cv::Mat W(3, 3, CV_64F, cv::Scalar(0));
    W.at<double>(0,1) = -1;
    W.at<double>(1,0) =  1;
    W.at<double>(2,2) =  1;

    R1 = u * W * vt;
    if(cv::determinant(R1) < 0)
      R1 = -R1;

    R2 = u * W.t() * vt;
    if(cv::determinant(R2) < 0)
      R2 = -R2;
  }

  void Triangulate(const cv::Point2d &kp1, const cv::Point2d &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D) {
    cv::Mat A(4,4,CV_64F);

    A.row(0) = kp1.x * P1.row(2) - P1.row(0);
    A.row(1) = kp1.y * P1.row(2) - P1.row(1);
    A.row(2) = kp2.x * P2.row(2) - P2.row(0);
    A.row(3) = kp2.y * P2.row(2) - P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3) / x3D.at<double>(3);

  }



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
