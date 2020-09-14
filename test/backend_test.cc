// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// This test file is modifiled from the CERES examples


// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: keir@google.com (Keir Mierle)
//
// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "landmark_parameter_block.h"
#include "timed_3d_parameter_block.h"
#include "timed_quat_parameter_block.h"

/***
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const quat,
                  const T* const translation,
                  const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::QuaternionRotatePoint(quat, point, p);

    // camera[3,4,5] are the translation.
    p[0] += translation[0];
    p[1] += translation[1];
    p[2] += translation[2];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[1];
    const T& l2 = camera[2];
    T r2 = xp*xp + yp*yp;
    T distortion = 1.0 + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const T& focal = camera[0];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 4, 3, 3, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};
***/

class ReprojectionError: public ceres::SizedCostFunction<
    2,  // number of residuals
    4,  // size of rotation parameter
    3,  // size of translation parameter
    3,  // size of camera extrinsic parameter
    3  // size of landmark position
    > {
 public:

  ReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  bool Evaluate(double const* const* parameters,
                double* residuals,
                double** jacobians) const {

    double const* _rotation = parameters[0];
    double const* _translation = parameters[1];
    double const* _camera_ext = parameters[2];
    double const* _landmark = parameters[3];

    double p[3];
    ceres::QuaternionRotatePoint(_rotation, _landmark, p);

    p[0] += _translation[0];
    p[1] += _translation[1];
    p[2] += _translation[2];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    double xp = - p[0] / p[2];
    double yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const double& l1 = _camera_ext[1];
    const double& l2 = _camera_ext[2];
    double r2 = xp*xp + yp*yp;
    double distortion = 1.0 + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const double& focal = _camera_ext[0];
    double predicted_x = focal * distortion * xp;
    double predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;


    // Jacobian Calculations
    if (jacobians != NULL) {
      
      // chain rule
      Eigen::MatrixXd J_residual_to_p(2,3);
      J_residual_to_p(0,0) = focal * distortion * (-1.0) / p[2];
      J_residual_to_p(0,1) = 0;
      J_residual_to_p(0,2) = focal * distortion * p[0] / (p[2]*p[2]);
      J_residual_to_p(1,0) = 0;
      J_residual_to_p(1,1) = focal * distortion * (-1.0) / p[2];
      J_residual_to_p(1,2) = focal * distortion * p[1] / (p[2]*p[2]);

      // rotation
      if (jacobians[0] != NULL) {
        // hallucinate Jacobian w.r.t. state
        Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J0(jacobians[0]);      

        Eigen::MatrixXd J_p_to_q(3,4);
        J_p_to_q(0,0) = _landmark[0]*( 2)*_rotation[0]+_landmark[1]*(-2)*_rotation[3]+_landmark[2]*( 2)*_rotation[2];
        J_p_to_q(0,1) = _landmark[0]*( 2)*_rotation[1]+_landmark[1]*( 2)*_rotation[2]+_landmark[2]*( 2)*_rotation[3];
        J_p_to_q(0,2) = _landmark[0]*(-2)*_rotation[2]+_landmark[1]*( 2)*_rotation[1]+_landmark[2]*( 2)*_rotation[0];
        J_p_to_q(0,3) = _landmark[0]*(-2)*_rotation[3]+_landmark[1]*(-2)*_rotation[0]+_landmark[2]*( 2)*_rotation[1];

        J_p_to_q(1,0) = _landmark[0]*( 2)*_rotation[3]+_landmark[1]*( 2)*_rotation[0]+_landmark[2]*(-2)*_rotation[1];
        J_p_to_q(1,1) = _landmark[0]*( 2)*_rotation[2]+_landmark[1]*(-2)*_rotation[1]+_landmark[2]*(-2)*_rotation[0];
        J_p_to_q(1,2) = _landmark[0]*( 2)*_rotation[1]+_landmark[1]*( 2)*_rotation[2]+_landmark[2]*( 2)*_rotation[3];
        J_p_to_q(1,3) = _landmark[0]*( 2)*_rotation[0]+_landmark[1]*(-2)*_rotation[3]+_landmark[2]*( 2)*_rotation[2];

        J_p_to_q(2,0) = _landmark[0]*(-2)*_rotation[2]+_landmark[1]*( 2)*_rotation[1]+_landmark[2]*( 2)*_rotation[0];
        J_p_to_q(2,1) = _landmark[0]*( 2)*_rotation[3]+_landmark[1]*( 2)*_rotation[0]+_landmark[2]*(-2)*_rotation[1];
        J_p_to_q(2,2) = _landmark[0]*(-2)*_rotation[0]+_landmark[1]*( 2)*_rotation[3]+_landmark[2]*(-2)*_rotation[2];
        J_p_to_q(2,3) = _landmark[0]*( 2)*_rotation[1]+_landmark[1]*( 2)*_rotation[2]+_landmark[2]*( 2)*_rotation[3];

        J0 = J_residual_to_p * J_p_to_q;

      }  

      // translation
      if (jacobians[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J1(jacobians[1]);       

        J1 = J_residual_to_p;
      }

      // camera extrinsic
      if (jacobians[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J2(jacobians[2]);       

        J2(0,0) = distortion * xp;
        J2(0,1) = focal * r2 * xp;
        J2(0,2) = focal * r2 * r2 * xp;

        J2(1,0) = distortion * yp;
        J2(1,1) = focal * r2 * yp;
        J2(1,2) = focal * r2 * r2 * yp;
      }  

      // landmark
      if (jacobians[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J3(jacobians[3]);     

        J3 = J_residual_to_p * Eigen::Quaterniond(_rotation[0], _rotation[1], _rotation[2], _rotation[3]).toRotationMatrix();
      }      
    }

    return true;


  }

 private:
  double observed_x;
  double observed_y;
};

// Read a Bundle Adjustment in the Large dataset.
class BALProblem {

 public:

  BALProblem() {
    initalized_ = false;
  }

  ~BALProblem() {
  }

  bool loadFile(const char* filename) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == NULL) {
      return false;
    };

    FscanfOrDie(fptr, "%d", &num_poses_);     
    FscanfOrDie(fptr, "%d", &num_landmarks_);   
    FscanfOrDie(fptr, "%d", &num_observations_);

    // declare all parameter blocks
    for (int i=0; i < num_poses_; ++i) {
      rotation_parameter_.push_back(TimedQuatParameterBlock(Eigen::Quaterniond(), i));
      translation_parameter_.push_back(Timed3dParameterBlock(Eigen::Vector3d(), i));
      camera_parameter_.push_back(new double[3]);
    }

    for (int i=0; i < num_landmarks_; ++i) {
      landmark_parameter_.push_back(LandmarkParameterBlock(Eigen::Vector3d(), true));
    }


    // read observation data
    int _pose_id;
    int _landmark_id;
    double _observation[2];

    for (int i = 0; i < num_observations_; ++i) {

      FscanfOrDie(fptr, "%d", &_pose_id);
      FscanfOrDie(fptr, "%d", &_landmark_id);

      for (int j = 0; j < 2; ++j)
        FscanfOrDie(fptr, "%lf", _observation + j);

      /***
      ceres::CostFunction* cost_function =
          SnavelyReprojectionError::Create(_observation[0],
                                           _observation[1]);
      ***/

      ceres::CostFunction* cost_function = new ReprojectionError(_observation[0], _observation[1]);

      optimization_problem_.AddResidualBlock(cost_function,
                               NULL,
                               rotation_parameter_.at(_pose_id).parameters(),
                               translation_parameter_.at(_pose_id).parameters(),
                               camera_parameter_.at(_pose_id), 
                               landmark_parameter_.at(_landmark_id).parameters());
    }


    // parameter initialization
    double _temp[3];
    for (int i=0; i < num_poses_; ++i) {

      for (int j=0; j<3; ++j)  
        FscanfOrDie(fptr, "%lf", _temp + j);

      double quat_arr[4];
      ceres::AngleAxisToQuaternion(_temp, quat_arr);
      rotation_parameter_.at(i).setEstimate(Eigen::Quaterniond(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3]));


      for (int j=0; j<3; ++j)  
        FscanfOrDie(fptr, "%lf", _temp + j);

      translation_parameter_.at(i).setEstimate(Eigen::Vector3d(_temp[0], _temp[1], _temp[2]));

      for (int j=0; j<3; ++j)
        FscanfOrDie(fptr, "%lf", camera_parameter_.at(i) + j);

    }

    for (int i=0; i < num_landmarks_; ++i) {
      for (int j=0; j<3; ++j)  
        FscanfOrDie(fptr, "%lf", _temp + j);

      landmark_parameter_.at(i).setEstimate(Eigen::Vector3d(_temp[0], _temp[1], _temp[2]));
    }


    // assume perfectly initialized
    optimization_problem_.SetParameterBlockConstant(rotation_parameter_.at(0).parameters());
    optimization_problem_.SetParameterBlockConstant(translation_parameter_.at(0).parameters());

    initalized_ = true;

    return true;
  }

  bool solveOptimizationProblem() {
    if (!initalized_)
      return false;

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.

    optimization_options_.linear_solver_type = ceres::DENSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }

 private:
  template<typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  bool initalized_;

  int num_poses_; 
  int num_landmarks_; 
  int num_observations_;

  ceres::Problem optimization_problem_;
  ceres::Solver::Options optimization_options_;
  ceres::Solver::Summary optimization_summary_;

  std::vector<TimedQuatParameterBlock> rotation_parameter_;
  std::vector<Timed3dParameterBlock> translation_parameter_;
  std::vector<double*> camera_parameter_;
  std::vector<LandmarkParameterBlock> landmark_parameter_;
};


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }

  BALProblem bal_problem;

  if (!bal_problem.loadFile(argv[1])) {
    std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
    return 1;
  }

  bal_problem.solveOptimizationProblem();

  return 0;
}
