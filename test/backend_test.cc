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

#include "SizedParameterBlock.h"
#include "LandmarkParameterBlock.h"
#include "Timed3dParameterBlock.h"
#include "TimedQuatParameterBlock.h"




// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
 public:
  ~BALProblem() {
    delete[] landmark_index_;
    delete[] pose_index_;
    delete[] observations_;
    delete[] parameters_;
  }

  int num_observations()       const  { return num_observations_;               }
  int num_landmarks()          const  { return num_landmarks_;                  }
  int num_poses()              const  { return num_poses_;                      }
  const double* observations() const  { return observations_;                   }
  double* getParamterPtr()            { return parameters_;                     }

  int pose_id_for_observation(int i)     { return pose_index_[i];               }
  int landmark_id_for_observation(int i) { return landmark_index_[i];           }

  bool LoadFile(const char* filename) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == NULL) {
      return false;
    };

    FscanfOrDie(fptr, "%d", &num_poses_);        // poses
    FscanfOrDie(fptr, "%d", &num_landmarks_);    // landmarks
    FscanfOrDie(fptr, "%d", &num_observations_);

    landmark_index_ = new int[num_observations_];
    pose_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_poses_ + 3 * num_landmarks_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", pose_index_ + i);
      FscanfOrDie(fptr, "%d", landmark_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
      }
    }

    // parameter initialization
    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }

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

  int num_poses_;      // poses
  int num_landmarks_;       // landmarks
  int num_observations_;
  int num_parameters_;   

  int* landmark_index_;
  int* pose_index_;
  double* observations_;
  double* parameters_;

};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
// functors!
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


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }

  BALProblem bal_problem;
  if (!bal_problem.LoadFile(argv[1])) {
    std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
    return 1;
  }


  /*** Parameters to be estimated ***/
  std::vector<TimedQuatParameterBlock> rotation_parameter;
  std::vector<Timed3dParameterBlock> translation_parameter;
  std::vector<double*> camera_parameter_ptr;
  std::vector<LandmarkParameterBlock> landmark_parameter;


  for (int i=0; i < bal_problem.num_poses(); ++i) {
    double* parameter_ptr = bal_problem.getParamterPtr() + 9*i;

    // rotation
    double quat_arr[4];
    ceres::AngleAxisToQuaternion(parameter_ptr, quat_arr);
    Eigen::Quaterniond rotation_init(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[4]);
    rotation_parameter.push_back(TimedQuatParameterBlock(rotation_init, i, i));

    // translation
    Eigen::Vector3d translation_init(parameter_ptr[3], parameter_ptr[4], parameter_ptr[5]);
    translation_parameter.push_back(Timed3dParameterBlock(translation_init, i, i));

    // camera extrinsic parameter
    double camera_ext_init[3];

    for (int j=0; j < 3; ++j) {
       camera_ext_init[j] = parameter_ptr[6+j];
    }
    camera_parameter_ptr.push_back(camera_ext_init);
  }


  for (int i=0; i < bal_problem.num_landmarks(); ++i) {
    double* landmark_ptr = bal_problem.getParamterPtr() + 9 * bal_problem.num_poses() + 3*i;

    Eigen::Vector3d landmark_init(landmark_ptr[0], landmark_ptr[1], landmark_ptr[2]);
    landmark_parameter.push_back(LandmarkParameterBlock(landmark_init, i, true));
  }

  const double* observations = bal_problem.observations();



  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.

    ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observations[2*i+0],
                                         observations[2*i+1]);


    int pose_id = bal_problem.pose_id_for_observation(i);
    int landmark_id = bal_problem.landmark_id_for_observation(i);
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             rotation_parameter.at(pose_id).parameters(),
                             translation_parameter.at(pose_id).parameters(),
                             camera_parameter_ptr.at(pose_id), 
                             landmark_parameter.at(landmark_id).parameters());
  }

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
  return 0;
}
