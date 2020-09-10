
// This test file verifies reprojection_error.h

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "landmark_parameter_block.h"
#include "timed_3d_parameter_block.h"
#include "timed_quat_parameter_block.h"
#include "reprojection_error.h"


class TriangularizationProblem {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  TriangularizationProblem() {
    focal_length_ = 1.0;
    principal_point_[0] = 0;
    principal_point_[1] = 0;
  }

  bool Initialize() {

    // state 0
    position_parameter_.push_back(new Timed3dParameterBlock(Eigen::Vector3d(1,1,1), 0, 0.0));
    rotation_parameter_.push_back(new TimedQuatParameterBlock(Eigen::Quaterniond::UnitRandom(), 0, 0.0));

    // state 1
    position_parameter_.push_back(new Timed3dParameterBlock(Eigen::Vector3d(-1,-1,-1), 1, 0.1));
    rotation_parameter_.push_back(new TimedQuatParameterBlock(Eigen::Quaterniond::UnitRandom(), 1, 0.1));

    // one landmark
    landmark_parameter_.push_back(new LandmarkParameterBlock(Eigen::Vector3d(23.1,22.1,23.1), 0));
    landmark.setEstimate(Eigen::Vector3d(13,13,13));

    return true;
  }

  bool SetupOptimizationProblem() {

    // add parameter blocks
    for (size_t i=0; i<2; ++i) {
      optimization_problem_.AddParameterBlock(position_parameter_.at(i)->parameters(), 3);
      optimization_problem_.SetParameterBlockConstant(position_parameter_.at(i)->parameters());

      optimization_problem_.AddParameterBlock(rotation_parameter_.at(i)->parameters(), 4);
      optimization_problem_.SetParameterBlockConstant(rotation_parameter_.at(i)->parameters());
    }

    // observation
    Eigen::Vector2d observation_error;
    observation_error(0,0) = -0.1;
    observation_error(1,0) = 0.2;


    // add constraints
    for (size_t i=0; i<2; ++i) {
      Eigen::Vector2d observation;

      Eigen::Vector3d temp_observation = rotation_parameter_.at(i)->estimate().inverse().toRotationMatrix()*(landmark.estimate() - position_parameter_.at(i)->estimate());
      observation(0) = -focal_length_*temp_observation(0)/temp_observation(2) + principal_point_[0];
      observation(1) = -focal_length_*temp_observation(1)/temp_observation(2) + principal_point_[1];

      ceres::CostFunction* cost_function = new ReprojectionError(observation,
                                                                 Eigen::Transform<double, 3, Eigen::Affine>::Identity(),
                                                                 focal_length_,
                                                                 principal_point_);

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             position_parameter_.at(i)->parameters(),
                                             rotation_parameter_.at(i)->parameters(),
                                             landmark_parameter_.at(0)->parameters()); 
    }

    return true;
  }

  bool SolveOptimizationProblem() {

    std::cout << "Begin solving the optimization problem." << std::endl;

    optimization_options_.linear_solver_type = ceres::DENSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }


  bool OutputOptimizationResult() {

    std::cout << "The true landmark position: " << landmark.estimate() <<  std::endl;
    std::cout << "The estimated landmark position: " << landmark_parameter_.at(0)->estimate() <<  std::endl;
    
    return true;
  }

 private:

  // camera intrinsic parameters
  double focal_length_;
  double principal_point_[2];

  // data storage (parameters to be optimized)
  std::vector<TimedQuatParameterBlock*> rotation_parameter_;
  std::vector<Timed3dParameterBlock*>  position_parameter_;
  std::vector<LandmarkParameterBlock*> landmark_parameter_;
  LandmarkParameterBlock landmark;

  // ceres parameter
  ceres::Problem optimization_problem_;
  ceres::Solver::Options optimization_options_;
  ceres::Solver::Summary optimization_summary_;
};


int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  TriangularizationProblem triangularization_problem;
  triangularization_problem.Initialize();
  triangularization_problem.SetupOptimizationProblem();
  triangularization_problem.SolveOptimizationProblem();
  triangularization_problem.OutputOptimizationResult();

  return 0;
}