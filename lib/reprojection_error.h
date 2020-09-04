/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Jan 4, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file ReprojectionError.h
 * @brief Header file for the ReprojectionError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_REPROJECTION_ERROR_H_
#define INCLUDE_REPROJECTION_ERROR_H_

#include <ceres/ceres.h>

/// \brief okvis Main namespace of this package.
// namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
// namespace ceres {

/// \brief Reprojection error base class.
class ReprojectionError:
    public ceres::SizedCostFunction<2,     // number of residuals
        3,                         // number of parameters in p_t
        4,                         // number of parameters in q_t
        3> {                       // number of landmark
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base class type.
  typedef ceres::SizedCostFunction<2, 3, 4, 3> base_t;

  /// \brief Number of residuals (2)
  static const int kNumResiduals = 2;

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d measurement_t;

  /// \brief Default constructor.
  ReprojectionError();

  /// \brief Construct with measurement and information matrix
  /// @param[in] measurement The measurement.
  /// @param[in] information The information (weight) matrix.
  ReprojectionError(const measurement_t & measurement, double focal, double* principle_point) {
    setMeasurement(measurement);

    focal_ = focal;
    principle_point_[0] = principle_point[0];
    principle_point_[1] = principle_point[1];
  }

  /// \brief Trivial destructor.
  ~ReprojectionError() {
  }

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  void setMeasurement(const measurement_t& measurement) {
    measurement_ = measurement;
  }


  /// \brief Get the measurement.
  /// \return The measurement vector.
  const measurement_t& measurement() const {
    return measurement_;
  }


  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  bool Evaluate(double const* const * parameters, 
                double* residuals,
                double** jacobians) const {

    Eigen::Vector3d position(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond rotation(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]);
    Eigen::Vector3d landmark(parameters[2][0], parameters[2][1], parameters[2][2]);

    Eigen::Vector3d rotated_pos = rotation * (landmark - position);
    
    residuals[0] = focal_ * (- rotated_pos[0] / rotated_pos[2]) + principle_point_[0] - measurement_(0);
    residuals[1] = focal_ * (- rotated_pos[1] / rotated_pos[2]) + principle_point_[1] - measurement_(1);


    /*********************************************************************************

                 Jacobian

    *********************************************************************************/


    if (jacobians != NULL) {
      
      // chain rule
      Eigen::MatrixXd J_residual_to_p(2,3);
      J_residual_to_p(0,0) = focal_ * (-1.0) / position(2);
      J_residual_to_p(0,1) = 0;
      J_residual_to_p(0,2) = focal_ * position(0) / (position(2)*position(2));
      J_residual_to_p(1,0) = 0;
      J_residual_to_p(1,1) = focal_ * (-1.0) / position(2);
      J_residual_to_p(1,2) = focal_ * position(1) / (position(2)*position(2));

      // position
      if (jacobians[0] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J0(jacobians[0]);       
        J0 = J_residual_to_p;
      }  

      // rotation
      if (jacobians[1] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J1(jacobians[1]);      

        Eigen::MatrixXd J_p_to_q(3,4);
        J_p_to_q(0,0) = landmark(0)*( 2)*rotation.w()+landmark(1)*(-2)*rotation.z()+landmark(2)*( 2)*rotation.y();
        J_p_to_q(0,1) = landmark(0)*( 2)*rotation.x()+landmark(1)*( 2)*rotation.y()+landmark(2)*( 2)*rotation.z();
        J_p_to_q(0,2) = landmark(0)*(-2)*rotation.y()+landmark(1)*( 2)*rotation.x()+landmark(2)*( 2)*rotation.w();
        J_p_to_q(0,3) = landmark(0)*(-2)*rotation.z()+landmark(1)*(-2)*rotation.w()+landmark(2)*( 2)*rotation.x();

        J_p_to_q(1,0) = landmark(0)*( 2)*rotation.z()+landmark(1)*( 2)*rotation.w()+landmark(2)*(-2)*rotation.x();
        J_p_to_q(1,1) = landmark(0)*( 2)*rotation.y()+landmark(1)*(-2)*rotation.x()+landmark(2)*(-2)*rotation.w();
        J_p_to_q(1,2) = landmark(0)*( 2)*rotation.x()+landmark(1)*( 2)*rotation.y()+landmark(2)*( 2)*rotation.z();
        J_p_to_q(1,3) = landmark(0)*( 2)*rotation.w()+landmark(1)*(-2)*rotation.z()+landmark(2)*( 2)*rotation.y();

        J_p_to_q(2,0) = landmark(0)*(-2)*rotation.y()+landmark(1)*( 2)*rotation.x()+landmark(2)*( 2)*rotation.w();
        J_p_to_q(2,1) = landmark(0)*( 2)*rotation.z()+landmark(1)*( 2)*rotation.w()+landmark(2)*(-2)*rotation.x();
        J_p_to_q(2,2) = landmark(0)*(-2)*rotation.w()+landmark(1)*( 2)*rotation.z()+landmark(2)*(-2)*rotation.y();
        J_p_to_q(2,3) = landmark(0)*( 2)*rotation.x()+landmark(1)*( 2)*rotation.y()+landmark(2)*( 2)*rotation.z();

        J1 = J_residual_to_p * J_p_to_q;

      }  

      // landmark
      if (jacobians[2] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J2(jacobians[2]);     
        J2 = J_residual_to_p * rotation.toRotationMatrix();
      }  

    }

    return true;

  }


  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const
  {
    return kNumResiduals;
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const
  {
    return "ReprojectionError";
  }

 protected:

  // the measurement
  measurement_t measurement_; ///< The (2D) measurement.

  double focal_;
  double principle_point_[2];


 
};

// }

// }

#endif /* INCLUDE_REPROJECTION_ERROR_H_ */
