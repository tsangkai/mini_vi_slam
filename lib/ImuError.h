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
 *  Created on: Sep 3, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file ImuError.hpp
 * @brief Header file for the ImuError class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_IMU_ERROR_H_
#define INCLUDE_IMU_ERROR_H_

#include <vector>
// #include <mutex>
// #include "ceres/ceres.h"
// #include <okvis/FrameTypedefs.hpp>
// #include <okvis/Time.hpp>
// #include <okvis/assert_macros.hpp>
// #include <okvis/Measurements.hpp>
// #include <okvis/Variables.hpp>
// #include <okvis/Parameters.hpp>
// #include <okvis/ceres/ErrorInterface.hpp>

/// \brief okvis Main namespace of this package.
// namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
// namespace ceres {

/// \brief Implements a nonlinear IMU factor.
class ImuError :
    public ceres::SizedCostFunction<10,     // number of residuals
        3,                         // number of parameters in p_{t+1}
        3,                         // number of parameters in v_{t+1}
        4,                         // number of parameters in q_{t+1}
        3,                         // number of parameters in p_t
        3,                         // number of parameters in v_t
        4> {                       // number of parameters in q_t
        // 3,                         // number of parameters of gyro bias
        // 3> {                        // number of parameters of accel bias
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base in ceres we derive from
  // typedef ceres::SizedCostFunction<10, 3, 3, 4, 3, 3, 4, 3, 3> base_t;
  typedef ceres::SizedCostFunction<10, 3, 3, 4, 3, 3, 4> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 10;

  /// \brief The type of the covariance.
  // typedef Eigen::Matrix<double, 15, 15> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  // typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  // typedef Eigen::Matrix<double, 15, 15> jacobian_t;

  /// \brief The type of the Jacobian w.r.t. poses --
  /// \warning This is w.r.t. minimal tangential space coordinates...
  // typedef Eigen::Matrix<double, 15, 7> jacobian0_t;

  /// \brief The type of Jacobian w.r.t. Speed and biases
  // typedef Eigen::Matrix<double, 15, 9> jacobian1_t;

  /// \brief Default constructor -- assumes information recomputation.
  ImuError() {
  }

  /// \brief Trivial destructor.
  ~ImuError() {
  }

  /// 
  ImuError(const Eigen::Vector3d gyro_measurement,
           const Eigen::Vector3d accel_measurement,
           const double dt) {
    gyro_measurement_ = gyro_measurement;
    accel_measurement_ = accel_measurement;
    dt_ = dt;
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

    Eigen::Vector3d _position_t1(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Vector3d _velocity_t1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond _rotation_t1(parameters[2][0], parameters[2][1], parameters[2][2], parameters[2][3]);
    Eigen::Vector3d _position_t(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Vector3d _velocity_t(parameters[4][0], parameters[4][1], parameters[4][2]);
    Eigen::Quaterniond _rotation_t(parameters[5][0], parameters[5][1], parameters[5][2], parameters[5][3]);

    Eigen::Map<Eigen::Vector3d > _r_position(residuals+0);      
    Eigen::Map<Eigen::Vector3d > _r_velocity(residuals+3);      
    Eigen::Map<Eigen::Quaterniond > _r_rotation(residuals+6);      

    Eigen::Vector3d _accel_plus_gravity = _rotation_t.normalized().toRotationMatrix()*accel_measurement_ + Eigen::Vector3d(0, 0, -9.81007);
    _r_position = _position_t1 - ( _position_t +  dt_*_velocity_t + (dt_*dt_*0.5)* _accel_plus_gravity);
    _r_velocity = _velocity_t1 - (                    _velocity_t +           dt_* _accel_plus_gravity);
    _r_rotation = _rotation_t1 * ( _rotation_t * Eigen::Quaterniond(1, 0.5*dt_*gyro_measurement_(0), 0.5*dt_*gyro_measurement_(1), 0.5*dt_*gyro_measurement_(2))).inverse();
    


    /*********************************************************************************

                 Jacobian

    *********************************************************************************/

    if (jacobians != NULL) {
      
      Eigen::MatrixXd J_p_to_q(3,4);
      J_p_to_q(0,0) = accel_measurement_(0)*( 2)*_rotation_t.w()+accel_measurement_(1)*(-2)*_rotation_t.z()+accel_measurement_(2)*( 2)*_rotation_t.y();
      J_p_to_q(0,1) = accel_measurement_(0)*( 2)*_rotation_t.x()+accel_measurement_(1)*( 2)*_rotation_t.y()+accel_measurement_(2)*( 2)*_rotation_t.z();
      J_p_to_q(0,2) = accel_measurement_(0)*(-2)*_rotation_t.y()+accel_measurement_(1)*( 2)*_rotation_t.x()+accel_measurement_(2)*( 2)*_rotation_t.w();
      J_p_to_q(0,3) = accel_measurement_(0)*(-2)*_rotation_t.z()+accel_measurement_(1)*(-2)*_rotation_t.w()+accel_measurement_(2)*( 2)*_rotation_t.x();

      J_p_to_q(1,0) = accel_measurement_(0)*( 2)*_rotation_t.z()+accel_measurement_(1)*( 2)*_rotation_t.w()+accel_measurement_(2)*(-2)*_rotation_t.x();
      J_p_to_q(1,1) = accel_measurement_(0)*( 2)*_rotation_t.y()+accel_measurement_(1)*(-2)*_rotation_t.x()+accel_measurement_(2)*(-2)*_rotation_t.w();
      J_p_to_q(1,2) = accel_measurement_(0)*( 2)*_rotation_t.x()+accel_measurement_(1)*( 2)*_rotation_t.y()+accel_measurement_(2)*( 2)*_rotation_t.z();
      J_p_to_q(1,3) = accel_measurement_(0)*( 2)*_rotation_t.w()+accel_measurement_(1)*(-2)*_rotation_t.z()+accel_measurement_(2)*( 2)*_rotation_t.y();

      J_p_to_q(2,0) = accel_measurement_(0)*(-2)*_rotation_t.y()+accel_measurement_(1)*( 2)*_rotation_t.x()+accel_measurement_(2)*( 2)*_rotation_t.w();
      J_p_to_q(2,1) = accel_measurement_(0)*( 2)*_rotation_t.z()+accel_measurement_(1)*( 2)*_rotation_t.w()+accel_measurement_(2)*(-2)*_rotation_t.x();
      J_p_to_q(2,2) = accel_measurement_(0)*(-2)*_rotation_t.w()+accel_measurement_(1)*( 2)*_rotation_t.z()+accel_measurement_(2)*(-2)*_rotation_t.y();
      J_p_to_q(2,3) = accel_measurement_(0)*( 2)*_rotation_t.x()+accel_measurement_(1)*( 2)*_rotation_t.y()+accel_measurement_(2)*( 2)*_rotation_t.z();


      // position_t1
      if (jacobians[0] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor> > J_p_t1(jacobians[0]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<3; ++j) {
            J_p_t1(i,j) = 0.0;
          }
        }

        J_p_t1(0,0) = 1.0;
        J_p_t1(1,1) = 1.0;
        J_p_t1(2,2) = 1.0;

      }  

      // velocity_t1
      if (jacobians[1] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor> > J_v_t1(jacobians[1]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<3; ++j) {
            J_v_t1(i,j) = 0.0;
          }
        }

        J_v_t1(3,0) = 1.0;
        J_v_t1(4,1) = 1.0;
        J_v_t1(5,2) = 1.0;
      }  

      // rotation_t1
      if (jacobians[2] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 4, Eigen::RowMajor> > J_q_t1(jacobians[2]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<4; ++j) {
            J_q_t1(i,j) = 0.0;
          }
        }

        J_q_t1(6,0) = 1.0;
        J_q_t1(7,1) = 1.0;
        J_q_t1(8,2) = 1.0;
        J_q_t1(9,3) = 1.0;
      }  

      // position_t
      if (jacobians[3] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor> > J_p_t(jacobians[3]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<3; ++j) {
            J_p_t(i,j) = 0.0;
          }
        }

        J_p_t(0,0) = -1.0;
        J_p_t(1,1) = -1.0;
        J_p_t(2,2) = -1.0;
      }  

      // velocity_t
      if (jacobians[4] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor> > J_v_t(jacobians[4]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<3; ++j) {
            J_v_t(i,j) = 0.0;
          }
        }

        J_v_t(1,0) = -dt_;
        J_v_t(2,1) = -dt_;
        J_v_t(3,2) = -dt_;

        J_v_t(3,0) = -1.0;
        J_v_t(4,1) = -1.0;
        J_v_t(5,2) = -1.0;
      }  

      // rotation_t
      if (jacobians[5] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 10, 4, Eigen::RowMajor> > J_q_t(jacobians[5]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<4; ++j) {
            J_q_t(i,j) = 0.0;
          }
        }

        J_q_t(6,0) = -1.0;
        J_q_t(7,1) = -1.0;
        J_q_t(8,2) = -1.0;
        J_q_t(9,3) = -1.0;        

        for (size_t i=0; i<3; ++i) {
          for (size_t j=0; j<4; ++j) {
            J_q_t(i,j) = -(0.5*dt_*dt_) * J_p_to_q(i,j);
            J_q_t(3+i,j) = -(dt_) * J_p_to_q(i,j);
          }
        }

      }  



    }



    return 1;
  }


  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Return parameter block type as string
  std::string typeInfo() const {
    return "ImuError";
  }

 protected:

  // measurements
  Eigen::Vector3d gyro_measurement_;
  Eigen::Vector3d accel_measurement_; 

  // times
  double dt_;

};

// }  // namespace ceres
// }  // namespace okvis

#endif /* INCLUDE_IMU_ERROR_H_ */
