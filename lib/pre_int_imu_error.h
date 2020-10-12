

/**
 * @file pre_int_imu_error.hpp
 * @brief Header file for the ImuError class.
 * @author Tsang-Kai Chang
 */

// TODO(tsangkai): Implement bias parameters (interface, residual, and jacobian)

#ifndef INCLUDE_PRE_INT_IMU_ERROR_H_
#define INCLUDE_PRE_INT_IMU_ERROR_H_

#include <vector>

Eigen::Matrix3d Jacob_R_to_qw(Eigen::Quaterniond quat) {

  Eigen::Matrix3d J_R_to_qw;

  J_R_to_qw <<          0, -2*quat.z(),  2*quat.y(),
               2*quat.z(),           0, -2*quat.x(),
              -2*quat.y(),  2*quat.x(),           0;

  return J_R_to_qw;
}

Eigen::Matrix3d Jacob_R_to_qx(Eigen::Quaterniond quat) {

  Eigen::Matrix3d J_R_to_qx;

  J_R_to_qx <<          0,  2*quat.y(),  2*quat.z(),
               2*quat.y(), -4*quat.x(), -2*quat.w(),
               2*quat.z(),  2*quat.w(), -4*quat.x();

  return J_R_to_qx;
}

Eigen::Matrix3d Jacob_R_to_qy(Eigen::Quaterniond quat) {

  Eigen::Matrix3d J_R_to_qy;

  J_R_to_qy << -4*quat.y(), 2*quat.x(),  2*quat.w(),
                2*quat.x(),          0,  2*quat.z(),
               -2*quat.w(), 2*quat.z(), -4*quat.y();

  return J_R_to_qy;
}

Eigen::Matrix3d Jacob_R_to_qz(Eigen::Quaterniond quat) {

  Eigen::Matrix3d J_R_to_qz;

  J_R_to_qz << -4*quat.z(), -2*quat.w(), 2*quat.x(),
                2*quat.w(), -4*quat.z(), 2*quat.y(),
                2*quat.x(),  2*quat.y(),          0;

  return J_R_to_qz;
}

/// \brief Implements a nonlinear IMU factor.
class PreIntImuError :
    public ceres::SizedCostFunction<10,     // number of residuals
        4,                         // number of parameters in q_{t+1}
        3,                         // number of parameters in v_{t+1}
        3,                         // number of parameters in p_{t+1}
        4,                         // number of parameters in q_t
        3,                         // number of parameters in v_t
        3> {                       // number of parameters in p_t
        // 3,                         // number of parameters of gyro bias
        // 3> {                        // number of parameters of accel bias
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base in ceres we derive from
  typedef ceres::SizedCostFunction<10, 4, 3, 3, 4, 3, 3> base_t;

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
  PreIntImuError() {
  }

  /// \brief Trivial destructor.
  ~PreIntImuError() {
  }

  /// 
  PreIntImuError(const Eigen::Matrix3d d_rotation,
                 const Eigen::Vector3d d_velocity,
                 const Eigen::Vector3d d_position,
                 const double dt) {
    d_rotation_ = d_rotation;
    d_velocity_ = d_velocity;
    d_position_ = d_position;
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

    Eigen::Quaterniond rotation_t1(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]);
    Eigen::Vector3d velocity_t1(parameters[1]);
    Eigen::Vector3d position_t1(parameters[2]);
    Eigen::Quaterniond rotation_t(parameters[3][0], parameters[3][1], parameters[3][2], parameters[3][3]);
    Eigen::Vector3d velocity_t(parameters[4]);
    Eigen::Vector3d position_t(parameters[5]);

    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);
    Eigen::Vector3d gyro_bias = Eigen::Vector3d(-0.003196, 0.021298, 0.078430);
    Eigen::Vector3d accel_bias = Eigen::Vector3d(-0.026176, 0.137568, 0.076295);

    // residual vectors
    Eigen::Map<Eigen::Quaterniond> r_rotation(residuals+0);
    Eigen::Map<Eigen::Vector3d> r_velocity(residuals+4);      
    Eigen::Map<Eigen::Vector3d> r_position(residuals+7);      

    Eigen::Vector3d v_diff = velocity_t1 - velocity_t - dt_*gravity;
    Eigen::Vector3d p_diff = position_t1 - position_t - dt_*velocity_t - 0.5*(dt_*dt_)*gravity;

    r_rotation = rotation_t1.conjugate() * rotation_t * Eigen::Quaterniond(d_rotation_);
    r_velocity = d_velocity_ - rotation_t.toRotationMatrix().transpose() * v_diff;
    r_position = d_position_ - rotation_t.toRotationMatrix().transpose() * p_diff;


    /*********************************************************************************

                 Jacobian

    *********************************************************************************/

    if (jacobians != NULL) {

      // rotation_t1
      if (jacobians[0] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 4, Eigen::RowMajor> > J_q_t1(jacobians[0]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<4; ++j) {
            J_q_t1(i,j) = 0.0;
          }
        }
        
        /***
        Eigen::Quaterniond r_t_mul_r_meas = rotation_t * Eigen::Quaterniond(d_rotation_);
        J_q_t1.block<4,4>(0,0) <<  r_t_mul_r_meas.w(),  r_t_mul_r_meas.x(),  r_t_mul_r_meas.y(),  r_t_mul_r_meas.z(),
                                   r_t_mul_r_meas.x(), -r_t_mul_r_meas.w(), -r_t_mul_r_meas.z(),  r_t_mul_r_meas.y(),
                                   r_t_mul_r_meas.y(),  r_t_mul_r_meas.z(), -r_t_mul_r_meas.w(), -r_t_mul_r_meas.x(),
                                   r_t_mul_r_meas.z(), -r_t_mul_r_meas.y(),  r_t_mul_r_meas.x(), -r_t_mul_r_meas.w();
        ***/
        J_q_t1(0,0) = 1.0;
        J_q_t1(1,1) = 1.0;
        J_q_t1(2,2) = 1.0;
        J_q_t1(3,3) = 1.0;

      }  

      // velocity_t1
      if (jacobians[1] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor> > J_v_t1(jacobians[1]);

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<3; ++j) {
            J_v_t1(i,j) = 0.0;
          }
        }

        J_v_t1.block<3,3>(4,0) = (-1) * rotation_t.toRotationMatrix().transpose();


      }  


      // position_t1
      if (jacobians[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor> > J_p_t1(jacobians[2]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<3; ++j) {
            J_p_t1(i,j) = 0.0;
          }
        }

        J_p_t1.block<3,3>(7,0) = (-1) * rotation_t.toRotationMatrix().transpose();

      }

      // rotation_t
      if (jacobians[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 10, 4, Eigen::RowMajor> > J_q_t(jacobians[3]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<4; ++j) {
            J_q_t(i,j) = 0.0;
          }
        }

        J_q_t(0,0) = -1.0;
        J_q_t(1,1) = -1.0;
        J_q_t(2,2) = -1.0;
        J_q_t(3,3) = -1.0;        

        J_q_t.block<3,1>(4,0) = (-1) * Jacob_R_to_qw(rotation_t).transpose() * v_diff;
        J_q_t.block<3,1>(4,1) = (-1) * Jacob_R_to_qx(rotation_t).transpose() * v_diff;
        J_q_t.block<3,1>(4,2) = (-1) * Jacob_R_to_qy(rotation_t).transpose() * v_diff;
        J_q_t.block<3,1>(4,3) = (-1) * Jacob_R_to_qy(rotation_t).transpose() * v_diff;
 
        J_q_t.block<3,1>(7,0) = (-1) * Jacob_R_to_qw(rotation_t).transpose() * p_diff;
        J_q_t.block<3,1>(7,1) = (-1) * Jacob_R_to_qx(rotation_t).transpose() * p_diff;
        J_q_t.block<3,1>(7,2) = (-1) * Jacob_R_to_qy(rotation_t).transpose() * p_diff;
        J_q_t.block<3,1>(7,3) = (-1) * Jacob_R_to_qy(rotation_t).transpose() * p_diff;
      }  


      // velocity_t
      if (jacobians[4] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor> > J_v_t(jacobians[4]);

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<3; ++j) {
            J_v_t(i,j) = 0.0;
          }
        }

        J_v_t.block<3,3>(4,0) = rotation_t.toRotationMatrix().transpose();
        J_v_t.block<3,3>(7,0) = dt_ * rotation_t.toRotationMatrix().transpose();

      }  

      // position_t
      if (jacobians[5] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 10, 3, Eigen::RowMajor> > J_p_t(jacobians[5]);      

        for (size_t i=0; i<10; ++i) {
          for (size_t j=0; j<3; ++j) {
            J_p_t(i,j) = 0.0;
          }
        }

        J_p_t.block<3,3>(7,0) = rotation_t.toRotationMatrix().transpose();
      }
    }

    return true;
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
    return "PreIntImuError";
  }

 protected:

  // measurements
  Eigen::Matrix3d d_rotation_;
  Eigen::Vector3d d_velocity_;
  Eigen::Vector3d d_position_;

  // times
  double dt_;

};

#endif /* INCLUDE_PRE_INT_IMU_ERROR_H_ */
