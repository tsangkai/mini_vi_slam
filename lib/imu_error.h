

/**
 * @file imu_error.hpp
 * @brief Header file for the ImuError class.
 * @author Tsang-Kai Chang
 */

// TODO(tsangkai): Implement bias parameters (interface, residual, and jacobian)

#ifndef INCLUDE_IMU_ERROR_H_
#define INCLUDE_IMU_ERROR_H_

#include "so3.h"
// #include <mutex>
// #include "ceres/ceres.h"
// #include <okvis/FrameTypedefs.hpp>
// #include <okvis/Time.hpp>
// #include <okvis/assert_macros.hpp>
// #include <okvis/Measurements.hpp>
// #include <okvis/Variables.hpp>
// #include <okvis/Parameters.hpp>
// #include <okvis/ceres/ErrorInterface.hpp>


/// \brief Implements a nonlinear IMU factor.
class ImuError :
    public ceres::SizedCostFunction<9,     // number of residuals
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
  typedef ceres::SizedCostFunction<9, 4, 3, 3, 4, 3, 3> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 9;

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
           const double dt,
           const double sigma_g_c = 12.0e-4,    // gyro noise density [rad/s/sqrt(Hz)]
           const double sigma_a_c = 8.0e-3) {   // accelerometer noise density [m/s^2/sqrt(Hz)]
    gyro_measurement_ = gyro_measurement;
    accel_measurement_ = accel_measurement;
    dt_ = dt;

    sigma_g_c_ = sigma_g_c;
    sigma_a_c_ = sigma_a_c;
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
    
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      
    Eigen::Vector3d gyro_bias = Eigen::Vector3d(0, 0, 0); //Eigen::Vector3d(-0.003196, 0.021298, 0.078430);
    Eigen::Vector3d accel_bias = Eigen::Vector3d(0, 0, 0); //(-0.026176, 0.137568, 0.076295);

    Eigen::Quaterniond _rotation_t1(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]);
    Eigen::Vector3d _velocity_t1(parameters[1]);
    Eigen::Vector3d _position_t1(parameters[2]);
    Eigen::Quaterniond _rotation_t(parameters[3][0], parameters[3][1], parameters[3][2], parameters[3][3]);
    Eigen::Vector3d _velocity_t(parameters[4]);
    Eigen::Vector3d _position_t(parameters[5]);

    // residual vectors
    Eigen::Map<Eigen::Vector3d > r_rotation(residuals+0);      
    Eigen::Map<Eigen::Vector3d > r_velocity(residuals+3);      
    Eigen::Map<Eigen::Vector3d > r_position(residuals+6);      

    Eigen::Vector3d _accel_plus_gravity = _rotation_t.normalized().toRotationMatrix()*(accel_measurement_ - accel_bias) + gravity;
    Eigen::Vector3d v_diff = dt_* _accel_plus_gravity;
    Eigen::Vector3d p_diff = dt_*_velocity_t + (dt_*dt_*0.5)* _accel_plus_gravity;

    Eigen::Quaterniond d_q(1, 0.5*dt_*(gyro_measurement_(0)-gyro_bias(0)),
                              0.5*dt_*(gyro_measurement_(1)-gyro_bias(1)),
                              0.5*dt_*(gyro_measurement_(2)-gyro_bias(2)));

    r_rotation = -(2/dt_) * Log_q((_rotation_t * d_q).conjugate() * _rotation_t1);



    r_velocity = _velocity_t1 - ( _velocity_t + v_diff);
    r_position = _position_t1 - ( _position_t + p_diff);


    // covariance adjustment
    double gyr_input_sigma = sigma_g_c_ / sqrt(dt_);
    double acc_input_sigma = sigma_a_c_ / sqrt(dt_);

    double q_noise_sigma = dt_ * gyr_input_sigma;
    double v_noise_sigma = dt_ * acc_input_sigma;
    double p_noise_sigma = 0.5 * dt_ * dt_ * acc_input_sigma;

    r_rotation = (1.0 / q_noise_sigma) * r_rotation;
    r_velocity = (1.0 / v_noise_sigma) * r_velocity;
    r_position = (1.0 / p_noise_sigma) * r_position;


    /*********************************************************************************

                 Jacobian

    *********************************************************************************/

    Eigen::MatrixXd p_log_q_2_q(3,4);
    p_log_q_2_q << 0, 1, 0, 0,
                   0, 0, 1, 0,
                   0, 0, 0, 1;

    Eigen::MatrixXd p_exp_q_2_q(4,3);
    p_exp_q_2_q << 0, 0, 0,
                   1, 0, 0,
                   0, 1, 0,
                   0, 0, 1;

    Eigen::MatrixXd p_exp_c_q_2_exp_q(4,4);
    p_exp_c_q_2_exp_q << 1,  0,  0,  0,
                         0, -1,  0,  0,
                         0,  0, -1,  0,
                         0,  0,  0, -1;



    if (jacobians != NULL) {

      // rotation_t1
      if (jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor> > J_q_t1(jacobians[0]);      
        J_q_t1.setZero();

        // TODO
        Eigen::Quaterniond q_t_inv_q_t1 = _rotation_t.conjugate() * _rotation_t1;
        Eigen::Matrix<double, 3, 3> J_res_2_dq_t1 = (-2/dt_) * RightJacobianInv(Log_q(q_t_inv_q_t1)) * _rotation_t.conjugate().toRotationMatrix();
      
        J_q_t1.block<3,4>(0,0) = (1/q_noise_sigma) * J_res_2_dq_t1 * QuatLiftJacobian(_rotation_t1);
      }  

      // velocity_t1
      if (jacobians[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_v_t1(jacobians[1]);      
        J_v_t1.setZero();

        J_v_t1.block<3,3>(3,0) = (1/v_noise_sigma) * Eigen::Matrix3d::Identity();
      }

      // position_t1
      if (jacobians[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_p_t1(jacobians[2]);      
        J_p_t1.setZero();

        J_p_t1.block<3,3>(6,0) = (1/p_noise_sigma) * Eigen::Matrix3d::Identity();
      }


      // rotation_t
      if (jacobians[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor> > J_q_t(jacobians[3]);      
        J_q_t.setZero();

        // TODO
        Eigen::Quaterniond q_t_inv_q_t1 = _rotation_t.conjugate() * _rotation_t1;
        Eigen::Matrix<double, 3, 3> J_res_2_dq_t = (-2/dt_) * RightJacobianInv(Log_q(q_t_inv_q_t1)) * (-1) * _rotation_t.toRotationMatrix().transpose();
      
        J_q_t.block<3,4>(0,0) = (1/q_noise_sigma) * J_res_2_dq_t * QuatLiftJacobian(_rotation_t);

        J_q_t.block<3,3>(3,0) = (1/v_noise_sigma) * (dt_) * Skew(_rotation_t.toRotationMatrix() * v_diff);
        J_q_t.block<3,3>(6,0) = (1/p_noise_sigma) * (0.5*dt_*dt_) * Skew(_rotation_t.toRotationMatrix() * p_diff);
      }  

      // velocity_t
      if (jacobians[4] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_v_t(jacobians[4]);      
        J_v_t.setZero();

        J_v_t.block<3,3>(3,0) = (1/v_noise_sigma) * (-1) * Eigen::Matrix3d::Identity();
        J_v_t.block<3,3>(6,0) = (1/p_noise_sigma) * (-dt_) * Eigen::Matrix3d::Identity();
      }  

      // position_t
      if (jacobians[5] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_p_t(jacobians[5]);      
        J_p_t.setZero();

        J_p_t.block<3,3>(6,0) = (1/p_noise_sigma) * (-1) * Eigen::Matrix3d::Identity();
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
    return "ImuError";
  }

 protected:

  // measurements
  Eigen::Vector3d gyro_measurement_;
  Eigen::Vector3d accel_measurement_; 

  // times
  double dt_;

  double sigma_g_c_;    // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;    // accelerometer noise density [m/s^2/sqrt(Hz)]

};

#endif /* INCLUDE_IMU_ERROR_H_ */
