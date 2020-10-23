
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "transformation.h"
#include "so3.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "imu_error.h"

#define _USE_MATH_DEFINES


struct ImuParameters{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Transformation T_BS; ///< Transformation from Body frame to IMU (sensor frame S).
  double a_max;  ///< Accelerometer saturation. [m/s^2]
  double g_max;  ///< Gyroscope saturation. [rad/s]
  double sigma_g_c;  ///< Gyroscope noise density.
  double sigma_bg;  ///< Initial gyroscope bias.
  double sigma_a_c;  ///< Accelerometer noise density.
  double sigma_ba;  ///< Initial accelerometer bias
  double sigma_gw_c; ///< Gyroscope drift noise density.
  double sigma_aw_c; ///< Accelerometer drift noise density.
  double tau;  ///< Reversion time constant of accerometer bias. [s]
  double g;  ///< Earth acceleration.
  Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
  int rate;  ///< IMU rate in Hz.
};

struct IMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp_;
  Eigen::Vector3d gyro_;
  Eigen::Vector3d accel_; 
};


// This class only handles the memory. The underlying estimate is handled by each parameter blocks.
class State {

 public:
  State(double timestamp) {
    timestamp_ = timestamp;

    rotation_block_ptr_ = new QuatParameterBlock();
    velocity_block_ptr_ = new Vec3dParameterBlock();
    position_block_ptr_ = new Vec3dParameterBlock();
  }

  ~State() {
    delete [] rotation_block_ptr_;
    delete [] velocity_block_ptr_;
    delete [] position_block_ptr_;
  }

  double GetTimestamp() {
    return timestamp_;
  }

  QuatParameterBlock* GetRotationBlock() {
    return rotation_block_ptr_;
  }

  Vec3dParameterBlock* GetVelocityBlock() {
    return velocity_block_ptr_;
  }

  Vec3dParameterBlock* GetPositionBlock() {
    return position_block_ptr_;
  }

 private:
  double timestamp_;
  QuatParameterBlock* rotation_block_ptr_;
  Vec3dParameterBlock* velocity_block_ptr_;
  Vec3dParameterBlock* position_block_ptr_;
};



int main(int argc, char **argv) {
  srand((unsigned int) time(0));

  google::InitGoogleLogging(argv[0]);

  // set the imu parameters
  ImuParameters imuParameters;
  imuParameters.a0.setZero();
  imuParameters.g = 9.81007;
  imuParameters.a_max = 1000.0;
  imuParameters.g_max = 1000.0;
  imuParameters.rate = 1000; // 1 kHz
  imuParameters.sigma_g_c = 6.0e-4;
  imuParameters.sigma_a_c = 2.0e-3;
  imuParameters.sigma_gw_c = 3.0e-6;
  imuParameters.sigma_aw_c = 2.0e-5;
  imuParameters.tau = 3600.0;

  // generate random motion
  const double w_omega_S_x = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_y = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_z = Eigen::internal::random(0.1,10.0); // circular frequency
  const double p_omega_S_x = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_y = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_z = Eigen::internal::random(0.0,M_PI); // phase
  const double m_omega_S_x = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_y = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_z = Eigen::internal::random(0.1,1.0); // magnitude
  const double w_a_W_x = Eigen::internal::random(0.1,10.0);
  const double w_a_W_y = Eigen::internal::random(0.1,10.0);
  const double w_a_W_z = Eigen::internal::random(0.1,10.0);
  const double p_a_W_x = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_y = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_z = Eigen::internal::random(0.1,M_PI);
  const double m_a_W_x = Eigen::internal::random(0.1,10.0);
  const double m_a_W_y = Eigen::internal::random(0.1,10.0);
  const double m_a_W_z = Eigen::internal::random(0.1,10.0);


  Transformation T_WS;
  T_WS.SetRandom(10.0, M_PI);

  // time increment
  const double dt = 0.01; // time discretization

  // states
  Eigen::Quaterniond rotation = T_WS.q();
  Eigen::Vector3d position = T_WS.t();
  Eigen::Vector3d velocity(0.1, 0.1, 0.1);


  // start
  double t_0 = 0;
  Transformation T_WS_0 = T_WS;
  Eigen::Vector3d velocity_0 = velocity;




  // just single IMU input
  double time = dt;

  Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
                          m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
                          m_omega_S_z*sin(w_omega_S_z*time+p_omega_S_z));

  Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
                      m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
                      m_a_W_z*sin(w_a_W_z*time+p_a_W_z));

  Eigen::Quaterniond dq = Exp_q(omega_S*dt);

  rotation = rotation * dq;

  // propagate position
  position += dt*velocity;
  
  // propagate speed
  velocity += dt*a_W;





  // end
  double t_1 = time;
  Transformation T_WS_1 = Transformation(rotation, position);
  Eigen::Vector3d velocity_1 = velocity;



  // generate measurements
  Eigen::Vector3d gyr_noise = imuParameters.sigma_g_c/sqrt(dt) * Eigen::Vector3d::Random();
  Eigen::Vector3d acc_noise = imuParameters.sigma_a_c/sqrt(dt) * Eigen::Vector3d::Random();

  Eigen::Vector3d gyr = omega_S + gyr_noise;
  Eigen::Vector3d acc = T_WS.inverse().C()*(a_W + Eigen::Vector3d(0,0,imuParameters.g)) + acc_noise;
  
  IMUData imu_data;

  imu_data.timestamp_ = time;
  imu_data.gyro_ = gyr;
  imu_data.accel_ = acc;
  
  //=========================================================================================================
  // Build the problem.
  ceres::Problem optimization_problem;
  ceres::LocalParameterization* quat_parameterization_ptr_ = new ceres::QuaternionParameterization();


  // create the pose parameter blocks
  Transformation T_disturb;
  T_disturb.SetRandom(1,0.2);
  Transformation T_WS_1_disturbed = T_WS_1 * T_disturb; //

  State* state_0 = new State(t_0);
  State* state_1 = new State(t_1);


  optimization_problem.AddParameterBlock(state_0->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(state_0->GetVelocityBlock()->parameters(), 3);
  optimization_problem.AddParameterBlock(state_0->GetPositionBlock()->parameters(), 3);

  optimization_problem.AddParameterBlock(state_1->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(state_1->GetVelocityBlock()->parameters(), 3);
  optimization_problem.AddParameterBlock(state_1->GetPositionBlock()->parameters(), 3);

  state_0->GetRotationBlock()->setEstimate(T_WS_0.q());
  state_0->GetVelocityBlock()->setEstimate(velocity_0);
  state_0->GetPositionBlock()->setEstimate(T_WS_0.t());

  state_1->GetRotationBlock()->setEstimate(T_WS_1_disturbed.q());
  state_1->GetVelocityBlock()->setEstimate(velocity_1);
  state_1->GetPositionBlock()->setEstimate(T_WS_1_disturbed.t());

  optimization_problem.SetParameterBlockConstant(state_0->GetRotationBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state_0->GetVelocityBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state_0->GetPositionBlock()->parameters());

  // add constraints
  ceres::CostFunction* cost_function = new ImuError(imu_data.gyro_,
                                                    imu_data.accel_,
                                                    dt,
                                                    Eigen::Vector3d(0,0,0),
                                                    Eigen::Vector3d(0,0,0),                   
                                                    imuParameters.sigma_g_c,
                                                    imuParameters.sigma_a_c);

  optimization_problem.AddResidualBlock(cost_function,
                                        NULL,
                                        state_1->GetRotationBlock()->parameters(),
                                        state_1->GetVelocityBlock()->parameters(),
                                        state_1->GetPositionBlock()->parameters(),
                                        state_0->GetRotationBlock()->parameters(),
                                        state_0->GetVelocityBlock()->parameters(),
                                        state_0->GetPositionBlock()->parameters());   

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  ceres::Solver::Options optimization_options;
  ceres::Solver::Summary optimization_summary;

  ceres::Solve(optimization_options, &optimization_problem, &optimization_summary);


  // print some infos about the optimization
  std::cout << optimization_summary.FullReport() << "\n";

  Transformation T_WS_1_opt = Transformation(state_1->GetRotationBlock()->estimate(), state_1->GetPositionBlock()->estimate());
  std::cout << "initial T_WS_1 : " << T_WS_1_disturbed.T() << "\n"
            << "optimized T_WS_1 : " << T_WS_1_opt.T() << "\n"
            << "correct T_WS_1 : " << T_WS_1.T() << "\n";

  std::cout << "rotation difference of the initial T_nb : " << 2*(T_WS_1.q() * T_WS_1_disturbed.q().inverse()).vec().norm() << "\n";
  std::cout << "rotation difference of the optimized T_nb : " << 2*(T_WS_1.q() * T_WS_1_opt.q().inverse()).vec().norm() << "\n";

  std::cout << "translation difference of the initial T_nb : " << (T_WS_1.t() - T_WS_1_disturbed.t()).norm() << "\n";
  std::cout << "translation difference of the optimized T_nb : " << (T_WS_1.t() - T_WS_1_opt.t()).norm() << "\n";

  // make sure it converged
  // assert(("cost not reducible", optimization_summary.final_cost < 1e-2));
  // assert(("quaternions not close enough", 2*(T_WS_1.q() * T_WS_1_opt.q().inverse()).vec().norm() < 1e-2));
  // assert(("translation not close enough", (T_WS_1.t() - T_WS_1_opt.t()).norm() < 0.04));

  return 0;
}

