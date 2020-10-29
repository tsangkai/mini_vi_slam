
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
#include "pre_int_imu_error.h"
#include "reprojection_error.h"

#define _USE_MATH_DEFINES

double sinc_test(double x){
  if(fabs(x)>1e-10) {
    return sin(x)/x;
  }
  else {
    static const double c_2=1.0/6.0;
    static const double c_4=1.0/120.0;
    static const double c_6=1.0/5040.0;
    const double x_2 = x*x;
    const double x_4 = x_2*x_2;
    const double x_6 = x_2*x_2*x_2;
    
    return 1.0 - c_2*x_2 + c_4*x_4 - c_6*x_6;
  }
}

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

struct PreIntIMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreIntIMUData(double dt, 
                Eigen::Matrix3d d_rotation, 
                Eigen::Vector3d d_velocity, 
                Eigen::Vector3d d_position) {
    dt_ = dt;
    d_rotation_ = d_rotation;
    d_velocity_ = d_velocity;
    d_position_ = d_position;
  }

  double dt_;
  Eigen::Matrix3d d_rotation_;  
  Eigen::Vector3d d_velocity_;
  Eigen::Vector3d d_position_; 
};

PreIntIMUData Preintegrate(std::vector<IMUData> imu_data_vec) {

	double imu_dt_ = 0.001;
    Eigen::Vector3d gyro_bias = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d accel_bias = Eigen::Vector3d(0, 0, 0);

    Eigen::Matrix3d Delta_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d Delta_V = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d Delta_P = Eigen::Vector3d(0, 0, 0);

    for (size_t i=0; i<imu_data_vec.size(); i++) {
      Delta_P = Delta_P + imu_dt_*Delta_V + 0.5*(imu_dt_*imu_dt_)*Delta_R*(imu_data_vec.at(i).accel_ - accel_bias);
      Delta_V = Delta_V + imu_dt_ * Delta_R*(imu_data_vec.at(i).accel_ - accel_bias);
      Delta_R = Delta_R * Exp(imu_dt_ * (imu_data_vec.at(i).gyro_ - gyro_bias));
    }

    return PreIntIMUData(imu_data_vec.size()*imu_dt_, Delta_R, Delta_V, Delta_P);
}

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
  // srand((unsigned int) time(0));

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


  // storage of IMU data
  std::vector<IMUData> imu_data_vec;

  // generate randomized measurements - duration 10 seconds
  const double duration = 1.0;
  // okvis::ImuMeasurementDeque imuMeasurements;
  Transformation T_WS;
  //T_WS.setRandom();

  // time increment
  const double dt = 1.0/imuParameters.rate; // time discretization

  // states
  Eigen::Quaterniond q = T_WS.q();
  Eigen::Vector3d position = T_WS.t();  // r
  Eigen::Vector3d velocity;
  // speedAndBias.setZero();
  // Eigen::Vector3d v=speedAndBias.head<3>();

  // start
  double t_0;
  Transformation T_WS_0;
  Eigen::Vector3d velocity_0;

  // end
  double t_1;
  Transformation T_WS_1;
  Eigen::Vector3d velocity_1;

  for(size_t i=0; i<size_t(duration*imuParameters.rate); ++i) {
    double time = double(i)/imuParameters.rate;

    if (i==10) { // set this as starting pose
      T_WS_0 = T_WS;
      velocity_0 = velocity;
      t_0 = time;
    }
	
    if (i==size_t(duration*imuParameters.rate)-10) { // set this as ending pose
      T_WS_1 = T_WS;
      velocity_1 = velocity;
      t_1 = time;
    }

    Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
                            m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
                            m_omega_S_z*sin(w_omega_S_z*time+p_omega_S_z));
    Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
                        m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
                        m_a_W_z*sin(w_a_W_z*time+p_a_W_z));

    //omega_S.setZero();
    //a_W.setZero();

    Eigen::Quaterniond dq;

    // propagate orientation
    const double theta_half = omega_S.norm()*dt*0.5;
    const double sinc_theta_half = sinc_test(theta_half);
    const double cos_theta_half = cos(theta_half);
    dq.vec() = sinc_theta_half*0.5*dt*omega_S;
    dq.w() = cos_theta_half;
    q = q * dq;

    // propagate speed
    velocity += dt*a_W;

    // propagate position
    position += dt*velocity;

    // T_WS
    T_WS = Transformation(q, position);

    // generate measurements
    Eigen::Vector3d gyr_noise = imuParameters.sigma_g_c/sqrt(dt) * Eigen::Vector3d::Random();
    Eigen::Vector3d acc_noise = imuParameters.sigma_a_c/sqrt(dt) * Eigen::Vector3d::Random();

    Eigen::Vector3d gyr = omega_S + gyr_noise;
    Eigen::Vector3d acc = T_WS.inverse().C()*(a_W + Eigen::Vector3d(0,0,imuParameters.g)) + acc_noise;
  
    IMUData imu_data;

    imu_data.timestamp_ = time;
    imu_data.gyro_ = gyr;
    imu_data.accel_ = acc;


    if (i>=10 && i<size_t(duration*imuParameters.rate)-10) {
      imu_data_vec.push_back(imu_data);
    }
  }
  


  PreIntIMUData pre_int_imu_data = Preintegrate(imu_data_vec);

  

  // Build the problem.
  ceres::Problem optimization_problem;
  ceres::LocalParameterization* quat_parameterization_ptr_ = new ceres::QuaternionParameterization();

  

  // create the pose parameter blocks
  Transformation T_disturb;
  T_disturb.SetRandom(1,0.02);
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



  // create the Imu error term      
  ceres::CostFunction* cost_function = new PreIntImuError(pre_int_imu_data.d_rotation_,
                                                          pre_int_imu_data.d_velocity_,
                                                          pre_int_imu_data.d_position_,
                                                          pre_int_imu_data.dt_);


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

