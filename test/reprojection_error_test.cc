
// This test file verifies reprojection_error.h
// modified from TestReprojectionError.h from okvis

#include <cassert>
#include <cmath>
#include <iostream>

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "reprojection_error.h"

#define _USE_MATH_DEFINES



Eigen::Vector2d Project(Eigen::Vector3d v, double fu, double fv, double cu, double cv) {
  assert(("the point is behind the camera", v[2] > 0));

  Eigen::Vector2d v_temp;  
  Eigen::Vector2d v_ret;
  
  v_temp[0] = v[0] / v[2];
  v_temp[1] = v[1] / v[2];

  v_ret[0] = fu * v_temp[0] + cu;
  v_ret[1] = fv * v_temp[0] + cv;

  return v_ret;
}

Eigen::Vector3d BackProject(Eigen::Vector2d v, double fu, double fv, double cu, double cv) {
  // unscale and center
  Eigen::Vector2d imagePoint2;
  imagePoint2[0] = (v[0] - cu) / fu;
  imagePoint2[1] = (v[1] - cv) / fv;

  Eigen::Vector3d v_ret;

  v_ret[0] = imagePoint2[0];
  v_ret[1] = imagePoint2[1];
  v_ret[2] = 1.0;

  return v_ret;
}


// Creates a random visible point in Euclidean coordinates.
Eigen::Vector3d CreateRandomVisiblePoint(double du, double dv,
                                         double fu, double fv, 
                                         double cu, double cv,
                                         double min_dist,
                                         double max_dist) {

  // Uniform random sample in image coordinates.
  // Add safety boundary for later inaccurate backprojection

  Eigen::Vector2d outPoint = Eigen::Vector2d::Random();
  outPoint += Eigen::Vector2d::Ones();
  outPoint *= 0.5;   // unif [0,1]

  outPoint[0] *= du;
  outPoint[1] *= dv;


  Eigen::Vector3d ray = BackProject(outPoint, fu, fv, cu, cv);
  ray.normalize();
  Eigen::Vector2d depth = Eigen::Vector2d::Random();
  Eigen::Vector3d point_c = (0.5 * (max_dist - min_dist) * (depth[0] + 1.0) + min_dist) * ray;    // rescale and offset

  return ray;
}



class Transformation {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Transformation() {
    t_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    q_ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  }

  Transformation(const Eigen::Quaterniond& q_AB, const Eigen::Vector3d& r_AB) {
    t_ = r_AB;
    q_ = q_AB.normalized();
  }

  void SetRandom(double translationMaxMeters,
                 double rotationMaxRadians) {
    // Create a random unit-length axis.
    Eigen::Vector3d axis = rotationMaxRadians * Eigen::Vector3d::Random();
    // Create a random rotation angle in radians.
    Eigen::Vector3d t = translationMaxMeters * Eigen::Vector3d::Random();
    t_ = t;
    q_ = Eigen::AngleAxisd(axis.norm(), axis.normalized());
  }

  Eigen::Matrix4d T() const {
    Eigen::Matrix4d T_ret;
    T_ret.topLeftCorner<3, 3>() = q_.toRotationMatrix();
    T_ret.topRightCorner<3, 1>() = t_;
    T_ret.bottomLeftCorner<1, 3>().setZero();
    T_ret(3, 3) = 1.0;

    return T_ret;
  }

  Eigen::Vector3d t() {
    return t_;
  }

  Eigen::Quaterniond q() {
    return q_;
  }

  // operator*
  Transformation operator*(const Transformation & rhs) const {
    return Transformation(q_ * rhs.q_, q_.toRotationMatrix() * rhs.t_ + t_);
  }

  Transformation& operator=(const Transformation & rhs) {
    t_ = rhs.t_;
    q_ = rhs.q_;
    return *this;
  }

 protected:
  Eigen::Vector3d t_;     ///< Translation {_A}r_{B}.
  Eigen::Quaterniond q_;  ///< Quaternion q_{AB}.
};


int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  double du = 752.0;          // image dimension
  double dv = 480.0;
  double fu = 458.654880721;  // focal length
  double fv = 457.296696463;
  double cu = 367.215803962;  // principal point
  double cv = 248.375340610;
  double noise_deviation = 3.0;


  // Build the problem.
  ceres::Problem optimization_problem;
  ceres::LocalParameterization* quat_parameterization_ptr_ = new ceres::QuaternionParameterization();

  // set up a random geometry
  std::cout << "set up a random geometry... " << std::flush;

  Transformation T_nb;                         // navigation to body
  T_nb.SetRandom(10.0, M_PI);

  Transformation T_disturb;
  T_disturb.SetRandom(1, 0.1);

  Transformation T_nb_init = T_nb * T_disturb; // navigation to body

  Transformation T_bc;                         // body to camera
  T_bc.SetRandom(0.2, M_PI);

  QuatParameterBlock* rotation_block_ptr = new QuatParameterBlock(T_nb.q());
  Vec3dParameterBlock* position_block_ptr = new Vec3dParameterBlock(T_nb.t());

  optimization_problem.AddParameterBlock(rotation_block_ptr->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(position_block_ptr->parameters(), 3);  
  optimization_problem.SetParameterBlockVariable(rotation_block_ptr->parameters()); // optimize this...
  optimization_problem.SetParameterBlockVariable(position_block_ptr->parameters());
  std::cout << " [ OK ] " << std::endl;


  // get some random points and build error terms
  const size_t N=100;
  std::cout << "create N=" << N << " visible points and add respective reprojection error terms... " << std::flush;
  for (size_t i=1; i<N; ++i){

    // create random visible point
    double max_dist = 100;
    double min_dist = double(i%10)*3 + 2.0;

    Eigen::Vector3d landmark_c = CreateRandomVisiblePoint(du, dv, fu, fv, cu, cv, min_dist, max_dist);

    Eigen::Vector4d h_landmark_c(landmark_c[0], landmark_c[1], landmark_c[2], 1);
    Eigen::Vector4d h_landmark_n = T_nb.T() *T_bc.T() * h_landmark_c;
    Eigen::Vector3d landmark = h_landmark_n.head<3>();

    Vec3dParameterBlock* landmark_ptr = new Vec3dParameterBlock(landmark);
    optimization_problem.AddParameterBlock(landmark_ptr->parameters(), 3);
    optimization_problem.SetParameterBlockConstant(landmark_ptr->parameters());


    // get a randomized projection
    Eigen::Vector2d keypoint = Project(h_landmark_c.head<3>(), fu, fv, cu, cv);

    keypoint += noise_deviation * Eigen::Vector2d::Random();


    ceres::CostFunction* cost_function = new ReprojectionError(keypoint,
                                                               T_bc.T(),
                                                               fu, fv,
                                                               cu, cv);
    optimization_problem.AddResidualBlock(cost_function, 
                                          NULL, 
                                          rotation_block_ptr->parameters(),
                                          position_block_ptr->parameters(),
                                          landmark_ptr->parameters());
  }

  std::cout << " [ OK ] " << std::endl;


  std::cout<<"run the solver... "<<std::endl;
  ceres::Solver::Options optimization_options;
  optimization_options.max_num_iterations = 100;
  ceres::Solver::Summary optimization_summary;

  ceres::Solve(optimization_options, &optimization_problem, &optimization_summary);
  std::cout << optimization_summary.FullReport() << "\n";

  std::cout << "initial T_nb : " << "\n" << T_nb_init.T() << "\n"
            << "optimized T_nb : " << "\n" << Transformation(rotation_block_ptr->estimate(), position_block_ptr->estimate()).T() << "\n"
            << "correct T_nb : " << "\n" << T_nb.T() << "\n";

  std::cout << "rotation difference of the initial T_nb : " << 2*(T_nb.q() * T_nb_init.q().inverse()).vec().norm() << "\n";
  std::cout << "rotation difference of the optimized T_nb : " << 2*(T_nb.q() * rotation_block_ptr->estimate().inverse()).vec().norm() << "\n";

  std::cout << "translation difference of the initial T_nb : " << (T_nb.t() - T_nb_init.t()).norm() << "\n";
  std::cout << "translation difference of the optimized T_nb : " << (T_nb.t() - position_block_ptr->estimate()).norm() << "\n";

  // make sure it converged
  assert(("quaternions not close enough", 2*(T_nb.q() * rotation_block_ptr->estimate().inverse()).vec().norm() < 1e-2));
  assert(("translation not close enough", (T_nb.t() - position_block_ptr->estimate()).norm() < 1e-1));


  return 0;
}