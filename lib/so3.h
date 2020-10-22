


#ifndef INCLUDE_SO3_H_
#define INCLUDE_SO3_H_


#include <cmath>


Eigen::Matrix3d Skew(Eigen::Vector3d v) {
  Eigen::Matrix3d m;
  m <<     0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;

  return m;
}


Eigen::Matrix3d Hat(Eigen::Vector3d v) {
  Eigen::Matrix3d m;
  m <<     0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;

  return m;
}


Eigen::Matrix3d Exp(Eigen::Vector3d omega) {
  Eigen::Matrix3d hatted_omega = Hat(omega);
  double bar_omega = acos(0.5*(hatted_omega.trace()-1));

  return Eigen::Matrix3d::Identity() + (sin(bar_omega)/bar_omega) * hatted_omega + ((1-cos(bar_omega))/(bar_omega*bar_omega)) * hatted_omega*hatted_omega;
}


Eigen::Vector3d Log_q(Eigen::Quaterniond q) {

  double a = acos(q.w() / q.norm());

  if (abs(a) < 1e-8) {
    return q.vec();
  }

  return (a / sin(a)) * q.vec();
}


// oplus() in okvis
Eigen::Matrix4d QuatRightMul(const Eigen::Quaterniond & q_BC) {       
  Eigen::Vector4d q = q_BC.coeffs();
  Eigen::Matrix4d Q;
  Q(0,0) = q.w(); Q(0,1) = -q.x(); Q(0,2) = -q.y(); Q(0,3) = -q.z();
  Q(1,0) = q.x(); Q(1,1) =  q.w(); Q(1,2) =  q.z(); Q(1,3) = -q.y();
  Q(2,0) = q.y(); Q(2,1) = -q.z(); Q(2,2) =  q.w(); Q(2,3) =  q.x();
  Q(3,0) = q.z(); Q(3,1) =  q.y(); Q(3,2) = -q.x(); Q(3,3) =  q.w();
  return Q;
}


// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool LiftJacobian(const Eigen::Quaterniond & q, double* jacobian) {

  Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > J_lift(jacobian);
  const Eigen::Quaterniond q_inv = q.conjugate();
  Eigen::Matrix4d q_inv_right_mul = QuatRightMul(q_inv);

  Eigen::Matrix<double, 3, 4> Jq_pinv;
  Jq_pinv.bottomRightCorner<3, 1>().setZero();
  Jq_pinv.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 2.0;

  J_lift = Jq_pinv * q_inv_right_mul;

  return true;
}


#endif /* INCLUDE_SO3_H_ */
