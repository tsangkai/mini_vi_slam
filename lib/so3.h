


#ifndef INCLUDE_SO3_H_
#define INCLUDE_SO3_H_


#include <cmath>


double sinc(double x){
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


Eigen::Quaterniond Exp_q(const Eigen::Vector3d v) {
  Eigen::Quaterniond q;

  const double v_half_norm = 0.5 * v.norm();
  const double sinc_v_half_norm = sinc(v_half_norm);
  const double cos_v_half_norm = cos(v_half_norm);

  q.w() = cos_v_half_norm;  
  q.vec() = 0.5 * sinc_v_half_norm * v;

  return q;
}

Eigen::Vector3d Log_q(const Eigen::Quaterniond q) {

  double a = acos(q.w() / q.norm());

  if (abs(a) < 1e-10) {
    return 2 * q.vec();
  }

  return 2*(a / sin(a)) * q.vec();
}


// plus() in okvis
Eigen::Matrix4d QuatLeftMul(const Eigen::Quaterniond & q) {       
  Eigen::Matrix4d Q;
  Q(0,0) = q.w(); Q(0,1) = -q.x(); Q(0,2) = -q.y(); Q(0,3) = -q.z();
  Q(1,0) = q.x(); Q(1,1) = -q.w(); Q(1,2) = -q.z(); Q(1,3) =  q.y();
  Q(2,0) = q.y(); Q(2,1) =  q.z(); Q(2,2) = -q.w(); Q(2,3) = -q.x();
  Q(3,0) = q.z(); Q(3,1) = -q.y(); Q(3,2) =  q.x(); Q(3,3) = -q.w();
  return Q;
}


// oplus() in okvis
Eigen::Matrix4d QuatRightMul(const Eigen::Quaterniond & q) {       
  Eigen::Matrix4d Q;
  Q(0,0) = q.w(); Q(0,1) = -q.x(); Q(0,2) = -q.y(); Q(0,3) = -q.z();
  Q(1,0) = q.x(); Q(1,1) =  q.w(); Q(1,2) =  q.z(); Q(1,3) = -q.y();
  Q(2,0) = q.y(); Q(2,1) = -q.z(); Q(2,2) =  q.w(); Q(2,3) =  q.x();
  Q(3,0) = q.z(); Q(3,1) =  q.y(); Q(3,2) = -q.x(); Q(3,3) =  q.w();
  return Q;
}


// directly from okvis
// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
Eigen::Matrix<double, 3, 4> QuatLiftJacobian(const Eigen::Quaterniond & q) {

  Eigen::Matrix<double, 3, 4> J_lift;

  const Eigen::Quaterniond q_inv = q.conjugate();
  Eigen::Matrix4d q_inv_right_mul = QuatRightMul(q_inv);

  Eigen::Matrix<double, 3, 4> Jq_pinv;
  Jq_pinv.bottomRightCorner<3, 1>().setZero();
  Jq_pinv.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * 2.0;

  J_lift = Jq_pinv * q_inv_right_mul;

  return J_lift;
}


Eigen::Matrix3d RightJacobian(const Eigen::Vector3d & v) {

  Eigen::Matrix3d right_jacobian;

  const double v_norm = v.norm();
  const double v_norm_2 = v_norm*v_norm;
  const double v_norm_3 = v_norm_2*v_norm;
  const Eigen::Matrix3d skewed_v = Skew(v);

  assert(("norm is too small", v_norm > 1e-10));

  right_jacobian = Eigen::Matrix3d::Identity() - ((1-cos(v_norm))/v_norm_2) * skewed_v + ((v_norm-sin(v_norm))/v_norm_3) * skewed_v * skewed_v;

  return right_jacobian;
}

Eigen::Matrix3d RightJacobianInv(const Eigen::Vector3d & v) {

  Eigen::Matrix3d right_jacobian_inv;

  const double v_norm = v.norm();
  const double v_norm_2 = v_norm*v_norm;
  const Eigen::Matrix3d skewed_v = Skew(v);

  assert(("norm is too small", v_norm > 1e-10));

  right_jacobian_inv = Eigen::Matrix3d::Identity() + 0.5 * skewed_v + (1/v_norm_2  - (1+cos(v_norm))/(2*v_norm*sin(v_norm))) * skewed_v * skewed_v;

  return right_jacobian_inv;
}

#endif /* INCLUDE_SO3_H_ */
