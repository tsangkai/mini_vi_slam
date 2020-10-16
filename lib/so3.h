


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


#endif /* INCLUDE_SO3_H_ */
