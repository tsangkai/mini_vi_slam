

/**
 * @file reprojection_error.h
 * @brief Header file for the ReprojectionError class.
 * @author Tsang-Kai Chang
 */

#ifndef INCLUDE_REPROJECTION_ERROR_H_
#define INCLUDE_REPROJECTION_ERROR_H_

#include <ceres/ceres.h>

/// \brief Reprojection error base class.
class ReprojectionError:
    public ceres::SizedCostFunction<2,     // number of residuals
        4,                                 // number of parameters in q_t
        3,                                 // number of parameters in p_t
        3> {                               // number of landmark
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base class type.
  typedef ceres::SizedCostFunction<2, 4, 3, 3> base_t;

  /// \brief Number of residuals (2)
  static const int kNumResiduals = 2;

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d measurement_t;

  /// \brief Default constructor.
  ReprojectionError();

  // TODO(tsangkai): set camera intrinsic as static class member
  ReprojectionError(const measurement_t & measurement, 
                    Eigen::Transform<double, 3, Eigen::Affine> T_bc,
                    double focal, 
                    double* principle_point) {
    setMeasurement(measurement);

    T_bc_ = T_bc;
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

    Eigen::Quaterniond rotation(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]);
    Eigen::Vector3d position(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Vector3d landmark(parameters[2][0], parameters[2][1], parameters[2][2]);

    // navigation to body, which is just the state
    Eigen::Transform<double, 3, Eigen::Affine> T_nb = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
    T_nb.rotate(rotation);
    T_nb.translate(position);

    Eigen::Vector3d landmark_minus_p = landmark - position;

    Eigen::Vector3d landmark_b = T_nb.inverse() * landmark;
    Eigen::Vector3d landmark_c = T_bc_.inverse() * landmark_b;

    residuals[0] = -focal_ * landmark_c(0) / landmark_c(2) + principle_point_[0] - measurement_(0);
    residuals[1] = -focal_ * landmark_c(1) / landmark_c(2) + principle_point_[1] - measurement_(1);



    /*********************************************************************************

                 Jacobian

    *********************************************************************************/


    if (jacobians != NULL) {
      
      // chain rule
      Eigen::MatrixXd J_residual_to_lc(2,3);
      J_residual_to_lc(0,0) = -focal_ / landmark_c(2);
      J_residual_to_lc(0,1) = 0;
      J_residual_to_lc(0,2) = focal_ * landmark_c(0) / (landmark_c(2)*landmark_c(2));
      J_residual_to_lc(1,0) = 0;
      J_residual_to_lc(1,1) = -focal_ / landmark_c(2);
      J_residual_to_lc(1,2) = focal_ * landmark_c(1) / (landmark_c(2)*landmark_c(2));

      Eigen::MatrixXd J_lc_to_lb(3,3);
      J_lc_to_lb = T_bc_.rotation().transpose();


      // rotation
      if (jacobians[0] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J0(jacobians[0]);      

        Eigen::MatrixXd J_lb_to_q(3,4);
        J_lb_to_q(0,0) = landmark_minus_p(0)*( 2)*rotation.w()+landmark_minus_p(1)*( 2)*rotation.z()+landmark_minus_p(2)*(-2)*rotation.y();
        J_lb_to_q(0,1) = landmark_minus_p(0)*(-2)*rotation.x()+landmark_minus_p(1)*(-2)*rotation.y()+landmark_minus_p(2)*(-2)*rotation.z();
        J_lb_to_q(0,2) = landmark_minus_p(0)*( 2)*rotation.y()+landmark_minus_p(1)*(-2)*rotation.x()+landmark_minus_p(2)*( 2)*rotation.w();
        J_lb_to_q(0,3) = landmark_minus_p(0)*( 2)*rotation.z()+landmark_minus_p(1)*(-2)*rotation.w()+landmark_minus_p(2)*(-2)*rotation.x();

        J_lb_to_q(1,0) = landmark_minus_p(0)*(-2)*rotation.z()+landmark_minus_p(1)*( 2)*rotation.w()+landmark_minus_p(2)*( 2)*rotation.x();
        J_lb_to_q(1,1) = landmark_minus_p(0)*(-2)*rotation.y()+landmark_minus_p(1)*( 2)*rotation.x()+landmark_minus_p(2)*(-2)*rotation.w();
        J_lb_to_q(1,2) = landmark_minus_p(0)*(-2)*rotation.x()+landmark_minus_p(1)*(-2)*rotation.y()+landmark_minus_p(2)*(-2)*rotation.z();
        J_lb_to_q(1,3) = landmark_minus_p(0)*( 2)*rotation.w()+landmark_minus_p(1)*( 2)*rotation.z()+landmark_minus_p(2)*(-2)*rotation.y();

        J_lb_to_q(2,0) = landmark_minus_p(0)*( 2)*rotation.y()+landmark_minus_p(1)*(-2)*rotation.x()+landmark_minus_p(2)*( 2)*rotation.w();
        J_lb_to_q(2,1) = landmark_minus_p(0)*(-2)*rotation.z()+landmark_minus_p(1)*( 2)*rotation.w()+landmark_minus_p(2)*( 2)*rotation.x();
        J_lb_to_q(2,2) = landmark_minus_p(0)*(-2)*rotation.w()+landmark_minus_p(1)*(-2)*rotation.z()+landmark_minus_p(2)*( 2)*rotation.y();
        J_lb_to_q(2,3) = landmark_minus_p(0)*(-2)*rotation.x()+landmark_minus_p(1)*(-2)*rotation.y()+landmark_minus_p(2)*(-2)*rotation.z();

        J0 = J_residual_to_lc * J_lc_to_lb * J_lb_to_q;
      }  


      // position
      if (jacobians[1] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J1(jacobians[1]);       

        J1 = J_residual_to_lc * J_lc_to_lb * (-1) * rotation.toRotationMatrix().transpose(); 

      }  


      // landmark
      if (jacobians[2] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J2(jacobians[2]);     

        J2 = J_residual_to_lc * J_lc_to_lb * rotation.toRotationMatrix().transpose();
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
  std::string typeInfo() const
  {
    return "ReprojectionError";
  }

 protected:

  // the measurement
  measurement_t measurement_; ///< The (2D) measurement.


  Eigen::Transform<double, 3, Eigen::Affine> T_bc_;
  double focal_;
  double principle_point_[2];
};

#endif /* INCLUDE_REPROJECTION_ERROR_H_ */
