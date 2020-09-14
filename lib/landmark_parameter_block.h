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
 *  Created on: Aug 30, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file HomogeneousPointParameterBlock.hpp
 * @brief Header file for the HomogeneousPointParameterBlock class.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_LANDMARK_PARAMETERBLOCK_H_
#define INCLUDE_LANDMARK_PARAMETERBLOCK_H_

#include <Eigen/Core>

#include "sized_parameter_block.h"
// #include "LandmarkLocalParameterization.h"


class LandmarkParameterBlock: public SizedParameterBlock<3, 3, Eigen::Vector3d> {
 public:

  /// \brief The estimate type (3D vector).
  typedef Eigen::Vector3d estimate_t;

  /// \brief The base class type.
  typedef SizedParameterBlock<3, 3, estimate_t> base_t;

  /// \brief Default constructor (assumes not fixed).
  LandmarkParameterBlock(): base_t::SizedParameterBlock(),
    initialized_(false) {
    setFixed(false);
  }


  /// \brief Constructor with estimate and time.
  /// @param[in] point The homogeneous point estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] initialized Whether or not the 3d position is considered initialised.
  LandmarkParameterBlock(const Eigen::Vector3d& point, bool initialized = true) {
    setEstimate(point);
    setInitialized(initialized);
    setFixed(false);
  }

  /// \brief Trivial destructor.
  ~LandmarkParameterBlock() {};

  /// @name Setters
  /// @{

  /// @brief Set estimate of this parameter block.
  /// @param[in] point The estimate to set this to.
  void setEstimate(const Eigen::Vector3d& point) {
    // hack: only do "Euclidean" points for now...
    for (int i = 0; i < base_t::Dimension; ++i)
      parameters_[i] = point[i];
  }

  /// \brief Set initialisaiton status.
  /// @param[in] initialized Whether or not the 3d position is considered initialised.
  void setInitialized(bool initialized) {
    initialized_ = initialized;
  }

  /// @}

  /// @name Getters
  /// @{

  /// @brief Get estimate
  /// \return The estimate.
  Eigen::Vector3d estimate() const {
    return Eigen::Vector3d(parameters_[0], parameters_[1], parameters_[2]);
  }

  /// \brief Get initialisaiton status.
  /// \return Whether or not the 3d position is considered initialised.
  bool initialized() const {
    return initialized_;
  }


  /// @brief Return parameter block type as string
  std::string typeInfo() const {
    return "LandmarkParameterBlock";
  }

 private:
  bool initialized_;  ///< Whether or not the 3d position is considered initialised.

};

#endif /* INCLUDE_LANDMAKRPARAMETERBLOCK_H_ */
