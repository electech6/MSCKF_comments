/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_CAM_STATE_H
#define MSCKF_VIO_CAM_STATE_H

#include <map>
#include <vector>
#include <Eigen/Dense>

#include "imu_state.h"

namespace msckf_vio
{
/*
    * @brief CAMState Stored camera state in order to
    *    form measurement model.
    */
struct CAMState
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // An unique identifier for the CAM state.
    StateIDType id;

    // Time when the state is recorded
    double time;

    // Orientation
    // Take a vector from the world frame to the camera frame.
    Eigen::Vector4d orientation;

    // Position of the camera frame in the world frame.
    Eigen::Vector3d position;

    // These two variables should have the same physical
    // interpretation with `orientation` and `position`.
    // There two variables are used to modify the measurement
    // Jacobian matrices to make the observability matrix
    // have proper null space.
    // 使可观测性矩阵具有适当的零空间的旋转平移
    Eigen::Vector4d orientation_null;
    Eigen::Vector3d position_null;

    // Takes a vector from the cam0 frame to the cam1 frame.
    // 两个相机间的外参
    static Eigen::Isometry3d T_cam0_cam1;

    CAMState() 
    : id(0), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d(0, 0, 0))
    {

    }

    CAMState(const StateIDType &new_id)
    : id(new_id), time(0),
    orientation(Eigen::Vector4d(0, 0, 0, 1)),
    position(Eigen::Vector3d::Zero()),
    orientation_null(Eigen::Vector4d(0, 0, 0, 1)),
    position_null(Eigen::Vector3d::Zero())
    {

    }
};

// 别看他长，其实就是一个map类
// key是 StateIDType 由 long long int typedef而来，把它当作int看就行
// value是CAMState
typedef std::map<StateIDType, CAMState,
    std::less<int>, Eigen::aligned_allocator< std::pair<const StateIDType, CAMState>>>
    CamStateServer;
} // namespace msckf_vio

#endif // MSCKF_VIO_CAM_STATE_H
