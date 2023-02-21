/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#ifndef MSCKF_VIO_FEATURE_H
#define MSCKF_VIO_FEATURE_H

#include <iostream>
#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "math_utils.hpp"
#include "imu_state.h"
#include "cam_state.h"

namespace msckf_vio
{

/** 
 * @brief Feature Salient part of an image. Please refer
 *    to the Appendix of "A Multi-State Constraint Kalman
 *    Filter for Vision-aided Inertial Navigation" for how
 *    the 3d position of a feature is initialized.
 * 一个特征可以理解为一个三维点，由多帧观测到
 */
struct Feature
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    typedef long long int FeatureIDType;

    /**
     * @brief OptimizationConfig Configuration parameters
     *    for 3d feature position optimization.
     * 优化参数
     */
    struct OptimizationConfig
    {
        // 位移是否足够，用于判断点是否能做三角化
        double translation_threshold;
        // huber参数
        double huber_epsilon;
        // 修改量阈值，优化的每次迭代都会有更新量，这个量如果太小则表示与目标值接近
        double estimation_precision;
        // LM算法lambda的初始值
        double initial_damping;

        // 内外轮最大迭代次数
        int outer_loop_max_iteration;
        int inner_loop_max_iteration;

        OptimizationConfig()
            : translation_threshold(0.2),
            huber_epsilon(0.01),
            estimation_precision(5e-7),
            initial_damping(1e-3),
            outer_loop_max_iteration(10),
            inner_loop_max_iteration(10)
        {
            return;
        }
    };

    // Constructors for the struct.
    Feature() 
        : id(0), position(Eigen::Vector3d::Zero()),
        is_initialized(false) {}

    Feature(const FeatureIDType &new_id)
        : id(new_id),
        position(Eigen::Vector3d::Zero()),
        is_initialized(false) {}

    /**
     * @brief cost Compute the cost of the camera observations
     * @param T_c0_c1 A rigid body transformation takes
     *    a vector in c0 frame to ci frame.
     * @param x The current estimation.
     * @param z The ith measurement of the feature j in ci frame.
     * @return e The cost of this observation.
     */
    inline void cost(
        const Eigen::Isometry3d &T_c0_ci,
        const Eigen::Vector3d &x, const Eigen::Vector2d &z,
        double &e) const;

    /**
     * @brief jacobian Compute the Jacobian of the camera observation
     * @param T_c0_c1 A rigid body transformation takes
     *    a vector in c0 frame to ci frame.
     * @param x The current estimation.
     * @param z The actual measurement of the feature in ci frame.
     * @return J The computed Jacobian.
     * @return r The computed residual.
     * @return w Weight induced by huber kernel.
     */
    inline void jacobian(
        const Eigen::Isometry3d &T_c0_ci,
        const Eigen::Vector3d &x, const Eigen::Vector2d &z,
        Eigen::Matrix<double, 2, 3> &J, Eigen::Vector2d &r,
        double &w) const;

    /**
     * @brief generateInitialGuess Compute the initial guess of
     *    the feature's 3d position using only two views.
     * @param T_c1_c2: A rigid body transformation taking
     *    a vector from c2 frame to c1 frame.
     * @param z1: feature observation in c1 frame.
     * @param z2: feature observation in c2 frame.
     * @return p: Computed feature position in c1 frame.
     */
    inline void generateInitialGuess(
        const Eigen::Isometry3d &T_c1_c2, const Eigen::Vector2d &z1,
        const Eigen::Vector2d &z2, Eigen::Vector3d &p) const;

    /**
     * @brief checkMotion Check the input camera poses to ensure
     *    there is enough translation to triangulate the feature
     *    positon.供外部调用
     * @param cam_states : input camera poses.
     * @return True if the translation between the input camera
     *    poses is sufficient.
     */
    inline bool checkMotion(
        const CamStateServer &cam_states) const;

    /**
     * @brief InitializePosition Intialize the feature position
     *    based on all current available measurements.供外部调用
     * @param cam_states: A map containing the camera poses with its
     *    ID as the associated key value.
     * @return The computed 3d position is used to set the position
     *    member variable. Note the resulted position is in world
     *    frame.
     * @return True if the estimated 3d position of the feature
     *    is valid.
     */
    inline bool initializePosition(
        const CamStateServer &cam_states);

    // An unique identifier for the feature.
    // In case of long time running, the variable
    // type of id is set to FeatureIDType in order
    // to avoid duplication.
    FeatureIDType id;

    // id for next feature
    static FeatureIDType next_id;

    // Store the observations of the features in the
    // state_id(key)-image_coordinates(value) manner.
    // 升序排序的map
    std::map<StateIDType, Eigen::Vector4d, std::less<StateIDType>,
            Eigen::aligned_allocator<std::pair<const StateIDType, Eigen::Vector4d>>>
        observations;

    // 3d postion of the feature in the world frame.
    Eigen::Vector3d position;

    // A indicator to show if the 3d postion of the feature
    // has been initialized or not.
    bool is_initialized;

    // Noise for a normalized feature measurement.
    static double observation_noise;

    // Optimization configuration for solving the 3d position.
    static OptimizationConfig optimization_config;
};

typedef Feature::FeatureIDType FeatureIDType;
typedef std::map<FeatureIDType, Feature, std::less<int>,
        Eigen::aligned_allocator<std::pair<const FeatureIDType, Feature>>>
    MapServer;

/**
 * @brief checkMotion 查看是否有足够视差.供外部调用
 * @param cam_states 所有参与计算的相机位姿
 * @return 是否有足够视差
 */
bool Feature::checkMotion(
    const CamStateServer &cam_states) const
{
    // 1. 取出对应的始末帧id
    const StateIDType &first_cam_id = observations.begin()->first;
    const StateIDType &last_cam_id = (--observations.end())->first;

    // 2. 分别赋值位姿
    Eigen::Isometry3d first_cam_pose;
    // Rwc
    first_cam_pose.linear() =
        quaternionToRotation(cam_states.find(first_cam_id)->second.orientation).transpose();
    
    // twc
    first_cam_pose.translation() =
        cam_states.find(first_cam_id)->second.position;

    Eigen::Isometry3d last_cam_pose;
    // Rwc
    last_cam_pose.linear() =
        quaternionToRotation(cam_states.find(last_cam_id)->second.orientation).transpose();
    // twc
    last_cam_pose.translation() =
        cam_states.find(last_cam_id)->second.position;

    // Get the direction of the feature when it is first observed.
    // This direction is represented in the world frame.
    // 3. 求出投影射线在世界坐标系啊下的方向
    // 在第一帧的左相机上的归一化坐标
    Eigen::Vector3d feature_direction(
        observations.begin()->second(0),
        observations.begin()->second(1), 1.0);
    // 转到世界坐标系下，求出了这个射线的方向
    feature_direction = feature_direction / feature_direction.norm();
    feature_direction = first_cam_pose.linear() * feature_direction;

    // Compute the translation between the first frame
    // and the last frame. We assume the first frame and
    // the last frame will provide the largest motion to
    // speed up the checking process.
    // 4. 求出始末两帧在世界坐标系下的位移（这段判断非常精彩！！！！）
    // 始指向末的向量
    Eigen::Vector3d translation =
        last_cam_pose.translation() - first_cam_pose.translation();

    // 这里相当于两个向量点乘 这个结果等于夹角的cos值乘上位移的模
    // 也相当于translation 在 feature_direction上的投影
    // 其实就是translation在feature_direction方向上的长度
    double parallel_translation =
        translation.transpose() * feature_direction;

    // 这块直接理解比较抽象，使用带入法，分别带入 0° 180° 跟90°
    // 当两个向量的方向相同 0°，这个值是0
    // 两个方向相反时 180°，这个值也是0
    // 90°时， cos为0，也就是看translation是否足够大
    // 所以这块的判断即考虑了角度，同时考虑了位移。即使90°但是位移不够也不做三角化
    Eigen::Vector3d orthogonal_translation =
        translation - parallel_translation * feature_direction;

    if (orthogonal_translation.norm() >
        optimization_config.translation_threshold)
        return true;
    else
        return false;
}

/**
 * @brief initializePosition 三角化+LM优化
 * @param cam_states 所有参与计算的相机位姿
 * @return 是否三角化成功
 */
bool Feature::initializePosition(
    const CamStateServer &cam_states)
{
    // Organize camera poses and feature observations properly.
    // 存放每个观测以及每个对应相机的pos，注意这块是左右目独立存放
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> cam_poses(0);
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> measurements(0);

    // 1. 准备数据
    for (auto &m : observations)
    {
        // TODO: This should be handled properly. Normally, the
        //    required camera states should all be available in
        //    the input cam_states buffer.
        auto cam_state_iter = cam_states.find(m.first);
        if (cam_state_iter == cam_states.end())
            continue;

        // Add the measurement.
        // 1.1 添加归一化坐标
        measurements.push_back(m.second.head<2>());
        measurements.push_back(m.second.tail<2>());

        // This camera pose will take a vector from this camera frame
        // to the world frame.
        // Twc
        Eigen::Isometry3d cam0_pose;
        cam0_pose.linear() =
            quaternionToRotation(cam_state_iter->second.orientation).transpose();
        cam0_pose.translation() = cam_state_iter->second.position;

        Eigen::Isometry3d cam1_pose;
        cam1_pose = cam0_pose * CAMState::T_cam0_cam1.inverse();

        // 1.2 添加相机位姿
        cam_poses.push_back(cam0_pose);
        cam_poses.push_back(cam1_pose);
    }

    // All camera poses should be modified such that it takes a
    // vector from the first camera frame in the buffer to this
    // camera frame.
    // 2. 中心化位姿，提高计算精度
    Eigen::Isometry3d T_c0_w = cam_poses[0];
    for (auto &pose : cam_poses)
        pose = pose.inverse() * T_c0_w;

    // Generate initial guess
    // 3. 使用首末位姿粗略计算出一个三维点坐标
    Eigen::Vector3d initial_position(0.0, 0.0, 0.0);
    generateInitialGuess(
        cam_poses[cam_poses.size() - 1], measurements[0],
        measurements[measurements.size() - 1], initial_position);
    // 弄成逆深度形式
    Eigen::Vector3d solution(
        initial_position(0) / initial_position(2),
        initial_position(1) / initial_position(2),
        1.0 / initial_position(2));

    // Apply Levenberg-Marquart method to solve for the 3d position.
    double lambda = optimization_config.initial_damping;
    int inner_loop_cntr = 0;
    int outer_loop_cntr = 0;
    bool is_cost_reduced = false;
    double delta_norm = 0;

    // Compute the initial cost.
    // 4. 利用初计算的点计算在各个相机下的误差，作为初始误差
    double total_cost = 0.0;
    for (int i = 0; i < cam_poses.size(); ++i)
    {
        double this_cost = 0.0;
        // 计算投影误差（归一化坐标）
        cost(cam_poses[i], solution, measurements[i], this_cost);
        total_cost += this_cost;
    }

    // Outer loop.
    // 5. LM优化开始， 优化三维点坐标，不优化位姿，比较简单
    do
    {
        // A是  J^t * J  B是 J^t * r
        // 可能有同学疑问自己当初学的时候是 -J^t * r
        // 这个无所谓，因为这里是负的更新就是正的，而这里是正的，所以更新是负的
        // 总之就是有一个是负的，总不能误差越来越大吧
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();

        // 5.1 计算AB矩阵
        for (int i = 0; i < cam_poses.size(); ++i)
        {
            Eigen::Matrix<double, 2, 3> J;
            Eigen::Vector2d r;
            double w;

            // 计算一目相机观测的雅可比与误差
            // J 归一化坐标误差相对于三维点的雅可比
            // r
            // w 权重，同信息矩阵
            jacobian(cam_poses[i], solution, measurements[i], J, r, w);

            // 鲁棒核约束
            if (w == 1)
            {
                A += J.transpose() * J;
                b += J.transpose() * r;
            }
            else
            {
                double w_square = w * w;
                A += w_square * J.transpose() * J;
                b += w_square * J.transpose() * r;
            }
        }

        // Inner loop.
        // Solve for the delta that can reduce the total cost.
        // 这层是在同一个雅可比下多次迭代，如果效果不好说明需要调整阻尼因子了，因为线性化不是很好
        // 如果多次一直误差不下降，退出循环重新计算雅可比
        do
        {
            // LM算法中的lambda
            Eigen::Matrix3d damper = lambda * Eigen::Matrix3d::Identity();
            Eigen::Vector3d delta = (A + damper).ldlt().solve(b);
            // 更新
            Eigen::Vector3d new_solution = solution - delta;
            // 统计本次修改量的大小，如果太小表示接近目标值或者陷入局部极小值，那么就没必要继续了
            delta_norm = delta.norm();

            // 计算更新后的误差
            double new_cost = 0.0;
            for (int i = 0; i < cam_poses.size(); ++i)
            {
                double this_cost = 0.0;
                cost(cam_poses[i], new_solution, measurements[i], this_cost);
                new_cost += this_cost;
            }

            // 如果更新后误差比之前小，说明确实是往好方向发展
            // 我们高斯牛顿的JtJ比较接近真实情况所以减少阻尼，增大步长，delta变大，加快收敛
            if (new_cost < total_cost)
            {
                is_cost_reduced = true;
                solution = new_solution;
                total_cost = new_cost;
                lambda = lambda / 10 > 1e-10 ? lambda / 10 : 1e-10;
            }
            // 如果不行，那么不要这次迭代的结果
            // 说明高斯牛顿的JtJ不接近二阶的海森矩阵
            // 那么增大阻尼，减小步长，delta变小
            // 并且算法接近一阶的最速下降法
            else
            {
                
                is_cost_reduced = false;
                lambda = lambda * 10 < 1e12 ? lambda * 10 : 1e12;
            }

        } while (inner_loop_cntr++ <
                        optimization_config.inner_loop_max_iteration &&
                    !is_cost_reduced);

        inner_loop_cntr = 0;

    // 直到迭代次数到了或者更新量足够小了
    } while (outer_loop_cntr++ <
                    optimization_config.outer_loop_max_iteration &&
                delta_norm > optimization_config.estimation_precision);

    // Covert the feature position from inverse depth
    // representation to its 3d coordinate.
    // 取出最后的结果
    Eigen::Vector3d final_position(solution(0) / solution(2),
                                    solution(1) / solution(2), 1.0 / solution(2));

    // Check if the solution is valid. Make sure the feature
    // is in front of every camera frame observing it.
    // 6. 深度验证
    bool is_valid_solution = true;
    for (const auto &pose : cam_poses)
    {
        Eigen::Vector3d position =
            pose.linear() * final_position + pose.translation();
        if (position(2) <= 0)
        {
            is_valid_solution = false;
            break;
        }
    }

    // 7. 更新结果
    // Convert the feature position to the world frame.
    position = T_c0_w.linear() * final_position + T_c0_w.translation();

    if (is_valid_solution)
        is_initialized = true;

    return is_valid_solution;
}

/**
 * @brief cost 计算投影误差（归一化坐标）
 * @param T_c0_ci 相对位姿，Tcic0 每一个单向机到第一个观测的相机的T
 * @param x 三维点坐标(x/z, y/z, 1/z)
 * @param z ci下的观测归一化坐标
 * @param e 误差
 */
void Feature::cost(
    const Eigen::Isometry3d &T_c0_ci,
    const Eigen::Vector3d &x, const Eigen::Vector2d &z,
    double &e) const
{
    // Compute hi1, hi2, and hi3 as Equation (37).
    const double &alpha = x(0);
    const double &beta = x(1);
    const double &rho = x(2);

    // h 等于  (R * P + t) * 1/Pz
    // h1    | R11 R12 R13    alpha / rho       t1    |
    // h2 =  | R21 R22 R23 *  beta  / rho   +   t2    |   *  rho
    // h3    | R31 R32 R33    1 / rho           t3    |
    Eigen::Vector3d h =
        T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) +
        rho * T_c0_ci.translation();
    double &h1 = h(0);
    double &h2 = h(1);
    double &h3 = h(2);

    // Predict the feature observation in ci frame.
    // 求出在另一个相机ci下的归一化坐标
    Eigen::Vector2d z_hat(h1 / h3, h2 / h3);

    // Compute the residual.
    // 两个归一化坐标算误差
    e = (z_hat - z).squaredNorm();
    return;
}

/**
 * @brief jacobian 求一个观测对应的雅可比
 * @param T_c0_ci 相对位姿，Tcic0 每一个单向机到第一个观测的相机的T
 * @param x 三维点坐标(x/z, y/z, 1/z)
 * @param z 归一化坐标
 * @param J 雅可比 归一化坐标误差相对于三维点的
 * @param r 误差
 * @param w 权重
 * @return 是否三角化成功
 */
void Feature::jacobian(
    const Eigen::Isometry3d &T_c0_ci,
    const Eigen::Vector3d &x, const Eigen::Vector2d &z,
    Eigen::Matrix<double, 2, 3> &J, Eigen::Vector2d &r,
    double &w) const
{

    // Compute hi1, hi2, and hi3 as Equation (37).
    const double &alpha = x(0);  // x/z
    const double &beta = x(1);  // y/z
    const double &rho = x(2);  // 1/z

    // h 等于 (R * P + t) * 1/Pz
    // h1    | R11 R12 R13    alpha / rho       t1    |
    // h2 =  | R21 R22 R23 *  beta  / rho   +   t2    |   *  rho
    // h3    | R31 R32 R33    1 / rho           t3    |
    Eigen::Vector3d h = T_c0_ci.linear() * Eigen::Vector3d(alpha, beta, 1.0) +
                        rho * T_c0_ci.translation();
    double &h1 = h(0);
    double &h2 = h(1);
    double &h3 = h(2);

    // Compute the Jacobian.
    // 首先明确一下误差与三维点的关系
    // 下面的r是误差 r = z_hat - z;  Eigen::Vector2d z_hat(h1 / h3, h2 / h3)
    // 我们要求r对三维点的雅可比，其中z是观测，与三维点坐标无关
    // 因此相当于求归一化坐标相对于 alpha beta rho的雅可比，首先要求出他们之间的关系
    // 归一化坐标设为x y
    // x = h1/h3 y = h2/h3
    // 先写出h与alpha beta rho的关系，也就是上面写的
    // 然后求h1 相对于alpha beta rho的导数 再求h2 的  在求 h3 的
    // R11 R12 t1
    // R21 R22 t2
    // R31 R32 t3
    // 链式求导法则
    // ∂x/∂alpha = ∂x/∂h1 * ∂h1/∂alpha + ∂x/∂h3 * ∂h3/∂alpha
    // ∂x/∂h1 = 1/h3       ∂h1/∂alpha = R11
    // ∂x/∂h3 = -h1/h3^2   ∂h3/∂alpha = R31
    // 剩下的就好求了
    Eigen::Matrix3d W;
    W.leftCols<2>() = T_c0_ci.linear().leftCols<2>();
    W.rightCols<1>() = T_c0_ci.translation();

    // h1 / h3 相对于 alpha beta rho的
    J.row(0) = 1 / h3 * W.row(0) - h1 / (h3 * h3) * W.row(2);
    // h1 / h3 相对于 alpha beta rho的
    J.row(1) = 1 / h3 * W.row(1) - h2 / (h3 * h3) * W.row(2);

    // Compute the residual.
    // 求取误差
    Eigen::Vector2d z_hat(h1 / h3, h2 / h3);
    r = z_hat - z;

    // Compute the weight based on the residual.
    // 使用鲁棒核函数约束
    double e = r.norm();
    if (e <= optimization_config.huber_epsilon)
        w = 1.0;
    // 如果误差大于optimization_config.huber_epsilon但是没超过他的2倍，那么会放大权重w>1
    // 如果误差大的离谱，超过他的2倍，缩小他的权重
    else
        w = std::sqrt(2.0 * optimization_config.huber_epsilon / e);

    return;
}

/**
 * @brief generateInitialGuess 两帧做一次三角化
 * @param T_c1_c2 两帧间的相对位姿
 * @param z1 观测1
 * @param z2 观测2 都是归一化坐标
 * @param p 三位点坐标
 */
void Feature::generateInitialGuess(
    const Eigen::Isometry3d &T_c1_c2, const Eigen::Vector2d &z1,
    const Eigen::Vector2d &z2, Eigen::Vector3d &p) const
{
    // 列出方程
    // P2 = R21 * P1 + t21  下面省略21
    // 两边左乘P2的反对称矩阵
    // P2^ * (R * P1 + t) = 0
    // 其中左右可以除以P2的深度，这样P2就成了z2，且P1可以分成z1（归一化平面） 乘上我们要求的深度d
    // 令m = R * z1
    // | 0   -1   z2y |    ( | m0 |         )
    // | 1    0  -z2x | *  ( | m1 | * d + t )  =  0
    // |-z2y z2x   0  |    ( | m2 |         )
    // 会发现这三行里面两行是线性相关的，所以只取前两行
    // Construct a least square problem to solve the depth.
    Eigen::Vector3d m = T_c1_c2.linear() * Eigen::Vector3d(z1(0), z1(1), 1.0);

    Eigen::Vector2d A(0.0, 0.0);
    A(0) = m(0) - z2(0) * m(2);  // 对应第二行

    // 按照上面推导这里应该是负的但是不影响，因为我们下边b(1)也给取负了
    A(1) = m(1) - z2(1) * m(2);  // 对应第一行

    Eigen::Vector2d b(0.0, 0.0);
    b(0) = z2(0) * T_c1_c2.translation()(2) - T_c1_c2.translation()(0);
    b(1) = z2(1) * T_c1_c2.translation()(2) - T_c1_c2.translation()(1);

    // Solve for the depth.
    // 解方程得出p1的深度值
    double depth = (A.transpose() * A).inverse() * A.transpose() * b;
    p(0) = z1(0) * depth;
    p(1) = z1(1) * depth;
    p(2) = depth;
    return;
}


} // namespace msckf_vio

#endif // MSCKF_VIO_FEATURE_H
