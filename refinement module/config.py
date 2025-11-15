import numpy as np

class Config:
    def __init__(self):
        # -------------------------- MPC核心参数 --------------------------
        self.N = 15               # 预测时域长度（步数）
        self.dt = 0.2             # 控制时间步长 [s]
        self.max_obs = 5          # 最大障碍物数量（预留扩展空间）

        # -------------------------- 车辆物理参数 --------------------------
        self.vehicle_length = 2.0# 车辆长度 [m]
        self.vehicle_width = 1.0   # 车辆宽度 [m]
        self.safety_distance = 0.0# 安全距离 [m]
        self.L = 1.5               # 轴距（前轴到后轴距离）[m]

        # -------------------------- 车辆运动约束 --------------------------
        self.v_max = 10.0               # 最大速度 [m/s]
        self.v_min = 0.0                # 最小速度 [m/s]
        self.a_max = 2.0                # 最大加速度 [m/s²]
        self.a_min = -3.0               # 最大减速度 [m/s²]
        self.delta_max = np.deg2rad(30) # 最大转向角 [rad]
        self.delta_min = -np.deg2rad(30)# 最小转向角 [rad]
        self.delta_dot_max = np.deg2rad(20) # 最大转向速率 [rad/s]
        self.theta_dot_max = np.deg2rad(45) # 最大航向角变化率 [rad/s]

        # -------------------------- 道路与场景参数 --------------------------
        self.road_length = 20.0   # 道路总长度 [m]
        self.y_min = -2.5         # 道路左侧边界（y坐标）[m]
        self.y_max = 2.5          # 道路右侧边界（y坐标）[m]
        self.start_pos = np.array([1.0, 0.0, 0.0, 1.0]) # 起点状态 [x, y, θ, v]
        self.goal_pos = np.array([18.5, 0.0, 0.0, 0.0]) # 目标状态 [x, y, θ, v]

        # -------------------------- 可视化参数 --------------------------
        self.xlim = (-0.1, self.road_length+0.1)  # X轴显示范围
        self.ylim = (self.y_min-0.1, self.y_max+0.1) # Y轴显示范围
        self.animation_interval = 100# 动画刷新间隔 [ms]

        # -------------------------- 成本函数权重 --------------------------
        self.w_x = 1.0         # x位置误差权重
        self.w_y = 1.0         # y位置误差权重
        self.w_theta = 1.0     # 航向角误差权重
        self.w_v = 0.1         # 速度误差权重
        self.w_a = 0.1         # 加速度权重
        self.w_delta_dot = 0.1# 转向速率权重
        self.w_bound = 50.0    # 道路边界惩罚权重
        self.w_obstacle = 20.0# 障碍物规避权重
        self.w_fx = 10.0       # 终端x位置误差权重
        self.w_fy = 10.0       # 终端y位置误差权重
        self.w_ftheta = 10.0   # 终端航向角误差权重
        self.w_fv = 0.5        # 终端速度误差权重

        # 障碍物惩罚参数
        self.obstacle_decay_rate = 4.0# 距离惩罚衰减速率（值越大，惩罚随距离衰减越快）
        self.obstacle_smooth_beta = 6.0# smooth_max函数的平滑参数（值越大越接近硬约束）

        # -------------------------- 仿真参数 --------------------------
        self.sim_time = 30.0# 仿真总时间 [s]
        self.ref_speed = 1.0# 参考速度 [m/s]
        self.goal_threshold = 0.2# 到达目标的判定阈值 [m]
        self.max_iter = 200    # MPC求解器最大迭代次数