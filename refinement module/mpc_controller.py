import numpy as np
from casadi import SX, vertcat, cos, sin, tan, Function, Opti, log, exp


class MPCObstacleAvoidance:
    def __init__(self, config):
        self.config = config  # 配置参数
        self.setup_optimizer()  # 初始化优化器

    def setup_optimizer(self):
        # 状态变量：x(位置), y(位置), theta(航向角), v(速度), delta(转向角)
        x, y, theta, v, delta = SX.sym('x'), SX.sym('y'), SX.sym('theta'), SX.sym('v'), SX.sym('delta')

        # 控制变量：a(加速度), delta_dot(转向角速率)
        a, delta_dot = SX.sym('a'), SX.sym('delta_dot')
        
        self.states = vertcat(x, y, theta, v, delta)  # 状态向量
        self.controls = vertcat(a, delta_dot)         # 控制向量

        # 状态更新方程：dx/dt = f(x,u)，离散化后为x_{k+1} = x_k + f(x_k,u_k)*dt
        states_dot = vertcat(
            v * cos(theta),                  # x方向速度：v*cos(theta)
            v * sin(theta),                  # y方向速度：v*sin(theta)
            v * tan(delta) / self.config.L,  # 航向角变化率：v*tan(delta)/L
            a,                               # 速度变化率：加速度a
            delta_dot                         # 转向角变化率：转向速率delta_dot
        )
        # 离散化动力学模型（前向欧拉法）
        self.dynamics = Function('dynamics', [self.states, self.controls], 
                                [self.states + states_dot * self.config.dt])

      
        self.opti = Opti()  # CasADi优化器实例

        # 预测时域内的状态和控制序列
        self.predicted_states = self.opti.variable(5, self.config.N + 1)  # [状态维度, 时域长度+1]
        self.controls_seq = self.opti.variable(2, self.config.N)          # [控制维度, 时域长度]

        # 优化问题参数
        self.initial_state = self.opti.parameter(5)          # 初始状态
        self.reference_trajectory = self.opti.parameter(5, self.config.N + 1)  # 参考轨迹
        self.obstacles = self.opti.parameter(7 * self.config.max_obs, self.config.N + 1)  # 障碍物信息

     
        self.cost = 0# 总成本

        # 轨迹跟踪成本 + 控制平滑成本 + 约束惩罚（逐时间步）
        for k in range(self.config.N):
            # 当前状态与参考状态
            state = self.predicted_states[:, k]
            ref_state = self.reference_trajectory[:, k]

            # 轨迹跟踪误差成本
            self.cost += self.config.w_x * (state[0] - ref_state[0])**2       # x位置误差
            self.cost += self.config.w_y * (state[1] - ref_state[1])**2       # y位置误差
            self.cost += self.config.w_theta * (state[2] - ref_state[2])**2   # 航向角误差
            self.cost += self.config.w_v * (state[3] - ref_state[3])**2       # 速度误差

            # 控制平滑成本（抑制剧烈控制）
            control = self.controls_seq[:, k]
            self.cost += self.config.w_a * control[0]**2             # 加速度平方项
            self.cost += self.config.w_delta_dot * control[1]**2     # 转向速率平方项

            # 道路边界惩罚（使用平滑最大函数近似硬约束）
            y_pos = state[1]
            boundary_penalty_left = self.smooth_max(self.config.y_min - y_pos)  # 左侧越界惩罚
            boundary_penalty_right = self.smooth_max(y_pos - self.config.y_max) # 右侧越界惩罚
            self.cost += self.config.w_bound * (boundary_penalty_left**2 + boundary_penalty_right**2)

            # 障碍物规避惩罚（对每个障碍物）
            for obs_idx in range(self.config.max_obs):
                # 障碍物参数：[x, y, heading, v, length, width, flag]（7个参数）
                obs_params = self.obstacles[obs_idx*7 : (obs_idx+1)*7, k]
                obs_x, obs_y, obs_heading = obs_params[0], obs_params[1], obs_params[2]
                obs_length, obs_width = obs_params[3], obs_params[4]
                obs_flag = obs_params[6]  # 障碍物激活标志（1表示激活）

                # 车辆与障碍物的相对位置（转换到障碍物局部坐标系）
                dx = state[0] - obs_x
                dy = state[1] - obs_y
                # 旋转到障碍物航向角坐标系（使障碍物对齐自身坐标系）
                rot_dx = dx * cos(obs_heading) + dy * sin(obs_heading)
                rot_dy = -dx * sin(obs_heading) + dy * cos(obs_heading)

                # 安全距离计算（归一化到障碍物尺寸）
                margin_x = obs_length/2 + self.config.vehicle_length/2 + self.config.safety_distance
                margin_y = obs_width/2 + self.config.vehicle_width/2 + self.config.safety_distance
                norm_dist = (rot_dx/margin_x)**2 + (rot_dy/margin_y)** 2# 归一化距离平方

                # 距离惩罚：当车辆进入安全边界内时（norm_dist < 1），惩罚随距离减小而增大
                obstacle_penalty = exp(-self.config.obstacle_decay_rate * (norm_dist - 1))
                # 仅对激活的障碍物施加惩罚
                self.cost += self.config.w_obstacle * self.smooth_max(obs_flag - 0.5) * obstacle_penalty

        # 终端成本（强调对终点的跟踪精度）
        final_state = self.predicted_states[:, -1]
        final_ref = self.reference_trajectory[:, -1]
        self.cost += self.config.w_fx * (final_state[0] - final_ref[0])**2       # 终端x误差
        self.cost += self.config.w_fy * (final_state[1] - final_ref[1])**2       # 终端y误差
        self.cost += self.config.w_ftheta * (final_state[2] - final_ref[2])**2   # 终端航向角误差
        self.cost += self.config.w_fv * (final_state[3] - final_ref[3])**2       # 终端速度误差

        # 设置优化目标：最小化总成本
        self.opti.minimize(self.cost)

       
        # 初始状态约束
        self.opti.subject_to(self.predicted_states[:, 0] == self.initial_state)

        # 动力学约束（状态转移方程）
        for k in range(self.config.N):
            next_state = self.dynamics(
                self.predicted_states[:, k], 
                self.controls_seq[:, k]
            )
            self.opti.subject_to(self.predicted_states[:, k+1] == next_state)

        # 状态约束（速度、转向角）
        for k in range(self.config.N + 1):
            self.opti.subject_to(self.predicted_states[3, k] >= self.config.v_min)  # 最小速度
            self.opti.subject_to(self.predicted_states[3, k] <= self.config.v_max)  # 最大速度
            self.opti.subject_to(self.predicted_states[4, k] >= self.config.delta_min)  # 最小转向角
            self.opti.subject_to(self.predicted_states[4, k] <= self.config.delta_max)  # 最大转向角

        # 控制约束（加速度、转向速率）
        for k in range(self.config.N):
            self.opti.subject_to(self.controls_seq[0, k] >= self.config.a_min)  # 最小加速度（最大刹车）
            self.opti.subject_to(self.controls_seq[0, k] <= self.config.a_max)  # 最大加速度
            self.opti.subject_to(self.controls_seq[1, k] >= -self.config.delta_dot_max)  # 最小转向速率
            self.opti.subject_to(self.controls_seq[1, k] <= self.config.delta_dot_max)  # 最大转向速率

        # 航向角变化率约束（避免剧烈转向）
        for k in range(1, self.config.N + 1):
            d_theta = self.predicted_states[2, k] - self.predicted_states[2, k-1]  # 航向角变化量
            self.opti.subject_to(d_theta <= self.config.theta_dot_max * self.config.dt)  # 最大变化
            self.opti.subject_to(d_theta >= -self.config.theta_dot_max * self.config.dt) # 最小变化

        # IPOPT求解器参数
        p_opts = {"expand": True} 
        s_opts = {
            "max_iter": self.config.max_iter,  # 最大迭代次数
            "print_level": 0,  
            "acceptable_tol": 1e-4# 可接受的收敛精度
        }
        self.opti.solver('ipopt', p_opts, s_opts)

    def smooth_max(self, z, beta=None):
        if beta isNone:
            beta = self.config.obstacle_smooth_beta
        return (1 / beta) * log(1 + exp(beta * z))

    def solve(self, current_state, ref_traj, obstacles):
        # 设置优化问题参数
        self.opti.set_value(self.initial_state, current_state)
        self.opti.set_value(self.reference_trajectory, ref_traj)

        # 处理障碍物信息（预测时域内的位置）
        obs_array = np.zeros((7 * self.config.max_obs, self.config.N + 1))  # 初始化障碍物数组
        for obs_idx, obs in enumerate(obstacles):
            if obs_idx >= self.config.max_obs:
                break# 不超过最大障碍物数量
            
            # 障碍物参数解析
            x0, y0, heading, v, length, width, flag = obs
            vx = v * np.cos(heading)  # 障碍物x方向速度
            vy = v * np.sin(heading)  # 障碍物y方向速度
            
            # 预测障碍物在每个时间步的位置
            for k in range(self.config.N + 1):
                t = self.config.dt * k  # 第k步的时间
                obs_array[obs_idx*7 + 0, k] = x0 + vx * t  # x坐标
                obs_array[obs_idx*7 + 1, k] = y0 + vy * t  # y坐标
                obs_array[obs_idx*7 + 2, k] = heading      # 航向角（假设不变）
                obs_array[obs_idx*7 + 3, k] = length       # 长度
                obs_array[obs_idx*7 + 4, k] = width        # 宽度
                obs_array[obs_idx*7 + 6, k] = flag         # 激活标志
        self.opti.set_value(self.obstacles, obs_array)

        # 求解优化问题
        try:
            solution = self.opti.solve()
            return solution.value(self.controls_seq), solution.value(self.predicted_states)
        except Exception as e:
            print(f"MPC求解失败: {e}")
            # 求解失败时返回安全控制
            safe_controls = np.zeros((2, self.config.N))
            safe_controls[0] = -1.0# 恒定刹车
            safe_states = np.tile(current_state, (self.config.N + 1, 1)).T  
            return safe_controls, safe_states