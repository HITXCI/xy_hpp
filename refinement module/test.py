import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from config import Config
from mpc_controller import MPCObstacleAvoidance


def generate_reference_path(current_state, goal_state, N, dt, config):
    current_x = current_state[0]
    goal_x = goal_state[0]
    
    x_ref = np.linspace(current_x, goal_x, N + 1)
    y_ref = np.zeros(N + 1)
    theta_ref = np.zeros(N + 1)
    
    v_ref = np.zeros(N + 1)
    total_distance = goal_x - current_x  
    
    for k in range(N + 1):
        dist_to_goal = goal_x - x_ref[k]  
        
        # 距离目标较远时：加速到参考速度
        if dist_to_goal > 0.7 * total_distance:
            v_ref[k] = min(config.ref_speed, current_state[3] + config.a_max * k * dt)
        # 中间段：保持参考速度
        elif dist_to_goal > 0.3 * total_distance:
            v_ref[k] = config.ref_speed
        # 接近目标时：减速
        elif dist_to_goal > 3.0:
            v_ref[k] = config.ref_speed * (dist_to_goal / (0.3 * total_distance))
        # 非常接近目标：进一步减速
        else:
            v_ref[k] = max(0.5, config.ref_speed * (dist_to_goal / 3.0))
    
    # 转向角参考：0（直线行驶）
    delta_ref = np.zeros(N + 1)
    
    return np.vstack([x_ref, y_ref, theta_ref, v_ref, delta_ref])


def animate_trajectory(state_history, obstacles, mpc, config, n_steps):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(config.xlim)
    ax.set_ylim(config.ylim)
    ax.set_aspect('equal')
    ax.set_title("MPC Obstacle Avoidance (Bicycle Model)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    
    # 绘制道路边界和中心线
    ax.axhline(config.y_min, color='gray', linestyle='-', linewidth=2, label='Road Boundary')
    ax.axhline(config.y_max, color='gray', linestyle='-', linewidth=2)
    ax.axhline(0, color='g', linestyle='--', linewidth=2, alpha=0.5, label='Road Centerline')

    # 起点和终点标记
    ax.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=8, label='Start')
    ax.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=12, label='Goal')
    
    # 动画元素：实际轨迹
    path_line, = ax.plot([], [], 'b-', lw=1.5, alpha=0.7, label='Actual Path')
    # 车辆可视化（多边形）
    car_patch = plt.Polygon([[0, 0]]*4, closed=True, color='blue', alpha=0.6)
    ax.add_patch(car_patch)
    
    # 障碍物可视化
    obstacle_patches = []
    safety_boundaries = []  # 安全边界（椭圆）
    for obs in obstacles:
        if obs[6] < 0.5:
            continue# 跳过未激活的障碍物
        
        # 障碍物多边形
        x0, y0, heading, _, length, width, _ = obs
        R = np.array([[np.cos(heading), -np.sin(heading)],
                      [np.sin(heading), np.cos(heading)]])  # 旋转矩阵
        # 障碍物局部坐标系四角
        corners_local = np.array([
            [length/2, width/2], [length/2, -width/2],
            [-length/2, -width/2], [-length/2, width/2]
        ])
        corners_global = (R @ corners_local.T).T + np.array([x0, y0])  # 转换到全局坐标
        obs_patch = plt.Polygon(corners_global, closed=True, color='red', alpha=0.5, label='Obstacle')
        ax.add_patch(obs_patch)
        obstacle_patches.append(obs_patch)
        
        # 安全边界（椭圆：障碍物+车辆+安全距离）
        ellipse_width = length + config.vehicle_length + 2*config.safety_distance
        ellipse_height = width + config.vehicle_width + 2*config.safety_distance
        ellipse = Ellipse(
            (x0, y0), width=ellipse_width, height=ellipse_height,
            angle=np.rad2deg(heading), fill=False, edgecolor='purple',
            linestyle='--', linewidth=0.5, alpha=0.2, label='Safety Boundary'
        )
        ax.add_patch(ellipse)
        safety_boundaries.append(ellipse)

    # 预测轨迹可视化
    pred_line, = ax.plot([], [], 'm--', lw=1.5, alpha=0.6, label='Predicted Path')
    pred_points = ax.plot([], [], 'mo', markersize=3, alpha=0.6)[0]

    # 信息文本（显示速度、航向等）
    info_text = ax.text(0.02, 0.7, '', transform=ax.transAxes, fontsize=10)

    def get_vehicle_corners(x, y, theta):
        # 车辆局部坐标系四角（相对于中心）
        corners_local = np.array([
            [config.vehicle_length/2,  config.vehicle_width/2],  # 前右
            [config.vehicle_length/2, -config.vehicle_width/2],  # 前左
            [-config.vehicle_length/2, -config.vehicle_width/2], # 后左
            [-config.vehicle_length/2,  config.vehicle_width/2]  # 后右
        ])
        # 旋转到全局坐标系
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return (R @ corners_local.T).T + np.array([x, y])

    def init_animation():
        path_line.set_data([], [])
        car_patch.set_xy(np.zeros((4, 2)))
        pred_line.set_data([], [])
        pred_points.set_data([], [])
        info_text.set_text('')
        return [path_line, car_patch, pred_line, pred_points, info_text] + obstacle_patches + safety_boundaries

    def update_animation(frame):
        idx = min(frame, len(state_history)-1)
        current_state = state_history[idx]
        x, y, theta, v, delta = current_state
        
        # 更新实际轨迹
        path_line.set_data(state_history[:idx+1, 0], state_history[:idx+1, 1])
        
        # 更新车辆位置和姿态
        car_corners = get_vehicle_corners(x, y, theta)
        car_patch.set_xy(car_corners)
        
        # 更新信息文本
        info_text.set_text(
            f"Time: {idx*config.dt:.1f}s\n"
            f"Speed: {v:.2f}m/s\n"
            f"Heading: {np.rad2deg(theta):.1f}°\n"
            f"Steering: {np.rad2deg(delta):.1f}°"
        )
        
        # 更新障碍物位置（动态障碍物）
        for obs_idx, obs in enumerate(obstacles):
            if obs[6] < 0.5:
                continue# 未激活障碍物不更新
            
            x0, y0, heading, v_obs, length, width, _ = obs
            t = config.dt * idx  # 当前时间
            # 障碍物当前位置
            x_obs = x0 + np.cos(heading) * v_obs * t
            y_obs = y0 + np.sin(heading) * v_obs * t
            
            # 更新障碍物多边形
            R = np.array([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading), np.cos(heading)]])
            corners_local = np.array([
                [length/2, width/2], [length/2, -width/2],
                [-length/2, -width/2], [-length/2, width/2]
            ])
            corners_global = (R @ corners_local.T).T + np.array([x_obs, y_obs])
            obstacle_patches[obs_idx].set_xy(corners_global)
            
            # 更新安全边界位置
            safety_boundaries[obs_idx].center = (x_obs, y_obs)
        
        if frame % 5 == 0and idx < len(state_history)-1:
            try:
                # 生成参考轨迹并求解MPC得到预测
                ref_traj = generate_reference_path(current_state, config.goal_pos, mpc.config.N, config.dt, config)
                _, X_pred = mpc.solve(current_state, ref_traj, obstacles)
                pred_line.set_data(X_pred[0, :], X_pred[1, :])
                pred_points.set_data(X_pred[0, :], X_pred[1, :])
            except:
                pass# 求解失败时不更新预测轨迹
        
        return [path_line, car_patch, pred_line, pred_points, info_text] + obstacle_patches + safety_boundaries

    # 创建动画
    ani = animation.FuncAnimation(
        fig, update_animation, frames=n_steps+1, init_func=init_animation,
        interval=config.animation_interval, blit=True, repeat=False
    )
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 去重图例
    ax.legend(by_label.values(), by_label.keys(), loc='lower right', ncol=4)
       
    return ani


def plot_final_results(state_history, pred_history, config):
    plt.figure(figsize=(10, 40))
    
    # 1. 轨迹图（实际轨迹+预测轨迹）
    plt.subplot(4, 1, 1)
    plt.plot(state_history[:, 0], state_history[:, 1], 'b-', label='Actual Path')
    plt.plot(config.start_pos[0], config.start_pos[1], 'go', markersize=10, label='Start')
    plt.plot(config.goal_pos[0], config.goal_pos[1], 'r*', markersize=12, label='Goal')
    
    # 绘制部分预测轨迹（每5步）
    for i, pred in enumerate(pred_history):
        if i % 5 == 0:
            plt.plot(pred[0, :], pred[1, :], 'm--', alpha=0.3)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Vehicle Trajectory')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.xlim(config.xlim)
    plt.ylim(config.ylim)
    
    # 2. 航向角和速度变化
    plt.subplot(4, 1, 2)
    time = np.arange(len(state_history)) * config.dt
    plt.plot(time, np.rad2deg(state_history[:, 2]), 'g-', label='Heading Angle')
    plt.plot(time, state_history[:, 3], 'b-', label='Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Heading and Speed Evolution')
    plt.grid(True)
    plt.legend()
    
    # 3. 控制输入变化（加速度和转向速率）
    plt.subplot(4, 1, 3)
    if len(state_history) > 1:
        acceleration = np.diff(state_history[:, 3]) / config.dt  # 加速度 = 速度变化/时间
        delta_dot = np.diff(np.rad2deg(state_history[:, 4])) / config.dt  # 转向速率
        time_ctrl = np.arange(len(acceleration)) * config.dt
        
        plt.plot(time_ctrl, acceleration, 'r-', label='Acceleration (m/s²)')
        plt.plot(time_ctrl, delta_dot, 'c-', label='Steering Rate (°/s)')
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')
        plt.title('Control Signals')
        plt.grid(True)
        plt.legend()
    
    # 4. 横向误差（偏离中心线的距离）
    plt.subplot(4, 1, 4)
    lateral_error = state_history[:, 1]  # 参考y=0，误差即y坐标
    plt.plot(time, lateral_error, 'm-', label='Lateral Error')
    plt.axhline(0, color='g', linestyle='--', alpha=0.5)  # 中心线
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Lateral Error from Centerline')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.subplots_adjust(bottom=0.05, hspace=0.4)         
    plt.show()


def main():
   
    # 初始化配置和MPC控制器
    config = Config()
    mpc = MPCObstacleAvoidance(config)
    
    # 初始状态 [x, y, θ, v, δ]
    initial_state = np.array([
        config.start_pos[0],  # x位置
        config.start_pos[1],  # y位置
        config.start_pos[2],  # 航向角
        config.start_pos[3],  # 速度
        0.0                   # 转向角（初始为0）
    ])
    
    # 障碍物定义：[x0, y0, 航向角(rad), 速度(m/s), 长度(m), 宽度(m), 激活标志(1=激活)]
    obstacles = [
        [5.0, 0.75, np.deg2rad(0), 0.5, 2.0, 1.0, 1.0],   
        [14.0, -1.3, np.deg2rad(180), 0.8, 2.0, 1.0, 1.0] 
    ]
    
    # 仿真初始化
    n_steps = int(config.sim_time / config.dt)  # 总步数
    state_history = np.zeros((n_steps + 1, 5))  # 存储状态历史
    state_history[0] = initial_state
    current_state = initial_state.copy()
    pred_history = []  # 存储预测轨迹

    real_steps = n_steps 
    for i in range(n_steps):
        ref_traj = generate_reference_path(
            current_state, config.goal_pos,
            mpc.config.N, mpc.config.dt, config
        )
        
        u_opt, x_pred = mpc.solve(current_state, ref_traj, obstacles)
        pred_history.append(x_pred)
        
        control = u_opt[:, 0]  # 只取第一个控制量
        next_state = mpc.dynamics(current_state, control).full().flatten()  # 更新状态
        
        current_state = next_state
        state_history[i+1] = current_state
        
        print(f"Step {i+1}/{n_steps}: "
              f"Pos=({current_state[0]:.2f}, {current_state[1]:.2f})m, "
              f"θ={np.rad2deg(current_state[2]):.1f}°, "
              f"v={current_state[3]:.2f}m/s")
        
        # 检查是否到达目标
        pos_error = np.linalg.norm(current_state[:2] - config.goal_pos[:2])
        if pos_error < config.goal_threshold:
            print(f"\n到达目标！位置误差: {pos_error:.2f}m")
            real_steps = i + 1
            break


    animate_trajectory(state_history[:real_steps+1], obstacles, mpc, config, real_steps)
    plot_final_results(state_history[:real_steps+1], pred_history, config)


if __name__ == "__main__":
    main()
