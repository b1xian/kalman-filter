import numpy as np

# 生成测试数据
dt = 0.1
time_steps = 100
t = np.linspace(0, (time_steps - 1) * dt, time_steps)
vx = 2 * t  # 假设 x 方向的速度为 2*t (即恒定加速度 2)
vy = 3 * t  # 假设 y 方向的速度为 3*t (即恒定加速度 3)

# 计算速度变化率
delta_vx = np.gradient(vx)
delta_vy = np.gradient(vy)

# 计算加速度
d_ax = delta_vx / dt
d_ay = delta_vy / dt

print("Computed acceleration in x direction:", d_ax)
print("Computed acceleration in y direction:", d_ay)
