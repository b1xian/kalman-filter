import time
import numpy as np
import matplotlib.pyplot as plt

from KalmanFilterCTRAUtil import KalmanCTRA
from KalmanFilterCAUtil import KalmanCA
from KalmanFilterHeadingUtil import KalmanHeading

def EncodeHeadingRad(heading):
    if heading >= 0 and heading <= np.pi:
        return heading
    else:
        return heading + 2 * np.pi

def DecodeHeadingRad(heading):
    if heading > np.pi:
        return heading - 2 * np.pi
    else:
        return heading


def normalize_angle(angle):
    """将角度标准化到 [-pi, pi] 范围内。"""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def generate_sim_traj():
    # 生成螺旋部分
    spiral_cnt = 150
    theta = np.linspace(5.0, 3.5 * np.pi, spiral_cnt)  # 角度
    r = theta / 0.3  # 半径
    y_spiral = -r * np.cos(theta) + 60
    x_spiral = r * np.sin(theta)
    heading_spiral = theta
    # 生成直线部分
    line_dis = 30
    line_cnt = 15
    y_line = np.linspace(y_spiral[-1], y_spiral[-1] - line_dis, line_cnt)
    x_line = np.ones_like(y_line) * x_spiral[-1]
    # 合并螺旋和直线部分
    y_traj = np.concatenate((y_spiral, y_line))
    x_traj = np.concatenate((x_spiral, x_line))

    # 添加高斯噪声
    noise_scale = 0.0  # 调整噪声的尺度
    y_traj_noisy = y_traj + np.random.normal(0, noise_scale, y_traj.shape)
    x_traj_noisy = x_traj + np.random.normal(0, noise_scale, x_traj.shape)
    # 翻转
    y_traj_noisy = y_traj_noisy[::-1]
    x_traj_noisy = x_traj_noisy[::-1]

    # 计算每一步的方向向量
    dx = np.gradient(x_traj_noisy)
    dy = np.gradient(y_traj_noisy)
    vx = dx / 0.1
    vy = dy / 0.1
    # 计算航向角
    heading = np.arctan2(dy, dx)
    # for idx, h in enumerate(heading):
    #     vh = calc_velo_heading(dx[idx], dy[idx])
    #     h = normalize_angle(h)
    #     print("{}, x:{:.1f} y:{:.1f}, dx:{:.1f} dy:{:.1f}, heading:{:.3f}, {:.3f}".format(
    #         idx, x_traj_noisy[idx], y_traj_noisy[idx], dx[idx], dy[idx], h, vh))
        # heading[idx] = vh

    sim_traj = np.column_stack((x_traj_noisy, y_traj_noisy, heading, vx, vy))
    return sim_traj

def draw_sim_traj(sim_traj, ca_pred, ctra_pred):
    # 绘制轨迹
    # print(sim_traj[0], sim_traj[1])
    # plt.quiver(sim_traj[:, 0], sim_traj[:, 1], np.cos(sim_traj[:, 2]), np.sin(sim_traj[:, 2]), color='r', label='Heading', scale=100)
    plt.scatter(sim_traj[:, 0], sim_traj[:, 1], label='Noisy Trajectory', s=3.0)

    if ca_pred is not None:
        plt.plot(ca_pred[:, 0], ca_pred[:, 1], label='ca', color='g')
    if ctra_pred is not None:
        plt.plot(ctra_pred[:, 0], ctra_pred[:, 1], label='ctra', color='b')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Whistle Shaped Trajectory with Gaussian Noise')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('./ctra_traj.jpg')
    plt.show()

    ca_x = ca_pred[:, 0]
    ca_y = ca_pred[:, 1]
    ca_vx = ca_pred[:, 2]
    ca_vy = ca_pred[:, 3]
    ca_ax = ca_pred[:, 4]
    ca_ay = ca_pred[:, 5]

    ctra_x = ctra_pred[:, 0]
    ctra_y = ctra_pred[:, 1]
    ctra_heading = ctra_pred[:, 2]
    ctra_heading_rate = ctra_pred[:, 4]
    ctra_vx = ctra_pred[:, 6]
    ctra_vy = ctra_pred[:, 7]
    ctra_ax = ctra_pred[:, 8]
    ctra_ay = ctra_pred[:, 9]
    delta_vx = np.gradient(ctra_vx)
    delta_vy = np.gradient(ctra_vy)
    d_ax = delta_vx / 0.1
    d_ay = delta_vy / 0.1

    steps = range(len(ca_pred))
    fig = plt.figure(figsize=(18, 10))
    plt.subplots_adjust(hspace=1)

    # plt.legend(loc='best')
    ax = fig.add_subplot(8, 1, 1)
    ax.set_title('heading')
    ax.grid(True)
    ax.plot(steps, sim_traj[:, 2], 'g', label='origin_velo_heading')
    ax.plot(steps, ctra_heading, 'b', label='ctra_heading')

    ax = fig.add_subplot(8, 1, 2)
    ax.set_title('heading_rate')
    ax.grid(True)
    ax.plot(steps, ctra_heading_rate, 'b', label='ctra_heading_rate')

    plt.legend(loc='best')
    ax = fig.add_subplot(8, 1, 3)
    ax.set_title('x_dis')
    ax.grid(True)
    ax.plot(steps, ca_x, 'g', label='ca_x')
    ax.plot(steps, ctra_x, 'b', label='ctra_x')

    plt.legend(loc='best')
    ax = fig.add_subplot(8, 1, 4)
    ax.set_title('vx')
    ax.grid(True)
    ax.plot(steps, ca_vx, 'g', label='ca_vx')
    ax.plot(steps, ctra_vx, 'b', label='ctra_vx')
    ax.plot(steps, sim_traj[:, 3], 'r', label='det_vx')

    plt.legend(loc='best')
    ax = fig.add_subplot(8, 1, 5)
    ax.set_title('ax')
    ax.grid(True)
    ax.plot(steps, d_ax, 'r', label='det_ax')
    # ax.plot(steps, ca_ax, 'g', label='ca_ax')
    ax.plot(steps, ctra_ax, 'b', label='ctra_ax')

    plt.legend(loc='best')
    ax = fig.add_subplot(8, 1, 6)
    ax.set_title('y_dis')
    ax.grid(True)
    ax.plot(steps, ca_y, 'g', label='ca_y')
    ax.plot(steps, ctra_y, 'b', label='ctra_y')

    plt.legend(loc='best')
    ax = fig.add_subplot(8, 1, 7)
    ax.set_title('vy')
    ax.grid(True)
    ax.plot(steps, ca_vy, 'g', label='ca_vy')
    ax.plot(steps, ctra_vy, 'b', label='ctra_vy')
    ax.plot(steps, sim_traj[:, 4], 'r', label='det_vy')

    plt.legend(loc='best')
    ax = fig.add_subplot(8, 1, 8)
    ax.set_title('ay')
    ax.grid(True)
    ax.plot(steps, d_ay, 'r', label='det_ay')
    # ax.plot(steps, ca_ay, 'g', label='ca_ay')
    ax.plot(steps, ctra_ay, 'b', label='ctra_ay')

    plt.axis('auto')
    plt.savefig('./ctra_info.jpg')
    plt.show()


def create_ctra_filter():
    n = 6
    m = 3
    # x, y, yaw, velo, w, a
    Q_diag = np.array([0.5, 0.5, 0.01, 1.0, 0.1, 0.1])  # Q矩阵的对角元素
    Q = np.diag(Q_diag)  # 创建对角矩阵Q
    R_diag = np.array([2.0, 2.0, 0.1])  # R矩阵的对角元素
    R = np.diag(R_diag)  # 创建对角矩阵R
    P = np.eye(n)
    return KalmanCTRA(n, m, Q, R, P)

def create_ca_filter():
    n = 6
    m = 2
    Q_diag = np.array([0.5, 0.5, 0.2, 0.2, 0.1, 0.1])  # Q矩阵的对角元素
    Q = np.diag(Q_diag)  # 创建对角矩阵Q
    R_diag = np.array([1.0, 1.0])  # R矩阵的对角元素
    R = np.diag(R_diag)  # 创建对角矩阵R
    P = np.eye(n)
    return KalmanCA(n, m, Q, R, P)

def create_heading_filter():
    n = 1
    m = 1
    Q_diag = np.array([0.001])  # Q矩阵的对角元素
    Q = np.diag(Q_diag)  # 创建对角矩阵Q
    R_diag = np.array([0.001])  # R矩阵的对角元素
    R = np.diag(R_diag)  # 创建对角矩阵R
    P = np.eye(n)
    return KalmanHeading(n, m, Q, R, P)

def ca_test():
    ca_pred = []
    for i, traj in enumerate(sim_traj):
        if i == 0:
            x =  np.array([traj[0], traj[1], 0.0, 20.0, 0.0, 0.0])
            ca_filter.init(x)
            print('init ca filter [{:.1f}, {:.1f}, {:.3f}]'.format(traj[0], traj[1], traj[2]))
        else:
            ca_filter.predict(dt)
            ca_filter.correct(traj[:2])
            # print('update {}, mea:{:.1f} {:.1f}, corr:{:.1f} {:.1f}, velo:{:.1f} {:.1f}, acc:{:.2f} {:.2f}'.format(
            #     i, traj[0], traj[1],
            #     ca_filter.x[0], ca_filter.x[1], ca_filter.x[2],
            #     ca_filter.x[3], ca_filter.x[4], ca_filter.x[5]))
        ca_pred.append(np.append(np.array(ca_filter.x), traj[2]))
    return np.array(ca_pred)

def ctra_test():
    ctra_pred = []
    for i, traj in enumerate(sim_traj):
        encode_heading = EncodeHeadingRad(traj[2])
        if i == 0:
            x =  np.array([traj[0], traj[1], traj[2], 20.0, 0.0, 0.0])
            ctra_filter.init(x)
            heading_filter.init(np.array([encode_heading]))
            print('init ctra filter [{:.1f}, {:.1f}, {:.3f}]'.format(traj[0], traj[1], traj[2]))
        else:
            heading_filter.predict()
            heading_filter.correct(encode_heading)
            decoding_heading = DecodeHeadingRad(heading_filter.x[0])
            # print("{}, vh:{:.3f}, track_vh:{:.3f}".format(i, traj[2], decoding_heading))

            ctra_filter.predict(dt)
            measure = np.array([traj[0], traj[1], traj[2]])
            ctra_filter.correct(measure)

            yaw = ctra_filter.x[2]
            velo = ctra_filter.x[3]
            vx = yaw * np.cos(velo)
            vy = yaw * np.sin(velo)
            # print('ctra update {}, mea:{:.1f} {:.1f}, {:.3f}, loc:{:.1f} {:.1f}, heading:{:.3f},'
            #       'velo:{:.1f} {:.1f}'.format(
            #     i, traj[0], traj[1], traj[2],
            #     ctra_filter.x[0], ctra_filter.x[1],
            #     yaw, vx, vy))
        ctra_res = np.asarray(ctra_filter.x)
        ctra_vx = ctra_res[3] * np.cos(ctra_res[2])
        ctra_vy = ctra_res[3] * np.sin(ctra_res[2])
        # 要考虑角速度的变化
        ctra_ax = ctra_res[5] * np.cos(ctra_res[2]) - ctra_res[3] * np.sin(ctra_res[2]) * ctra_res[4]
        ctra_ay = ctra_res[5] * np.sin(ctra_res[2]) + ctra_res[3] * np.cos(ctra_res[2]) * ctra_res[4]

        ctra_res = np.append(ctra_res, ctra_vx)
        ctra_res = np.append(ctra_res, ctra_vy)
        ctra_res = np.append(ctra_res, ctra_ax)
        ctra_res = np.append(ctra_res, ctra_ay)
        ctra_pred.append(ctra_res)
    return np.array(ctra_pred)


if __name__ == '__main__':
    plt.figure(figsize=(20, 20))

    sim_traj = generate_sim_traj()

    # 创建滤波器
    ctra_filter = create_ctra_filter()
    ca_filter = create_ca_filter()
    heading_filter = create_heading_filter()

    dt = 0.1
    start = time.time()
    ca_pred = ca_test()
    print('ca cost:{}'.format(time.time() - start))

    start = time.time()
    ctra_pred = ctra_test()
    print('ctra cost:{}'.format(time.time() - start))

    draw_sim_traj(sim_traj, ca_pred, ctra_pred)
