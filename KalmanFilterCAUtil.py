import numpy as np
import matplotlib.pyplot as plt

class KalmanCA():
    def __init__(self, n, m, Q, R, P):
        self.n = n
        self.m = m
        self.Q = Q
        self.R = R
        self.I = np.eye(n, n)
        self.F = np.eye(n, n)
        self.H = np.eye(m, n)
        self.P = P
        self.x = np.zeros(n)

    def init(self, x):
        self.x = x

    def set_F(self, dt):
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[2, 4] = dt
        self.F[3, 5] = dt
        self.F[0, 4] = dt * dt * 0.5
        self.F[1, 5] = dt * dt * 0.5

    def predict(self, dt):
        self.set_F(dt)
        # if np.abs(yawrate[filterstep])<0.0001: # Driving straight
        #     self.x[4] = 0.0001
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z):
        self.z = z
        # 更新测量矩阵
        S = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)  # 卡尔曼增益
        G = self.P @ self.H.T @ S  # 校正

        # 更新状态向量和协方差矩阵
        self.x += G @ (z - self.H @ self.x)
        self.P = (self.I - G @ self.H) @ self.P