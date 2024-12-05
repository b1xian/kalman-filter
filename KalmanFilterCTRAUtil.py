import numpy as np
import matplotlib.pyplot as plt

class KalmanCTRA():
    def __init__(self, n, m, Q, R, P):
        self.n = n
        self.m = m
        self.Q = Q
        self.R = R
        self.I = np.eye(n, n)
        self.F = np.eye(n, n)
        self.set_H()
        self.P = P
        x = np.zeros(n)

    def init(self, x):
        self.x = x

    def set_H(self):
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

    def set_F(self, dt):
        x = self.x
        if np.abs(x[4])<0.0001: # Driving straight
            x[4] = 0.0001
        a13 = (-x[4]*x[3]*np.cos(x[2]) + x[5]*np.sin(x[2]) - x[5]*np.sin(dt*x[4] + x[2]) + (dt*x[4]*x[5] + x[4]*x[3])*np.cos(dt*x[4] + x[2])) / x[4]**2
        a14 = (-x[4]*np.sin(x[2]) + x[4]*np.sin(dt*x[4] + x[2]))/x[4]**2
        a15 = (-dt*x[5]*np.sin(dt*x[4] + x[2]) + dt*(dt*x[4]*x[5] + x[4]*x[3])* np.cos(dt*x[4] + x[2]) - x[3]*np.sin(x[2]) + (dt*x[5] + x[3])*         np.sin(dt*x[4] + x[2]))/x[4]**2 - 2*(-x[4]*x[3]*np.sin(x[2]) - x[5]*         np.cos(x[2]) + x[5]*np.cos(dt*x[4] + x[2]) + (dt*x[4]*x[5] + x[4]*x[3])*         np.sin(dt*x[4] + x[2]))/x[4]**3
        a16 = ((dt*x[4]*np.sin(dt*x[4] + x[2]) - np.cos(x[2]) + np.cos(dt * x[4] + x[2]))/x[4]**2)
        a23 = (-x[4] * x[3] * np.sin(x[2]) - x[5] * np.cos(x[2]) + x[5] * np.cos(dt * x[4] + x[2]) - (-dt * x[4]*x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2])) / x[4]**2
        a24 = (x[4] * np.cos(x[2]) - x[4]*np.cos(dt*x[4] + x[2]))/x[4]**2
        a25 = (dt * x[5]*np.cos(dt*x[4] + x[2]) - dt * (-dt*x[4]*x[5] - x[4] * x[3]) * np.sin(dt * x[4] + x[2]) + x[3]*np.cos(x[2]) + (-dt*x[5] - x[3])*np.cos(dt*x[4] + x[2]))/ x[4]**2 - 2*(x[4]*x[3]*np.cos(x[2]) - x[5] * np.sin(x[2]) + x[5] * np.sin(dt*x[4] + x[2]) +         (-dt * x[4] * x[5] - x[4] * x[3])*np.cos(dt*x[4] + x[2]))/x[4]**3
        a26 = (-dt*x[4]*np.cos(dt*x[4] + x[2]) - np.sin(x[2]) + np.sin(dt*x[4] + x[2]))/x[4]**2

        self.F = np.matrix([[1.0, 0.0, a13, a14, a15, a16],
                            [0.0, 1.0, a23, a24, a25, a26],
                            [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    def predict(self, dt):
        x = self.x  # x y yaw v w a
        if np.abs(x[4]) < 0.0001: # Driving straight
            x[4] = 0.0001
        x[0] = x[0] + (1 / x[4]**2) * ((x[3]*x[4] + x[5] * x[4] * dt) *
                                                      np.sin(x[2] + x[4]* dt) + x[5] * np.cos(x[2] + x[4] * dt) - x[3] *
                                                      x[4] * np.sin(x[2]) - x[5] * np.cos(x[2]))
        x[1] = x[1] + (1 / x[4]**2) * ((-x[3]*x[4] - x[5] * x[4] * dt) *
                                                      np.cos(x[2] + x[4]* dt) + x[5] * np.sin(x[2] + x[4] * dt) + x[3] *
                                                      x[4] * np.cos(x[2]) - x[5] * np.sin(x[2]))
        # print('==============yaw {:.3f} + {:.3f}  = {:.3f} '.format(x[2], x[4] * dt, x[2] + x[4] * dt))
        x[2] = x[2] + x[4] * dt
        while x[2] > np.pi:
            # print('-- yaw > pi:{:.3f} -> {:.3f}'.format(x[2], x[2] - 2.0 * np.pi))
            x[2] -= 2.0 * np.pi
        while x[2] < -np.pi:
            # print('-- yaw < pi:{:.3f} -> {:.3f}'.format(x[2], x[2] + 2.0 * np.pi))
            x[2] += 2.0 * np.pi
        x[3] = x[3] + x[5] * dt
        x[4] = x[4]
        x[5] = x[5]
        self.x = x
        self.set_F(dt)
        self.P = self.F * self.P * self.F.transpose() + self.Q

    def correct(self, z):
        S = np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        K = self.P @ self.H.T @ S
        y = z - (self.H @ self.x)                         # Innovation or Residual
        while y[2] > np.pi:
            y[2] -= 2.0 * np.pi
        while y[2] < -np.pi:
            y[2] += 2.0 * np.pi
        self.x = self.x + K @ y
        self.x = self.x[0, :].A1
        self.P = (self.I - K @ self.H) @ self.P