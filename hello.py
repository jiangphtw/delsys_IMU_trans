import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 定義MahonyAHRS
class MahonyAHRS:
    def __init__(self, sample_rate, kp=1.0, ki=0.0):
        self.sample_rate = sample_rate
        self.kp = kp
        self.ki = ki
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.eInt = np.array([0.0, 0.0, 0.0])

    def update(self, gyro, acc, mag):
     # 陀螺儀單位度每秒轉換為弧度每秒
     gyro = np.radians(gyro)
    
     # 正規化加速規量測
     if np.linalg.norm(acc) != 0:
         acc = acc / np.linalg.norm(acc)
     
     # 正規化磁力計量測
     if np.linalg.norm(mag) != 0:
         mag = mag / np.linalg.norm(mag)
    
     # 四元數元素
     q1, q2, q3, q4 = self.quaternion

     # 加速規和磁力計的估計方向
     hx = 2.0 * (mag[0] * (0.5 - q3**2 - q4**2) + mag[1] * (q2*q3 - q1*q4) + mag[2] * (q2*q4 + q1*q3))
     hy = 2.0 * (mag[0] * (q2*q3 + q1*q4) + mag[1] * (0.5 - q2**2 - q4**2) + mag[2] * (q3*q4 - q1*q2))
     bx = np.sqrt((hx * hx) + (hy * hy))
     bz = 2.0 * (mag[0] * (q2*q4 - q1*q3) + mag[1] * (q3*q4 + q1*q2) + mag[2] * (0.5 - q2**2 - q3**2))

     # 加速規的目標方向
     halfvx = q2*q4 - q1*q3
     halfvy = q1*q2 + q3*q4
     halfvz = q1**2 - 0.5 + q4**2

     # 磁力計的目標方向
     halfwx = bx * (0.5 - q3**2 - q4**2) + bz * (q2*q4 - q1*q3)
     halfwy = bx * (q2*q3 - q1*q4) + bz * (q1*q2 + q3*q4)
     halfwz = bx * (q1*q3 + q2*q4) + bz * (0.5 - q2**2 - q3**2)

     # 誤差交叉計算
     halfex = (acc[1] * halfvz - acc[2] * halfvy) + (mag[1] * halfwz - mag[2] * halfwy)
     halfey = (acc[2] * halfvx - acc[0] * halfvz) + (mag[2] * halfwx - mag[0] * halfwz)
     halfez = (acc[0] * halfvy - acc[1] * halfvx) + (mag[0] * halfwy - mag[1] * halfwx)

     # 積分誤差gain
     self.eInt[0] += halfex
     self.eInt[1] += halfey
     self.eInt[2] += halfez

     # 調整後的陀螺儀資訊
     gyro[0] += self.kp * halfex + self.ki * self.eInt[0]
     gyro[1] += self.kp * halfey + self.ki * self.eInt[1]
     gyro[2] += self.kp * halfez + self.ki * self.eInt[2]

     # 四元數微分方程
     qDot1 = 0.5 * (-q2 * gyro[0] - q3 * gyro[1] - q4 * gyro[2])
     qDot2 = 0.5 * (q1 * gyro[0] + q3 * gyro[2] - q4 * gyro[1])
     qDot3 = 0.5 * (q1 * gyro[1] - q2 * gyro[2] + q4 * gyro[0])
     qDot4 = 0.5 * (q1 * gyro[2] + q2 * gyro[1] - q3 * gyro[0])

     # 四元數積分
     q1 += qDot1 * (1.0 / self.sample_rate)
     q2 += qDot2 * (1.0 / self.sample_rate)
     q3 += qDot3 * (1.0 / self.sample_rate)
     q4 += qDot4 * (1.0 / self.sample_rate)
 
     # 正規化四元數
     norm = np.sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4)
     self.quaternion = np.array([q1, q2, q3, q4]) / norm

    
    def get_euler_angles(self):
      q1, q2, q3, q4 = self.quaternion
      roll = np.arctan2(2 * (q1*q2 + q3*q4), 1 - 2 * (q2*q2 + q3*q3))
      pitch = np.arcsin(2 * (q1*q3 - q4*q2))
      yaw = np.arctan2(2 * (q1*q4 + q2*q3), 1 - 2 * (q3*q3 + q4*q4))
      return np.arctan2(2 * (q1*q2 + q3*q4), 1 - 2 * (q2*q2 + q3*q3)), np.arcsin(2 * (q1*q3 - q4*q2)), np.arctan2(2 * (q1*q4 + q2*q3), 1 - 2 * (q3*q3 + q4*q4))
    

def euler_to_rotation_matrix(roll, pitch, yaw):
     R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
     R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
     R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
     R = np.dot(R_z, np.dot(R_y, R_x))
     return R



# read Excel file
df = pd.read_excel('不動.xlsx', sheet_name='工作表1')

# get imu data
accelerometer_data = df[['Trigno IM sensor 1: Acc 1.X (IM) [g]', 'Trigno IM sensor 1: Acc 1.Y (IM) [g]', 'Trigno IM sensor 1: Acc 1.Z (IM) [g]']].values
gyroscope_data = df[['Trigno IM sensor 1: Gyro 1.X (IM) [deg/sec]', 'Trigno IM sensor 1: Gyro 1.Y (IM) [deg/sec]', 'Trigno IM sensor 1: Gyro 1.Z (IM) [deg/sec]']].values
magnetometer_data = df[['Trigno IM sensor 1: Mag 1.X (IM) [uTesla]', 'Trigno IM sensor 1: Mag 1.Y (IM) [uTesla]', 'Trigno IM sensor 1: Mag 1.Z (IM) [uTesla]']].values

# 初始化Mahony算法
sample_rate = 148.15  # 这里替换为您从数据中计算得出的采样率
mahony_filter = MahonyAHRS(sample_rate=sample_rate)

# 計算歐拉角與旋轉矩陣
euler_angles = []
rotation_matrices = []
for acc, gyro, mag in zip(accelerometer_data, gyroscope_data, magnetometer_data):
    mahony_filter.update(gyro, acc, mag)
    angles = mahony_filter.get_euler_angles()
    euler_angles.append(angles)
    
    # 使用已计算的欧拉角来生成旋转矩阵
    R = euler_to_rotation_matrix(*angles)
    rotation_matrices.append(R)# 創建時間軸





# 计算欧拉角和旋转矩阵
euler_angles = []
rotation_matrices = []
for acc, gyro, mag in zip(accelerometer_data, gyroscope_data, magnetometer_data):
    mahony_filter.update(gyro, acc, mag)
    roll, pitch, yaw = mahony_filter.get_euler_angles()
    euler_angles.append((roll, pitch, yaw))
    
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    rotation_matrices.append(R)

# 绘制欧拉角度变化图
plt.figure(figsize=(12, 9))

# 绘制Roll角度变化
plt.subplot(3, 1, 1)
plt.plot(time_axis, [np.degrees(angle[0]) for angle in euler_angles], label='Roll')
plt.title('Roll Angle Over Time')
plt.ylabel('Angle (degrees)')
plt.grid(True)

# 绘制Pitch角度变化
plt.subplot(3, 1, 2)
plt.plot(time_axis, [np.degrees(angle[1]) for angle in euler_angles], label='Pitch')
plt.title('Pitch Angle Over Time')
plt.ylabel('Angle (degrees)')
plt.grid(True)

# 绘制Yaw角度变化
plt.subplot(3, 1, 3)
plt.plot(time_axis, [np.degrees(angle[2]) for angle in euler_angles], label='Yaw')
plt.title('Yaw Angle Over Time')
plt.ylabel('Angle (degrees)')
plt.xlabel('Time (seconds)')
plt.grid(True)

plt.tight_layout()
plt.legend()
plt.show()

# 三维可视化传感器朝向...
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
initial_vector = np.array([1, 0, 0])
step = 10

for i, R in enumerate(rotation_matrices):
    if i % step == 0:
        transformed_vector = np.dot(R, initial_vector)
        ax.quiver(0, 0, 0, transformed_vector[0], transformed_vector[1], transformed_vector[2], length=0.1, normalize=True)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sensor Orientation Over Time')
plt.show()
