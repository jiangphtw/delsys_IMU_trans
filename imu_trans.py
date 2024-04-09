import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 首先定义MahonyAHRS类，这里假设它已经在您的环境中实现
class MahonyAHRS:
    def __init__(self, sample_rate, kp=1.0, ki=0.0):
        self.sample_rate = sample_rate
        self.kp = kp
        self.ki = ki
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.eInt = np.array([0.0, 0.0, 0.0])

    def update(self, gyro, acc, mag):
    # 将陀螺仪数据从度每秒转换为弧度每秒
     gyro = np.radians(gyro)
    
     # 正规化加速度计测量
     if np.linalg.norm(acc) != 0:
         acc = acc / np.linalg.norm(acc)
     
     # 正规化磁力计测量
     if np.linalg.norm(mag) != 0:
         mag = mag / np.linalg.norm(mag)
    
     # 四元数元素
     q1, q2, q3, q4 = self.quaternion

     # 加速度计和磁力计的估计方向
     hx = 2.0 * (mag[0] * (0.5 - q3**2 - q4**2) + mag[1] * (q2*q3 - q1*q4) + mag[2] * (q2*q4 + q1*q3))
     hy = 2.0 * (mag[0] * (q2*q3 + q1*q4) + mag[1] * (0.5 - q2**2 - q4**2) + mag[2] * (q3*q4 - q1*q2))
     bx = np.sqrt((hx * hx) + (hy * hy))
     bz = 2.0 * (mag[0] * (q2*q4 - q1*q3) + mag[1] * (q3*q4 + q1*q2) + mag[2] * (0.5 - q2**2 - q3**2))

     # 加速度计的目标方向
     halfvx = q2*q4 - q1*q3
     halfvy = q1*q2 + q3*q4
     halfvz = q1**2 - 0.5 + q4**2

     # 磁力计的目标方向
     halfwx = bx * (0.5 - q3**2 - q4**2) + bz * (q2*q4 - q1*q3)
     halfwy = bx * (q2*q3 - q1*q4) + bz * (q1*q2 + q3*q4)
     halfwz = bx * (q1*q3 + q2*q4) + bz * (0.5 - q2**2 - q3**2)

     # 误差是交叉乘积之和
     halfex = (acc[1] * halfvz - acc[2] * halfvy) + (mag[1] * halfwz - mag[2] * halfwy)
     halfey = (acc[2] * halfvx - acc[0] * halfvz) + (mag[2] * halfwx - mag[0] * halfwz)
     halfez = (acc[0] * halfvy - acc[1] * halfvx) + (mag[0] * halfwy - mag[1] * halfwx)

     # 积分误差比例积分增益
     self.eInt[0] += halfex
     self.eInt[1] += halfey
     self.eInt[2] += halfez

     # 调整后的陀螺仪测量
     gyro[0] += self.kp * halfex + self.ki * self.eInt[0]
     gyro[1] += self.kp * halfey + self.ki * self.eInt[1]
     gyro[2] += self.kp * halfez + self.ki * self.eInt[2]

     # 四元数微分方程
     qDot1 = 0.5 * (-q2 * gyro[0] - q3 * gyro[1] - q4 * gyro[2])
     qDot2 = 0.5 * (q1 * gyro[0] + q3 * gyro[2] - q4 * gyro[1])
     qDot3 = 0.5 * (q1 * gyro[1] - q2 * gyro[2] + q4 * gyro[0])
     qDot4 = 0.5 * (q1 * gyro[2] + q2 * gyro[1] - q3 * gyro[0])

     # 四元数积分
     q1 += qDot1 * (1.0 / self.sample_rate)
     q2 += qDot2 * (1.0 / self.sample_rate)
     q3 += qDot3 * (1.0 / self.sample_rate)
     q4 += qDot4 * (1.0 / self.sample_rate)
 
     # 正规化四元数
     norm = np.sqrt(q1*q1 + q2*q2 + q3*q3 + q4*q4)
     self.quaternion = np.array([q1, q2, q3, q4]) / norm

    
    def get_euler_angles(self):
      q1, q2, q3, q4 = self.quaternion
      roll = np.arctan2(2 * (q1*q2 + q3*q4), 1 - 2 * (q2*q2 + q3*q3))
      pitch = np.arcsin(2 * (q1*q3 - q4*q2))
      yaw = np.arctan2(2 * (q1*q4 + q2*q3), 1 - 2 * (q3*q3 + q4*q4))
      return roll, pitch, yaw  # in radians

# 读取Excel文件
df = pd.read_excel('不動.xlsx', sheet_name='工作表1')

# 提取IMU数据 - 根据您实际的Excel文件列名进行替换
accelerometer_data = df[['Trigno IM sensor 1: Acc 1.X (IM) [g]', 'Trigno IM sensor 1: Acc 1.Y (IM) [g]', 'Trigno IM sensor 1: Acc 1.Z (IM) [g]']].values
gyroscope_data = df[['Trigno IM sensor 1: Gyro 1.X (IM) [deg/sec]', 'Trigno IM sensor 1: Gyro 1.Y (IM) [deg/sec]', 'Trigno IM sensor 1: Gyro 1.Z (IM) [deg/sec]']].values
magnetometer_data = df[['Trigno IM sensor 1: Mag 1.X (IM) [uTesla]', 'Trigno IM sensor 1: Mag 1.Y (IM) [uTesla]', 'Trigno IM sensor 1: Mag 1.Z (IM) [uTesla]']].values

# 初始化Mahony算法
sample_rate = 148.15  # 这里替换为您从数据中计算得出的采样率
mahony_filter = MahonyAHRS(sample_rate=sample_rate)

# 计算欧拉角
euler_angles = []
for acc, gyro, mag in zip(accelerometer_data, gyroscope_data, magnetometer_data):
    mahony_filter.update(gyro, acc, mag)
    euler_angles.append(mahony_filter.get_euler_angles())

# 创建时间轴
time_axis = np.arange(len(euler_angles)) / sample_rate

# 提取每个欧拉角
roll_angles = [np.degrees(angle[0]) for angle in euler_angles]
pitch_angles = [np.degrees(angle[1]) for angle in euler_angles]
yaw_angles = [np.degrees(angle[2]) for angle in euler_angles]

# 绘制图表
plt.figure(figsize=(12, 9))

# 绘制Roll角度变化
plt.subplot(3, 1, 1)
plt.plot(time_axis, roll_angles, label='Roll')
plt.title('Roll Angle Over Time')
plt.ylabel('Angle (degrees)')
plt.grid(True)

# 绘制Pitch角度变化
plt.subplot(3, 1, 2)
plt.plot(time_axis, pitch_angles, label='Pitch')
plt.title('Pitch Angle Over Time')
plt.ylabel('Angle (degrees)')
plt.grid(True)

# 绘制Yaw角度变化
plt.subplot(3, 1, 3)
plt.plot(time_axis, yaw_angles, label='Yaw')
plt.title('Yaw Angle Over Time')
plt.ylabel('Angle (degrees)')
plt.xlabel('Time (seconds)')
plt.grid(True)

# 显示图例
plt.legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()