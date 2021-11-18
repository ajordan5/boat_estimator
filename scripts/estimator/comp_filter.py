import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.constants import g as gravity

class compFilter:
     def __init__(self, kp, ki) -> None:
          # Filter gains
          self.ki = ki
          self.kp = kp
          # Quadratic polynomial for gyro estimates
          self.gyroPrev = np.zeros((3,1))
          self.gyro2Prev = np.zeros((3,1))
          
     def run(self,baseStates,imu,dt):
          """Update the phi and theta estimates based on gyro and accelerometer readings."""
          
          # Current base attitude estimates in quaternion. Convert to [w, x, y, z]
          quat = R.from_euler('xyz', baseStates.euler.T, degrees=False).as_quat() # quaternion [x, y, z, w]
          q_hat = np.array([[quat.item(3)],
                         [quat.item(0)],
                         [quat.item(1)],
                         [quat.item(2)]])

          ## Error in attitude as predicted by the accelerometers
          g = np.array([[0.0, 0.0, -1.0]]).T                      # Gravity vector in gs TODO should this be negative w/ z down?
          accel = imu.accelerometers #/ gravity                   # Accelerometer values converted to gs
          a_normal = accel/np.linalg.norm(accel) 
          gamma = (a_normal + g)/np.linalg.norm(a_normal + g)
          
          
          # Rotation between accel estimate and inertial frame
          q_acc_1 = a_normal.T @ gamma
          q_acc_2 = np.cross(a_normal.T, gamma.T)
          q_acc = np.concatenate((q_acc_1, q_acc_2.T), axis=0)

          # Quaternion Inverse
          q_acc_inv = np.copy(q_acc)
          q_acc_inv[0] *= -1

          # Error
          q_tilde = self.quaternion_multiply(q_acc_inv, q_hat)
          s_tilde = q_tilde.item(0)
          v_tilde = q_tilde[1:4]

          w_acc = 2*s_tilde*v_tilde

          ## Gyro quadratic polynomial
          w_ = self.gyro_quadratic(imu.gyros)

          ## Composite omega
          bias = np.copy(baseStates.bias)
          bias -= 2*self.ki*w_acc * dt
          w_comp = w_ - bias + self.kp*w_acc

          ## Filter evaluation with matrix exponential propogation
          q_hat = self.filter_eval(w_comp, q_hat, dt)

          ## Convert to Euler
          phi = np.arctan2((2*(q_hat[1][0]*q_hat[0][0] + q_hat[2][0]*q_hat[3][0])), 1 - (q_hat[1][0]**2 + q_hat[2][0]**2))
          theta = np.arcsin(2*(q_hat[0][0]*q_hat[2][0] - q_hat[1][0]*q_hat[3][0]))
          
          """ut = [imu.accelerometers.item(0),imu.accelerometers.item(1),imu.accelerometers.item(2), \
                    imu.gyros.item(0),imu.gyros.item(1),imu.gyros.item(2), \
                    np.array([phi]), np.array([theta])]"""

          return phi, theta, bias

     def filter_eval(self, w_bar, q_hat_prev, dt):
          """Evaluate the complementary filter using a matrix exponential approximation"""
          w_norm = np.linalg.norm(w_bar)
          cos_w = np.cos(w_norm*dt/2)
          sin_w = np.sin(w_norm*dt/2)
          I4 = np.eye(4)

          # Skew symmetric matrix 
          w_4 = np.array([[0.0, -w_bar.item(0), -w_bar.item(1), -w_bar.item(2)],
                         [w_bar.item(0), 0.0, w_bar.item(2), -w_bar.item(1)],
                         [w_bar.item(1), -w_bar.item(2), 0.0, w_bar.item(0)],
                         [w_bar.item(2), w_bar.item(1), -w_bar.item(0), 0.0]])

          return (cos_w*I4 + sin_w*w_4/w_norm) @ q_hat_prev



     def quaternion_multiply(self, q0, q1):
          """Quaternion multiply two given quaternions
               q = [qw, qx, qy, qz]
          """
          w0, x0, y0, z0 = q0
          w1, x1, y1, z1 = q1
          return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                              x1*w0 + y1*z0 - z1*y0 + w1*x0,
                              -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                              x1*y0 - y1*x0 + z1*w0 + w1*z0])

     def gyro_quadratic(self, gyro):
      """Update a quadratic polynomial used to estimate true body angular velocity
         
         Input:
            gyro (np.array, 3x1): Most recent gyro reading
         Output:
            omega_bar (np.array, 3x1): true gyro rate estimate """

      omega_bar = (1/12) * (-self.gyro2Prev + 8*self.gyroPrev + 5*gyro)
      
      # Save previous values
      self.gyroPrev = gyro
      self.gyro2Prev = self.gyroPrev

      return omega_bar