import numpy as np

class BaseStates:
   def __init__(self,p0,euler0,vb0):
      self.p = p0                   # Boat position
      self.euler = euler0           # Attitude in euler
      self.vb = vb0                 # boat velocity in boat frame
      self.bias = np.zeros((3,1))   # Gyro bias

      # Low pass filtered gyro measurments
      self.wLpf = np.zeros((3,1))
      self.alpha = 0.1

   def update_w_lpf(self,gyros):
      # Pass the gyros through a low-pass filter
      self.wLpf = self.alpha*gyros+(1-self.alpha)*self.wLpf

   def update_state(self, belief, phi, theta, bias):
      """Update state estimate from estimator class"""
      self.p = belief.p
      self.euler = np.array([[phi.squeeze(),theta.squeeze(),belief.psi.squeeze()]]).T
      self.vb = belief.vb
      self.bias = bias
