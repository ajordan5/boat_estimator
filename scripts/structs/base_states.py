import numpy as np

class BaseStates:
   def __init__(self,p0,euler0,vb0):
      self.p = p0  
      self.euler = euler0
      self.vb = vb0

      # Low pass filtered gyro measurments
      self.wLpf = np.zeros((3,1))

      self.alpha = 0.1

   def update_w_lpf(self,gyros):
      # Pass the gyros through a low-pass filter
      self.wLpf = self.alpha*gyros+(1-self.alpha)*self.wLpf