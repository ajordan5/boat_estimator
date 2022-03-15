import numpy as np

class BaseStates:
   def __init__(self,p0,euler0,vb0,vr0):
      self.p = p0                   #Base relative position with respect to the base
      self.euler = euler0           #Base euler attitude representation in radians
      self.vb = vb0                 #Base velocity in the inertial frame
      self.vr = vr0                 # Rover velocity in the inertial frame
      self.wLpf = np.zeros((3,1))   #filtered angular velocity.  Only used in measurement model.

      self.alpha = 0.1
      self.bias = np.zeros((3,1))   # Gyro bias

   def update_w_lpf(self,gyros):
      self.wLpf = self.alpha*gyros+(1-self.alpha)*self.wLpf