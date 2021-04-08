import numpy as np

class Inputs:
   def __init__(self,ut):
      self.accelerometers = np.array([ut[0:3]]).T
      self.gyros = np.array([ut[3:6]]).T
      self.phi = ut[6]
      self.theta = ut[7]