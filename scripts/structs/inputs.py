import numpy as np

class Inputs:
   """Struct class for the inputs to a dynamic model"""
   def __init__(self,ut):
      self.accelerometers = np.array([ut[0:3]]).T
      self.gyros = np.array([ut[3:6]]).T
      self.phi = np.array([ut[6]])
      self.theta = np.array([ut[7]])