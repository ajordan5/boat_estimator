import navpy
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

import sys

import ekf
import comp_filter
from belief import Belief
from dynamic_model import DynamicModel
from inputs import Inputs
from base_states import BaseStates

class Estimator:
   def __init__(self,params):
      self.params = params
      self.belief = Belief(self.params.p0,self.params.vr0,self.params.psi0,self.params.vb0,self.params.P0)
      self.baseStates = BaseStates(self.params.p0,self.params.euler0,self.params.vb0)
      self.wLpf = np.zeros((3,1))
      self.refLlaSet = False
      self.latRef = 0.0
      self.lonRef = 0.0
      self.altRef = 0.0
      self.refEcef = np.zeros((3,1))
      self.imuPrevTime = 0.0
      self.firstImu = True
      self.alpha = 0.05

   def imu_callback(self,imu):
      if self.firstImu:
         self.imuPrevTime = imu.time
         self.firstImu = False
         return
      imu.remove_bias(self.params.accelBias,self.params.gyroBias)
      dt = imu.time - self.imuPrevTime
      self.imuPrevTime = imu.time

      ut = Inputs(comp_filter.run(self.baseStates,imu,dt,self.params.kp))
      self.baseStates.update_w_lpf(ut.gyros)
      ft = DynamicModel(ekf.update_dynamic_model(self.belief,ut))
      At = ekf.update_jacobian_A(self.belief,ut)
      Bt = ekf.update_jacobian_B(self.belief,ut)

      ekf.propagate(self.belief,self.params.RProcess,self.params.RInputs,ft,At,Bt,dt)
      self.update_full_state(ut.phi,ut.theta)

   def relPos_callback(self,relPos):
      flagsBinary = bin(relPos.flags)
      if flagsBinary[-3] != '1':
         #print('relPos not valid = ', relPos.flags)
         return
      zt = relPos.base2RoverRelPos
      ht = ekf.update_rtk_relPos_model(self.belief.p,self.baseStates.euler,self.params.antennaOffset)
      Ct = ekf.get_jacobian_C_relPos(self.baseStates,self.params.antennaOffset)

      ekf.update(self.belief,self.params.QtRtk,zt,ht,Ct)

   def rover_gps_callback(self,gps):
      if gps.fix != 3:
         #print('rover gps not in fix.  Fix = ', gps.fix)
         return
      if not self.refLlaSet:
         self.latRef = gps.lla[0]
         self.lonRef = gps.lla[1]
         self.altRef = gps.lla[2]
         refEcef1D = navpy.lla2ecef(self.latRef,self.lonRef,self.altRef)
         self.refEcef = np.array([refEcef1D]).T
         self.refLlaSet = True
         print('ref lla = ', self.latRef, ' ', self.lonRef, ' ', self.altRef)

      velocityNed1D = navpy.ecef2ned(gps.velocityEcef, self.latRef, self.lonRef, self.altRef)
      velocityNed = np.array([velocityNed1D]).T

      zt = velocityNed
      ht = ekf.update_rover_gps_velocity_model(self.belief.vr)
      Ct = ekf.get_jacobian_C_rover_velocity()

      ekf.update(self.belief,self.params.QtGpsVelocity,zt,ht,Ct)

   def base_gps_callback(self,gps):
      if gps.fix != 3:
         print('base gps not in fix.  Fix = ', gps.fix)
         return
      if not self.refLlaSet:
         return
      velocityNed1D = navpy.ecef2ned(gps.velocityEcef, self.latRef, self.lonRef, self.altRef)
      velocityNed = np.array([velocityNed1D]).T

      zt = velocityNed
      ht = ekf.update_base_gps_velocity_model(self.baseStates.euler,self.belief.vb,self.baseStates.wLpf,self.params.antennaOffset)
      Ct = ekf.get_jacobian_C_base_velocity(self.baseStates.euler,self.belief.vb)

      ekf.update(self.belief,self.params.QtGpsVelocity,zt,ht,Ct)

   def gps_compass_callback(self,gpsCompass):
      flagsBinary = bin(gpsCompass.flags)
      if gpsCompass.flags < 256:
         print('Compass not valid = ', gpsCompass.flags)
         return
      elif gpsCompass.flags > 512 and flagsBinary[3] != 1:
         print('Compass not valid = ', gpsCompass.flags)
         return

      zt = gpsCompass.heading
      ht = ekf.update_rtk_compass_model(self.belief.psi,zt)
      Ct = ekf.get_jacobian_C_compass()
      
      ekf.update(self.belief,self.params.QtRtkCompass,zt,ht,Ct)

   def update_full_state(self,phi,theta):
      self.baseStates.p = self.belief.p
      self.baseStates.euler = np.array([[phi.squeeze(),theta.squeeze(),self.belief.psi.squeeze()]]).T
      Rb2i = R.from_euler('xyz',self.baseStates.euler.squeeze())
      newVb = Rb2i.apply(self.belief.vb.T).T
      self.baseStates.vb = self.low_pass_filter(newVb,self.baseStates.vb)

   def low_pass_filter(self,new,old):
      filtered = self.alpha*new+(1-self.alpha)*old
      return filtered
