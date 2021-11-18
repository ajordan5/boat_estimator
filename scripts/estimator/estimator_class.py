import navpy
import numpy as np
import math

import sys
sys.path.append('/home/alex/rtk_ws/src/boat_estimator/src/structs')

import ekf
from comp_filter import compFilter
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
      self.compFilter = compFilter(self.params.kp, self.params.ki)

   def imu_callback(self,imu):
      """Each time an IMU measurment is received, estimate base roll/pitch with complementary filter,
      update the base states and propogate the state in the EKF"""

      if self.firstImu:
         self.imuPrevTime = imu.time
         self.firstImu = False
         return

      # Run the complementary filter to estimate roll and pitch of the boat
      dt = imu.time - self.imuPrevTime
      self.imuPrevTime = imu.time
      phi, theta, bias = self.compFilter.run(self.baseStates,imu,dt)

      # Inputs for the EKF dynamic model
      ut = Inputs([imu.accelerometers.item(0),imu.accelerometers.item(1),imu.accelerometers.item(2), \
                  imu.gyros.item(0),imu.gyros.item(1),imu.gyros.item(2), \
                  np.array([phi]), np.array([theta])])
      self.baseStates.update_w_lpf(ut.gyros)
      ft = DynamicModel(ekf.update_dynamic_model(self.belief,ut))
      At = ekf.update_jacobian_A(self.belief,ut)
      Bt = ekf.update_jacobian_B(self.belief,ut)
      ekf.propagate(self.belief,self.params.RProcess,self.params.RInputs,ft,At,Bt,dt)

      # Update the state belief based on dynamics propogation
      self.baseStates.update_state(self.belief, phi, theta, bias)

   def relPos_callback(self,relPos):
      if relPos.flags[-3] != '1':
         print('relPos not valid = ', relPos.flags)
         return
      zt = relPos.base2RoverRelPos
      ht = ekf.update_rtk_relPos_model(self.belief.p,self.baseStates.euler,self.params.antennaOffset)
      Ct = ekf.get_jacobian_C_relPos(self.baseStates,self.params.antennaOffset)

      ekf.update(self.belief,self.params.QtRtk,zt,ht,Ct)

   def rover_gps_callback(self,gps):
      if gps.fix != 3:
         print('rover gps not in fix.  Fix = ', gps.fix)
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
      Ct = ekf.get_jacobian_C_base_velocity(self.baseStates)

      ekf.update(self.belief,self.params.QtGpsVelocity,zt,ht,Ct)

   def gps_compass_callback(self,gpsCompass):
      if gpsCompass.flags[1] != '1':
      #if gpsCompass.flags[-3] != '1':
         #print('Compass not valid = ', gpsCompass.flags)
         return
      zt = gpsCompass.heading
      ht = ekf.update_rtk_compass_model(self.belief.psi)
      Ct = ekf.get_jacobian_C_compass()
      
      ekf.update(self.belief,self.params.QtRtkCompass,zt,ht,Ct)

   def update_full_state(self,phi,theta):
      self.baseStates.p = self.belief.p
      self.baseStates.euler = np.array([[phi.squeeze(),theta.squeeze(),self.belief.psi.squeeze()]]).T
      self.baseStates.vb = self.belief.vb
