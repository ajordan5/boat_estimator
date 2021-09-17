import numpy as np

def run(baseStates,imu,dt,kp):
     """Update the phi and theta estimates based on gyro and accelerometer readings."""
     
     # Transformation from gyro angular rates to euler angle derivatives?
     sphi = np.sin(baseStates.euler.item(0))
     cphi = np.cos(baseStates.euler.item(0))
     cth = np.cos(baseStates.euler.item(1))
     tth = np.tan(baseStates.euler.item(1))
     attitudeModelInversion = np.array([[1.0, sphi*tth, cphi*tth],
                                  [0.0, cphi, -sphi],
                                  [0.0, sphi/cth, cphi/cth]])
     
     # aBody = imu.accelerometers #This was negative
     eulerAccel = np.array([[0.0,0.0,0.0]]).T

     # Estimate base angles using the accelerometer assuming the multi-rotor is in the air in a hold position
     eulerAccel[0][0] = np.arctan(imu.accelerometers.item(1)/imu.accelerometers.item(2)) #switched from arctan2 to arctan
     if imu.accelerometers.item(0) > 9.8:
          print("accelerometer forward value too high, ", imu.accelerometers)
          imu.accelerometers[0] = 9.8
     eulerAccel[1][0] = np.arcsin(imu.accelerometers.item(0)/9.81)
     eulerAccel[2][0] = baseStates.euler.item(2) #We update this with rtk compassing

     # Difference between previous angle estimates and current estimate based on only the accelerometers
     eulerError = eulerAccel - baseStates.euler

     # New change in base angles
     dEuler = attitudeModelInversion @ imu.gyros + kp*eulerError
     
     phi = baseStates.euler.item(0) + dEuler.item(0)*dt
     theta = baseStates.euler.item(1) + dEuler.item(1)*dt

     ut = [imu.accelerometers.item(0),imu.accelerometers.item(1),imu.accelerometers.item(2), \
               imu.gyros.item(0),imu.gyros.item(1),imu.gyros.item(2), \
               np.array([phi]), np.array([theta])]

     return ut