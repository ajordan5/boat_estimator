import numpy as np

def run(baseStates,imu,dt,kp, ki):
     #I added the attitude model inversion.  It seemed to help.  It is not from Dr. Beard
     sphi = np.sin(baseStates.euler.item(0))
     cphi = np.cos(baseStates.euler.item(0))
     cth = np.cos(baseStates.euler.item(1))
     tth = np.tan(baseStates.euler.item(1))
     attitudeModelInversion = np.array([[1.0, sphi*tth, cphi*tth],
                                  [0.0, cphi, -sphi],
                                  [0.0, sphi/cth, cphi/cth]])
                                  
     # Euler as estimated by the accelerometers
     eulerAccel = np.array([[0.0,0.0,0.0]]).T
     #print(imu.accelerometers)
     eulerAccel[0][0] = np.arctan(imu.accelerometers.item(1)/imu.accelerometers.item(2)) #switched from arctan2 to arctan
     if imu.accelerometers.item(0) > 9.8:
          print("accelerometer forward value too high, ", imu.accelerometers)
          imu.accelerometers[0] = 9.8
     if imu.accelerometers.item(0) < -9.8:
          print("accelerometer forward value too high, ", imu.accelerometers)
          imu.accelerometers[0] = -9.8
     eulerAccel[1][0] = np.arcsin(imu.accelerometers.item(0)/9.81)
     eulerAccel[2][0] = baseStates.euler.item(2) #We update this with rtk compassing
     eulerError = eulerAccel - baseStates.euler

     # Bias
     baseStates.bias -= dt*ki*eulerError
     dEuler = (attitudeModelInversion @ imu.gyros - baseStates.bias) + kp*eulerError
     # print(dEuler.T)
     # print(eulerError.T)
     # print(kp, ki)
     phi = baseStates.euler.item(0) + dEuler.item(0)*dt
     theta = baseStates.euler.item(1) + dEuler.item(1)*dt

     ut = [imu.accelerometers.item(0),imu.accelerometers.item(1),imu.accelerometers.item(2), \
               imu.gyros.item(0),imu.gyros.item(1),imu.gyros.item(2), \
               np.array([phi]), np.array([theta])]

     return ut