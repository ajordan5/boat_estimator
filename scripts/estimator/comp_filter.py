import numpy as np

def run(baseStates,imu,dt,kp, ki):
     # Attitude model inversion to convert angular rates in the bode frame to euler rates
     sphi = np.sin(baseStates.euler.item(0))
     cphi = np.cos(baseStates.euler.item(0))
     cth = np.cos(baseStates.euler.item(1))
     tth = np.tan(baseStates.euler.item(1))
     attitudeModelInversion = np.array([[1.0, sphi*tth, cphi*tth],
                                  [0.0, cphi, -sphi],
                                  [0.0, sphi/cth, cphi/cth]])
                                  
     # Euler as estimated by the accelerometers
     eulerAccel = np.array([[0.0,0.0,0.0]]).T
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

     # Bias Estimate
     baseStates.bias -= dt*ki*eulerError

     # Time derivative of Euler rates
     dEuler = (attitudeModelInversion @ imu.gyros - baseStates.bias) + kp*eulerError
     phi = baseStates.euler.item(0) + dEuler.item(0)*dt
     theta = baseStates.euler.item(1) + dEuler.item(1)*dt

     # Input to the dynamic model including roll and pitch estimate from this filter
     ut = [imu.accelerometers.item(0),imu.accelerometers.item(1),imu.accelerometers.item(2), \
               imu.gyros.item(0),imu.gyros.item(1),imu.gyros.item(2), \
               np.array([phi]), np.array([theta])]

     return ut