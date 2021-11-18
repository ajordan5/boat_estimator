import sys
import os
print(os.getcwd())
sys.path.append('./scripts/estimator')
sys.path.append('./scripts/structs')
#sys.path.append('..')
print(os.getcwd())
import numpy as np

import pytest
from base_states import BaseStates
from comp_filter import compFilter
from sensors import ImuMsg

# Orientaitons for tests of complementary filter on a stationary boat in degrees.
@pytest.mark.parametrize("euler", [(0,0,0), (0,10,0), (10,0,0), (10,-10,0), (-10,-10,0)])

def test_no_change(euler):
    # Test that comp filter roughly matches the accelerometer when the base is stationary at a given orientation
    p0 = np.zeros((3,1))
    q0 = np.radians(euler)
    v0 = np.zeros((3,1))
    base = BaseStates(p0, q0, v0)

    # Filter object
    kp = 0.1
    ki = 0.9
    cf = compFilter(kp, ki)

    # Generate IMUs
    gyro_noise = np.random.normal(size=(3,1))*.00001
    accel_noise = np.random.normal(size=(3,))*.00001
    accelerometers = [9.81*np.sin(q0.item(1))+accel_noise.item(0),
                     -9.81*np.cos(q0.item(1))*np.sin(q0.item(0))+accel_noise.item(1),
                     -9.81*np.cos(q0.item(1))*np.cos(q0.item(0))+accel_noise.item(2)]

    
    gyros = [gyro_noise.item(0), gyro_noise.item(1), gyro_noise.item(2)]
    imu = ImuMsg(0.0,accelerometers,gyros)
    dt = 0.1
    phi, theta, bias = cf.run(base,imu,dt)

    qExpected = q0

    # Phi
    tol = 3e-3
    assert np.isclose(phi, qExpected.item(0), atol=tol)

    # Theta
    assert np.isclose(theta, qExpected.item(1), atol=tol)
    """
    # Pitched up at 20 deg, no gyro movement
    pitch = np.radians(12.55)
    q0 = np.array(([[0.0,pitch,0.0]])).T
    accelerometers = [9.81*np.sin(pitch),0.0,-9.81*np.cos(pitch)]
    imu = ImuMsg(0.0,accelerometers,gyros)
    
    base = BaseStates(p0, q0, v0)
    qExpected = q0
    u = comp_filter.run(base,imu,dt,kp,ki, bias0)
    phi, theta = u[6:8]

    # Theta
    assert np.isclose(theta, qExpected.item(1))"""



