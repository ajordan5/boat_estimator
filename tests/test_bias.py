import sys
sys.path.append('/home/alex/px4_ws/src/boat_estimator/scripts/estimator')
sys.path.append('/home/alex/px4_ws/src/boat_estimator/scripts/structs')
import numpy as np

import unittest
from states_covariance import StatesCovariance
from estimator_class import Estimator
from sensors import ImuMsg
from sensors import GpsMsg


def test_pitch_xz_acceleration(self):
    time = 123.023
    accelerometers = [0.0,0.0,-9.81]
    gyros = [0.0,1.0,0.0]
    imu = ImuMsg(time,accelerometers,gyros)
    dt = 0.1

    estimator = Estimator()
    estimator.imuPrevTime = time
    steps = 5
    for i in range(steps):
        imu.time = estimator.imuPrevTime + dt
        pPrev = estimator.belief.p
        qPrev = estimator.belief.q
        vPrev = estimator.belief.v
        baPrev = estimator.belief.ba
        bgPrev = estimator.belief.bg
        estimator.imu_callback(imu)
    positionEcef = [-1800198.22956286, -4532912.78883003,  4098399.82768237]
    velocityEcef = [0.0,0.0,0.0]
    gps = GpsMsg(positionEcef,velocityEcef)
    estimator.set_ref_lla_callback(40.23,-111.66,1370.0)
    estimator.gps_callback(gps)
    print('accelerometer bias = ', estimator.belief.ba)
    print('gyro bias = ', estimator.belief.bg)

