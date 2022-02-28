import numpy as np

class TruthMsg:
    def __init__(self,timeSeconds = 0.0,positionMetersNed = [0.0,0.0,0.0],orientationRadNed = [0.0,0.0,0.0], \
                    velocityMetersPerSecondBody = [0.0,0.0,0.0],accelerationMetersPerSecondSquaredBody = [0.0,0.0,0.0], \
                    angularVelocityRadPerSecondBody = [0.0,0.0,0.0]):
        self.time = timeSeconds
        self.position = np.array([positionMetersNed]).T
        self.orientation = np.array([orientationRadNed]).T
        self.velocity = np.array([velocityMetersPerSecondBody]).T
        self.acceleration = np.array([accelerationMetersPerSecondSquaredBody]).T
        self.angularVelocity = np.array([angularVelocityRadPerSecondBody]).T

class ImuMsg:
    def __init__(self,timeSeconds = 0.0,accelerometersMetersPerSecondSquared = [0.0,0.0,0.0], \
                    gyrosRadiansPerSecond = [0.0,0.0,0.0]):
        self.time = timeSeconds
        self.accelerometers = np.array([accelerometersMetersPerSecondSquared]).T
        self.gyros = np.array([gyrosRadiansPerSecond]).T

    def remove_bias(self,accelBias,gyroBias):
        self.accelerometers -= accelBias
        self.gyros -= gyroBias

class RelPosMsg:
    def __init__(self,base2RoverRelativePositionNedMeters = [0.0,0.0,0.0],flags = '0b00000000'):
        self.base2RoverRelPos = np.array([base2RoverRelativePositionNedMeters]).T
        self.flags = flags

class GpsMsg:
    def __init__(self,positionEcefMeters = [0.0,0.0,0.0],velocityEcefMetersPerSecond = [0.0,0.0,0.0],latLonAltDegM = [0.0,0.0,0.0],fix = 0):
        self.positionEcef = np.array([positionEcefMeters]).T
        self.velocityEcef = np.array([velocityEcefMetersPerSecond]).T
        self.lla = np.array(latLonAltDegM)
        self.fix = fix

class GpsCompassMsg:
    def __init__(self,headingRad = 0.0,flags = '0b00000000'):
        self.heading = np.array([[headingRad]]).T
        self.flags = flags

class ApriltagMsg:
    def __init__(self,apriltag):
        self.position = apriltag.pose.pose.pose.position
        self.orientation = apriltag.pose.pose.pose.orientation
        self.t = np.array([[self.position.x, self.position.y, self.position.z]]).T