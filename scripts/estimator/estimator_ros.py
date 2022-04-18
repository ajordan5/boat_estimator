#!/usr/bin/env python3

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('../rtk_ws/src/boat_estimator/scripts/structs')
sys.path.append('../rtk_ws/src/boat_estimator/params')
#sys.path.append('/home/matt/px4_ws/src/boat_estimator/scripts/structs')
#sys.path.append('/home/matt/px4_ws/src/boat_estimator/params')

from geometry_msgs.msg import Vector3Stamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from ublox.msg import PosVelEcef
from ublox.msg import RelPos
from sensors import ImuMsg,RelPosMsg, GpsMsg, GpsCompassMsg, ApriltagMsg

from estimator_params_class import EstimatorParams
from estimator_class import Estimator
from apriltag_ros.msg import AprilTagDetectionArray

class EstimatorRos:
    def __init__(self):
        
        self.relPosEstimate = Vector3Stamped()
        self.odomEstimate = Odometry()
        self.relVel = Odometry()
        self.eulerEstimate = Vector3Stamped()
        params = EstimatorParams()
        self.estimator = Estimator(params)
        
        self.apriltagID = self.estimator.params.apriltagID
        
        self.boat_estimate_pub_ = rospy.Publisher('base_odom', Odometry, queue_size=5, latch=True)
        self.relative_velocity_pub_ = rospy.Publisher('rel_vel', Odometry, queue_size=5, latch=True)
        self.boat_euler_pub_ = rospy.Publisher('base_euler', Vector3Stamped, queue_size=5, latch=True)
        self.imu_sub_ = rospy.Subscriber('imu', Imu, self.imuCallback, queue_size=5)
        self.base_2_rover_relPos_sub_ = rospy.Subscriber('base_2_rover_relPos', RelPos, self.relPosCallback, queue_size=5)
        self.rover_pos_vel_ecef_sub_ = rospy.Subscriber('rover_posVelEcef', PosVelEcef, self.roverPosVelEcefCallback, queue_size=5)
        self.base_pos_vel_ecef_sub_ = rospy.Subscriber('base_posVelEcef', PosVelEcef, self.basePosVelEcefCallback, queue_size=5)
        self.comp_relPos_sub_ = rospy.Subscriber('compass_relPos', RelPos, self.compassRelPosCallback, queue_size=5)
        self.aprilTag_sub_ = rospy.Subscriber('tag_detections', AprilTagDetectionArray, self.aprilTagCallback, queue_size=5)
        print("HERE",self.apriltagID)
        while not rospy.is_shutdown():
            rospy.spin()

    def imuCallback(self,msg):
        timeSeconds = msg.header.stamp.secs + msg.header.stamp.nsecs*1E-9
        gyrosRadiansPerSecond = [msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]
        accelerometersMetersPerSecondSquared = [msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z]
        imu = ImuMsg(timeSeconds,accelerometersMetersPerSecondSquared,gyrosRadiansPerSecond)

        self.estimator.imu_callback(imu)
        self.publish_estimates()

    def relPosCallback(self,msg):
        base2RoverRelativePositionNedMeters = np.array(msg.relPosNED) + np.array(msg.relPosHPNED)
        flags = msg.flags#bin(msg.flags)
        relPos = RelPosMsg(base2RoverRelativePositionNedMeters,flags)

        self.estimator.relPos_callback(relPos)

    def roverPosVelEcefCallback(self,msg):
        positionEcefMeters = msg.position
        velocityEcefMetersPerSecond = msg.velocity
        latLonAltDegM = msg.lla
        fix = msg.fix
        gps = GpsMsg(positionEcefMeters,velocityEcefMetersPerSecond,latLonAltDegM,fix)
 
        self.estimator.rover_gps_callback(gps)       
    
    def basePosVelEcefCallback(self,msg):
        positionEcefMeters = msg.position
        velocityEcefMetersPerSecond = msg.velocity
        latLonAltDegM = msg.lla
        fix = msg.fix
        gps = GpsMsg(positionEcefMeters,velocityEcefMetersPerSecond,latLonAltDegM,fix)
 
        self.estimator.base_gps_callback(gps)

    def compassRelPosCallback(self,msg):
        headingRad = msg.relPosHeading
        flags = msg.flags #bin(msg.flags)
        gpsCompass = GpsCompassMsg(headingRad,flags)

        self.estimator.gps_compass_callback(gpsCompass)

    def aprilTagCallback(self,msg):
        # Update state with apriltag if you locate the specific tag from the boat
        for detection in msg.detections:
            if detection.id == (self.apriltagID,):
                Rtag2i = R.from_euler('xyz' ,self.estimator.baseStates.euler.squeeze(), degrees=False) 
                apriltag = ApriltagMsg(detection)
                self.estimator.apriltag_callback(apriltag, Rtag2i)

    def publish_estimates(self):
        timeStamp = rospy.Time.now()

        # Construct boat_odom
        self.odomEstimate.header.stamp = timeStamp
        self.relVel.header.stamp = timeStamp
        self.eulerEstimate.header.stamp = timeStamp
        # Relative position 
        self.odomEstimate.pose.pose.position.x = self.estimator.baseStates.p.item(0)
        self.odomEstimate.pose.pose.position.y = self.estimator.baseStates.p.item(1)
        self.odomEstimate.pose.pose.position.z = self.estimator.baseStates.p.item(2)
        # Boat attitude
        self.eulerEstimate.vector.x = self.estimator.baseStates.euler.item(0)
        self.eulerEstimate.vector.y = self.estimator.baseStates.euler.item(1)
        self.eulerEstimate.vector.z = self.estimator.baseStates.euler.item(2)
        quat = R.from_euler('xyz', self.estimator.baseStates.euler.T, degrees=False).as_quat()
        self.odomEstimate.pose.pose.orientation.x = quat.item(0)
        self.odomEstimate.pose.pose.orientation.y = quat.item(1)
        self.odomEstimate.pose.pose.orientation.z = quat.item(2)
        self.odomEstimate.pose.pose.orientation.w = quat.item(3)
        # Boat velocity
        self.odomEstimate.twist.twist.linear.x = self.estimator.baseStates.vb.item(0)
        self.odomEstimate.twist.twist.linear.y = self.estimator.baseStates.vb.item(1)
        self.odomEstimate.twist.twist.linear.z = self.estimator.baseStates.vb.item(2)
        # Relative velocity
        self.relVel.twist.twist.linear.x = self.estimator.baseStates.vb.item(0) - self.estimator.baseStates.vr.item(0)
        self.relVel.twist.twist.linear.y = self.estimator.baseStates.vb.item(1) - self.estimator.baseStates.vr.item(1)
        self.relVel.twist.twist.linear.z = self.estimator.baseStates.vb.item(2) - self.estimator.baseStates.vr.item(2)

        self.boat_estimate_pub_.publish(self.odomEstimate)
        self.relative_velocity_pub_.publish(self.relVel)
        self.boat_euler_pub_.publish(self.eulerEstimate)

if __name__ == '__main__':
    rospy.init_node('estimator_ros', anonymous=True)
    try:
        estimatorRos = EstimatorRos()
    except:
        rospy.ROSInterruptException
    pass

