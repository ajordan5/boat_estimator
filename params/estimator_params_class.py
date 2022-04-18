import numpy as np
import rospy

class EstimatorParams:
    def __init__(self):
        self.p0 = np.array([rospy.get_param('~p0',[0.0,0.0,0.0])]).T
        self.vr0 = np.array([rospy.get_param('~vr0',[0.0,0.0,0.0])]).T
        self.euler0 = np.array([rospy.get_param('~euler0',[0.0,0.0,0.0])]).T
        self.psi0 = np.array([[self.euler0.item(2)]])
        self.vb0 = np.array([rospy.get_param('~vb0',[0.0,0.0,0.0])]).T

        sigma0 = rospy.get_param('~sigma0',[3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0,3.0])
        self.P0 = np.diag([
                    sigma0[0]**2,  # pn
                    sigma0[1]**2,  # pe
                    sigma0[2]**2,  # pd
                    sigma0[3]**2,  # vrn
                    sigma0[4]**2,  # vre
                    sigma0[5]**2,  # vrd
                    np.radians(sigma0[6])**2,  # psi
                    sigma0[7]**2,  # ub
                    sigma0[8]**2,  # vb
                    sigma0[9]**2,  # wb
                   ])

        sigmaProcess = rospy.get_param('~sigmaProcess',[0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003])
        self.RProcess = 1.0 * np.diag([
                        sigmaProcess[0]**2,  # pn
                        sigmaProcess[1]**2,  # pe
                        sigmaProcess[2]**2,  # pd
                        sigmaProcess[3]**2,  # vrn
                        sigmaProcess[4]**2,  # vre
                        sigmaProcess[5]**2,  # vrd
                        sigmaProcess[6]**2,  # psi
                        sigmaProcess[7]**2,  # u
                        sigmaProcess[8]**2,  # v
                        sigmaProcess[9]**2,  # w
                        ])

        sigmaInputs = rospy.get_param('~sigmaInputs',[0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003,0.0003])
        self.RInputs = 1.0 * np.diag([
                        sigmaInputs[0]**2,  # ax
                        sigmaInputs[1]**2,  # ay
                        sigmaInputs[2]**2,  # az
                        sigmaInputs[3]**2,  # p
                        sigmaInputs[4]**2,  # q
                        sigmaInputs[5]**2,  # r
                        sigmaInputs[6]**2,  # phi complimentary filter
                        sigmaInputs[7]**2,  # theta complimentary filter
                        ])

        sigmaGps = rospy.get_param('~sigmaGps',[0.4,0.7,0.4,0.02,0.04,0.02])
        self.QtRtk = np.diag([
                    sigmaGps[3]**2, # rtkN
                    sigmaGps[3]**2, # rtkE
                    sigmaGps[4]**2, # rtkD
                    ])

        self.QtGpsVelocity = np.diag([
                    sigmaGps[2]**2, # gpsVelocityN
                    sigmaGps[2]**2, # gpsVelocityE
                    sigmaGps[2]**2, # gpsVelocityD
                    ])

        self.QtRtkCompass = np.array([[sigmaGps[5]**2]]) # rtk compass heading
        
        sigmaApriltag = rospy.get_param('~sigmaApriltag', 0.05)
        self.QtApriltag = np.diag([
                    sigmaApriltag**2, # x
                    sigmaApriltag**2, # y
                    sigmaApriltag**2, # z
                    ])
        self.apriltagID = rospy.get_param('~apriltagID', 1)

        self.kp = rospy.get_param('~compFiltKp',0.1)
        self.ki = rospy.get_param('~compFiltKi',0.0)

        self.antennaOffset = np.array([rospy.get_param('~antennaOffset',[0.74,-0.73,0.11])]).T
        self.cameraOffset = np.array([rospy.get_param('~cameraOffset',[0.0,0.0,0.0])]).T
        self.cameraRotation = np.array(rospy.get_param('~cameraRotation',[0.0,0.0,0.0]))
        self.tagRotation = np.array(rospy.get_param('~tagRotation',[0.0,0.0,0.0]))

        self.accelBias = np.array([rospy.get_param('~accelBias',[0.0,0.0,0.0])]).T
        self.gyroBias = np.array([rospy.get_param('~gyroBias',[0.0,0.0,0.0])]).T