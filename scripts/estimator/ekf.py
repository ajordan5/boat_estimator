import numpy as np
from scipy.spatial.transform import Rotation as R

def propagate(belief,RProcess,RInputs,ft,At,Bt,dt):
     belief.p = belief.p + ft.dp*dt
     belief.vr = belief.vr + ft.dvr*dt
     belief.psi = belief.psi + ft.dpsi*dt
     belief.vb = belief.vb + ft.dvb*dt

     Ad = np.identity(belief.m) + At*dt
     Bd = Bt*dt

     belief.P = Ad@belief.P@Ad.T + Bd@RInputs@Bd.T + RProcess*dt**2

def update(belief,Qt,zt,ht,Ct):
     Lt = belief.P@Ct.T@np.linalg.inv(Ct@belief.P@Ct.T+Qt)
     dx = Lt@(zt-ht)
     belief.p = belief.p + dx[0:3]
     belief.vr = belief.vr + dx[3:6]
     belief.psi = belief.psi + dx[6]
     belief.vb = belief.vb + dx[7:10]

     belief.P = (np.identity(belief.m) - Lt@Ct)@belief.P

def update_dynamic_model(belief,ut):
     euler = np.array([ut.phi,ut.theta,belief.psi])
     Rb2i = R.from_euler('xyz',euler.squeeze())
     Ri2b = Rb2i.inv()

     sphi = np.sin(ut.phi)
     cphi = np.cos(ut.phi)
     cth = np.cos(ut.theta)

     gravity = np.array([[0.0,0.0,9.81]]).T

     dp = Rb2i.apply(belief.vb.T).T - belief.vr
     dvr = np.zeros((3,1))
     dpsi = sphi/cth*ut.gyros.item(1) + cphi/cth*ut.gyros.item(2)
     dvb = ut.accelerometers + Ri2b.apply(gravity.T).T - np.cross(ut.gyros.T,belief.vb.T).T
     ft = np.concatenate((dp,dvr,dpsi,dvb),axis=0)

     return ft

def update_rtk_relPos_model(relPosHat,eulerAnglesHat,antennaOffset):
     Rb2i = R.from_euler('xyz',eulerAnglesHat.squeeze())
     ht = -relPosHat + Rb2i.apply(antennaOffset.T).T
     return ht

def update_rover_gps_velocity_model(roverVelocityHat):
     ht = roverVelocityHat
     return ht

def update_base_gps_velocity_model(eulerAnglesHat,baseVelocityHat,wLpf,antennaOffset):
     Rb2i = R.from_euler('xyz',eulerAnglesHat.squeeze())
     ht = Rb2i.apply(baseVelocityHat.T).T - np.cross(wLpf.T,antennaOffset.T).T
     return ht

def update_rtk_compass_model(psiHat,zPsi):
     if zPsi-psiHat > np.pi:
          ht = psiHat + 2.0*np.pi
     elif zPsi-psiHat < -np.pi:
          ht = psiHat - 2.0*np.pi
     else:
          ht = psiHat
     return ht

def update_apriltag_model(p, Rc2i, cameraOffset):
     # Expected output of apriltag reading from multirotor with attitude Rm and translation between camera and multirotor tm
     Ri2c = Rc2i.inv()
     ht = Ri2c.apply(p.T).T - cameraOffset
     # print("h", ht.T)
     # print("p", p.T)
     return ht

def update_jacobian_A(belief,ut):
     sphi = np.sin(ut.phi).squeeze()
     cphi = np.cos(ut.phi).squeeze()
     sth = np.sin(ut.theta).squeeze()
     cth = np.cos(ut.theta).squeeze()
     spsi = np.sin(belief.psi).squeeze()
     cpsi = np.cos(belief.psi).squeeze()

     euler = np.array([ut.phi,ut.theta,belief.psi]).T
     Rb2i = R.from_euler('xyz',euler.squeeze())

     dpDp = np.zeros((3,3))
     dpDvr = -np.identity(3)
     dpDpsi = np.array([[(-cth*sphi)*belief.vb.item(0) + (-sphi*sth*spsi-cphi*cpsi)*belief.vb.item(1) + (-cphi*sth*spsi+sphi*cpsi)*belief.vb.item(2)],
                        [(cth*cpsi)*belief.vb.item(0) + (sphi*sth*cpsi-cphi*spsi)*belief.vb.item(1) + (cphi*sth*cpsi+sphi*spsi)*belief.vb.item(2)],
                        [0.0]])
     dpDvb = Rb2i.as_matrix()
     dpDx = np.concatenate((dpDp,dpDvr,dpDpsi,dpDvb),axis=1)

     dvrDp = np.zeros((3,3))
     dvrDvr = np.zeros((3,3))
     dvrDpsi = np.zeros((3,1))
     dvrDvb = np.zeros((3,3))
     dvrDx = np.concatenate((dvrDp,dvrDvr,dvrDpsi,dvrDvb),axis=1)

     dpsiDp = np.zeros((1,3))
     dpsiDvr = np.zeros((1,3))
     dpsiDpsi = np.zeros((1,1))
     dpsiDvb = np.zeros((1,3))
     dpsiDx = np.concatenate((dpsiDp,dpsiDvr,dpsiDpsi,dpsiDvb),axis=1)

     dvbDp = np.zeros((3,3))
     dvbDvr = np.zeros((3,3))
     dvbDpsi = np.zeros((3,1))
     dvbDvb = np.array([[0.0,ut.gyros.item(2),-ut.gyros.item(1)],
                        [-ut.gyros.item(2),0.0,ut.gyros.item(0)],
                        [ut.gyros.item(1),-ut.gyros.item(0),0.0]])
     dvbDx = np.concatenate((dvbDp,dvbDvr,dvbDpsi,dvbDvb),axis=1)

     At = np.concatenate((dpDx,dvrDx,dpsiDx,dvbDx),axis=0)
     return At
    
def update_jacobian_B(belief,ut):
     sphi = np.sin(ut.phi).squeeze()
     cphi = np.cos(ut.phi).squeeze()
     sth = np.sin(ut.theta).squeeze()
     cth = np.cos(ut.theta).squeeze()
     tth = np.tan(ut.theta).squeeze()
     spsi = np.sin(belief.psi).squeeze()
     cpsi = np.cos(belief.psi).squeeze()

     gravity = np.array([[0.0,0.0,9.81]]).T

     dpDa = np.zeros((3,3))
     dpDw = np.zeros((3,3))
     dpDphi = np.array([[(cphi*sth*cpsi+sphi*spsi)*belief.vb.item(1) + (-sphi*sth*cpsi+cphi*spsi)*belief.vb.item(2)],
                        [(cphi*sth*spsi-sphi*cpsi)*belief.vb.item(1) + (-sphi*sth*spsi-cphi*cpsi)*belief.vb.item(2)],
                        [(cphi*cth)*belief.vb.item(1) + (-sphi*cth)*belief.vb.item(2)]])
     dpDth = np.array([[(-sth*cpsi)*belief.vb.item(0) + (sphi*cth*cpsi)*belief.vb.item(1) + (cphi*cth*cpsi)*belief.vb.item(2)],
                       [(-sth*spsi)*belief.vb.item(0) + (sphi*cth*spsi)*belief.vb.item(1) + (cphi*cth*spsi)*belief.vb.item(2)],
                       [(-cth)*belief.vb.item(0) + (-sphi*sth)*belief.vb.item(1) + (-cphi*sth)*belief.vb.item(2)]])
     dpDu = np.concatenate((dpDa,dpDw,dpDphi,dpDth),axis=1)

     dvrDa = np.zeros((3,3))
     dvrDw = np.zeros((3,3))
     dvrDphi = np.zeros((3,1))
     dvrDth = np.zeros((3,1))
     dvrDu = np.concatenate((dvrDa,dvrDw,dvrDphi,dvrDth),axis=1)

     dpsiDa = np.zeros((1,3))
     dpsiDw = np.array([[0.0, sphi/cth, cphi*cth]])
     dpsiDphi = np.array([[cphi/cth*ut.gyros.item(1) - sphi/cth*ut.gyros.item(2)]])
     dpsiDth = np.array([[sphi*tth/cth*ut.gyros.item(1) + cphi*tth/cth*ut.gyros.item(2)]])
     dpsiDu = np.concatenate((dpsiDa,dpsiDw,dpsiDphi,dpsiDth),axis=1)

     dvbDa = np.identity(3)
     dvbDw = np.array([[0.0, -belief.vb.item(2), belief.vb.item(1)],
                      [belief.vb.item(2), 0.0, -belief.vb.item(0)],
                      [-belief.vb.item(1), belief.vb.item(0), 0.0]])
     dvbDphi = np.array([[0.0, cphi*cth, -sphi*cth]]).T*gravity.item(2)
     dvbDth = np.array([[-cth, -sphi*sth, -cphi*sth]]).T*gravity.item(2)
     dvbDu = np.concatenate((dvbDa,dvbDw,dvbDphi,dvbDth),axis=1)

     Bt = np.concatenate((dpDu, dvrDu, dpsiDu, dvbDu), axis=0)

     return Bt

def get_jacobian_C_relPos(baseStates,antennaOffset):

     dyDp = -np.identity(3)
     dyDvr = np.zeros((3,3))
     dyDpsi = Rb2i_dot(baseStates.euler) @ antennaOffset
     dyDvb = np.zeros((3,3))
     Ct = np.concatenate((dyDp,dyDvr,dyDpsi,dyDvb),axis=1)
     
     return Ct

def get_jacobian_C_rover_velocity():
     dyDp = np.zeros((3,3))
     dyDvr = np.identity(3)
     dyDpsi = np.zeros((3,1))
     dyDvb = np.zeros((3,3))
     Ct = np.concatenate((dyDp,dyDvr,dyDpsi,dyDvb),axis=1)

     return Ct

def get_jacobian_C_base_velocity(euler,vb):
     Rb2i = R.from_euler('xyz',euler.squeeze())
     dyDp = np.zeros((3,3))
     dyDvr = np.zeros((3,3))
     dyDpsi = Rb2i_dot(euler) @ vb
     dyDvb = Rb2i.as_matrix()
     Ct = np.concatenate((dyDp,dyDvr,dyDpsi,dyDvb),axis=1)

     return Ct

def get_jacobian_C_compass():
     dyDp = np.zeros((1,3))
     dyDvr = np.zeros((1,3))
     dyDpsi = np.array([[1.0]])
     dyDvb = np.zeros((1,3))
     Ct = np.concatenate((dyDp,dyDvr,dyDpsi,dyDvb),axis=1)

     return Ct

def get_jacobian_C_apriltag(Rc2i, euler, p):
     zero = np.zeros((3,3))
     dyDvr = zero
     dyDvb = zero
     dyDp = Rc2i.as_matrix().T
     # dyDpsi = Rb2i_dot(euler) @ p
     dyDpsi = Ri2c_dot(euler) @ p
     Ct = np.concatenate((dyDp,dyDvr,dyDpsi,dyDvb),axis=1)
     return Ct

def Rb2i_dot(euler):
     # Derivative of the roation matrix from body to inertial for the boat
     sphi = np.sin(euler.item(0))
     cphi = np.cos(euler.item(0))
     sth = np.sin(euler.item(1))
     cth = np.cos(euler.item(1))
     spsi = np.sin(euler.item(2))
     cpsi = np.cos(euler.item(2))

     return np.array([[(-cth*sphi) , (-sphi*sth*spsi-cphi*cpsi) , (-cphi*sth*spsi+sphi*cpsi)],
                        [(cth*cpsi) , (sphi*sth*cpsi-cphi*spsi) , (cphi*sth*cpsi+sphi*spsi)],
                        [0.0 , 0.0, 0.0]])


def Ri2c_dot(euler):
     # Derivative of the roation matrix from body to inertial for the boat
     sphi = np.sin(euler.item(0))
     cphi = np.cos(euler.item(0))
     sth = np.sin(euler.item(1))
     cth = np.cos(euler.item(1))
     spsi = np.sin(euler.item(2))
     cpsi = np.cos(euler.item(2))

     # return np.array([[(-sphi*sth*spsi-cphi*cpsi) , (sphi*sth*cpsi-cphi*spsi) , 0.0 ],
     #                    [(cth*spsi) , (-cth*cpsi) , 0.0],
     #                    [ (-cphi*sth*spsi+sphi*cpsi) , (cphi*sth*cpsi+sphi*spsi) , 0.0]])
     return np.array([[(sphi*sth*spsi+cphi*cpsi) , (-sphi*sth*cpsi+cphi*spsi) , 0.0 ],
                        [(-cth*spsi) , (cth*cpsi) , 0.0],
                        [ (-cphi*sth*spsi+sphi*cpsi) , (cphi*sth*cpsi+sphi*spsi) , 0.0]])

