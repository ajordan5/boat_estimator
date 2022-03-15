from rosbag_parser import Parser
from IPython.core.debugger import set_trace 
import matplotlib.pyplot as plt
import rosbag
import numpy as np
import navpy

def main():
	odomTopic = '/new/base_odom'
	truthTopic = '/dummyTopic'
	mocapTopic = '/dummyTopic'
	imuTopic = '/base/imu'
	ubloxRelPosTopic = '/rover/RelPos'
	baseGpsTopic = '/base/PosVelEcef'
	roverGpsTopic = '/rover/PosVelEcef'
	rtkCompassTopic = '/base/compass/RelPos'
	data = Parser(odomTopic,truthTopic,mocapTopic,imuTopic,ubloxRelPosTopic,baseGpsTopic,roverGpsTopic,rtkCompassTopic)
	filenames = ['compare_roscopter1.bag','compare_roscopter2.bag']
	bag1 = rosbag.Bag('/home/matt/data/px4flight/sim/' + filenames[0])
	bag2 = rosbag.Bag('/home/matt/data/px4flight/sim/' + filenames[1])

	estRelPos1,estOdom1,ubloxRelPos1,baseGps1,refLla1 = get_data(data,bag1)
	estRelPos2,estOdom2,ubloxRelPos2,baseGps2,refLla2 = get_data(data,bag2)
	baseGpsNed = ecef2ned(baseGps1,refLla1)

	get_pn_data(ubloxRelPos1,estRelPos1,estRelPos2)
	get_pe_data(ubloxRelPos1,estRelPos1,estRelPos2)
	get_pd_data(ubloxRelPos1,estRelPos1,estRelPos2)
	get_ub_data(baseGpsNed,estOdom1,estOdom2)
	get_vb_data(baseGpsNed,estOdom1,estOdom2)
	get_wb_data(baseGpsNed,estOdom1,estOdom2)
	get_attitude_data(estOdom1,estOdom2)

	plt.show()

def get_data(data, bag):
	estRelPos, estOdom = data.get_odom(bag)
	estRelPos.time -= estRelPos.time[0]
	estOdom.time -= estOdom.time[0]

	measuredRelPos = data.get_ublox_relPos(bag)
	measuredRelPos.time -= measuredRelPos.time[0]

	baseGps = data.get_base_gps(bag)
	baseGps.time -= baseGps.time[0]

	roverGps,refLla = data.get_rover_gps(bag)

	return estRelPos, estOdom, measuredRelPos, baseGps, refLla

def ecef2ned(gps,refLla):
	ecefOrigin1D = navpy.lla2ecef(refLla.lat,refLla.lon,refLla.alt)
	ecefOrigin = np.array([ecefOrigin1D]).T
	ecefPositionLocal = gps.position - ecefOrigin
	gpsPositionNed = navpy.ecef2ned(ecefPositionLocal,refLla.lat,refLla.lon,refLla.alt)
	gpsVelocityNed = navpy.ecef2ned(gps.velocity,refLla.lat,refLla.lon,refLla.alt)
	gpsNed = GpsNed(gps.time,gpsPositionNed,gpsVelocityNed)
	return gpsNed

def get_pn_data(ubloxRelPos, estRelPos1, estRelPos2):
	fig_num = 1
	plot_3(fig_num, ubloxRelPos.time, -ubloxRelPos.position[0], 'ublox n', estRelPos1.time, estRelPos1.position[0], 'estimate1 n', estRelPos2.time, estRelPos2.position[0], 'estimate2 n')

def get_pe_data(ubloxRelPos, estRelPos1, estRelPos2):
	fig_num = 2
	plot_3(fig_num, ubloxRelPos.time, -ubloxRelPos.position[1], 'ublox e', estRelPos1.time, estRelPos1.position[1], 'estimate1 e', estRelPos2.time, estRelPos2.position[1], 'estimate2 e')

def get_pd_data(ubloxRelPos, estRelPos1, estRelPos2):
	fig_num = 3
	plot_3(fig_num, ubloxRelPos.time, -ubloxRelPos.position[2], 'ublox d', estRelPos1.time, estRelPos1.position[2], 'estimate1 d', estRelPos2.time, estRelPos2.position[2], 'estimate2 d')

def get_ub_data(odom, gpsNed1, gpsNed2):
	fig_num = 4
	plot_3(fig_num, odom.time, odom.velocity[0], 'odom u', gpsNed1.time, gpsNed1.velocity[0], 'gps1 u', gpsNed2.time, gpsNed2.velocity[0], 'gps2 u')

def get_vb_data(odom, gpsNed1, gpsNed2):
	fig_num = 5
	plot_3(fig_num, odom.time, odom.velocity[1], 'odom v', gpsNed1.time, gpsNed1.velocity[1], 'gps1 v', gpsNed2.time, gpsNed2.velocity[1], 'gps2 v')

def get_wb_data(odom, gpsNed1, gpsNed2):
	fig_num = 6
	plot_3(fig_num, odom.time, odom.velocity[2], 'odom w', gpsNed1.time, gpsNed1.velocity[2], 'gps1 w', gpsNed2.time, gpsNed2.velocity[2], 'gps2 w')

def get_attitude_data(odom1,odom2):
	fig_num = 7
	plot_6(fig_num, odom1.time, odom1.euler, 'phi1 deg', 'theta1 deg', 'psi1 deg', odom2.time, odom2.euler, 'phi2 deg', 'theta2 deg', 'psi2 deg')

def plot_3(fig_num, t_x, x, xlabel, t_y, y, ylabel, t_z, z, zlabel):
	plt.figure(fig_num)
	plt.plot(t_x, x, label = xlabel)
	plt.plot(t_y, y, label = ylabel)
	plt.plot(t_z, z, label = zlabel)
	plt.legend(loc = "upper right")

def plot_6(fig_num, t_x, x3, labelx1, labelx2, labelx3, t_y, y3, labely1, labely2, labely3):
	plt.figure(fig_num)
	plt.plot(t_x,x3[0],label = labelx1)
	plt.plot(t_x,x3[1],label = labelx2)
	plt.plot(t_x,x3[2],label = labelx3)
	plt.plot(t_y,y3[0],label = labely1)
	plt.plot(t_y,y3[1],label = labely2)
	plt.plot(t_y,y3[2],label = labely3)
	plt.legend(loc = "upper right")

class GpsNed:
	def __init__(self,time,position,velocity):
		self.time = time
		self.position = position.T
		self.velocity = velocity.T

if __name__ == '__main__':
	main()
