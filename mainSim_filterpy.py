#!/usr/bin/env python

import numpy as np
import math
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from tf.transformations import euler_from_quaternion
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

a = 6378137.0
e_sq = 6.6943799901377997e-3


#state transition matrix
def iterate_x(x_in, timestep):

	global roll, pitch, yaw

	T = np.array([[1, math.sin(roll) * math.tan(pitch), math.cos(roll) * math.tan(pitch)],
		[0, math.cos(roll), -math.sin(roll)],
		[0, math.sin(roll) / math.cos(pitch), math.cos(roll) / math.cos(pitch)]])


	
	#Rotation matrix using euler angles to convert from body frame to world ENU frame
	R = np.array([[math.cos(yaw) * math.cos(pitch), math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.cos(roll) * math.sin(yaw), math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll)], 
		[math.cos(pitch) * math.sin(yaw), math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll), math.cos(roll) * math.sin(yaw) * math.sin(pitch) - math.cos(yaw) * math.sin(roll)],
		[-math.sin(pitch), math.cos(pitch) * math.cos(roll), math.cos(roll) * math.cos(pitch)]])

	g = [0, 0, 9.80665]

	g_enu = np.dot(R, g)


	a_prev = x_in[12:]    #previous linear acceleration in body frame 
	a_prev_enu = np.dot(R, a_prev)  #previous linear acceleration in world frame


	ret = np.zeros(len(x_in))

	ret[0] = x_in[0] + x_in[6] * timestep + 0.5 * a_prev_enu[0] * pow(timestep, 2)   # x = u_x*t + 0.5 * a * t^2  (x position in world frame)
	ret[1] = x_in[1] + x_in[7] * timestep + 0.5 * a_prev_enu[1] * pow(timestep, 2)	# y = u_y * t + 0.5 * a * t^2  (y position in world frame)
	ret[2] = x_in[2] + x_in[8] * timestep + 0.5 * (a_prev_enu[2] - g_enu[2]) * pow(timestep, 2)	  # z = u_z * t + 0.5 * a * t^2 (z position in world frame) 
	ret[3] = x_in[3] + timestep * x_in[9]		#roll in body frame
	ret[4] = x_in[4] + timestep * x_in[10]		#pitch in body frame
	ret[5] = x_in[5] + timestep * x_in[11]		#yaw in body frame
	ret[6] = x_in[6] + timestep * a_prev_enu[0]		# x velocity in world frame  v_x = u_x + a * t
	ret[7] = x_in[7] + timestep * a_prev_enu[1]		# y velocity in world frame	v_y = u_y + a * t
	ret[8] = x_in[8] + timestep * (a_prev_enu[2] - g_enu[2]) 	#z velocity in world frame v_z = u_z + a * t
	ret[9] = x_in[9]
	ret[10] = x_in[10]
	ret[11] = x_in[11]
	ret[12] = x_in[12]
	ret[13] = x_in[13]
	ret[14] = x_in[14]

	print(x_in[0], x_in[1], x_in[2])

	return ret

def measurement_func(x_in):
	return x_in[[9, 10, 11, 12, 13, 14]]



def imu_receiver(data):

	global roll, pitch, yaw, ukf

	orientation_q = data.orientation
	orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
	(roll, pitch, yaw) = euler_from_quaternion(orientation_list)
	ukf.predict()
	imu_data = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z, data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
	ukf.update(imu_data)


def main():
	global state_estimator, r_imu, ukf

	timestep = 0.01
	
	r_imu = [0.01, 0.01, 0.01, 0.01, 0.01, 100.0]
	
	sigmas = MerweScaledSigmaPoints(n = 15, alpha = .1, beta=2., kappa = -12)
	ukf = UKF(dim_x = 15, dim_z = 6, fx = iterate_x, hx = measurement_func, dt = timestep, points = sigmas)

	ukf.x = np.zeros(15)
	ukf.R = np.diag([r_imu])
	ukf.P = 0.000001 * np.eye(15)
	ukf.Q = 0.0001 * np.eye(15)

	rospy.init_node('mainSim_filterpy', anonymous = True)
	rospy.Subscriber('raw_imu', Imu, imu_receiver)
	rospy.spin()

if __name__ == "__main__":
	main()