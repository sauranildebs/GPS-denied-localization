#!/usr/bin/env python

from ukf import UKF
import numpy as np
import math
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path

a = 6378137.0
e_sq = 6.6943799901377997e-3

true_path = Path()
predicted_path = Path()


#state transition matrix
def iterate_x(x_in, timestep):

	global roll, pitch, yaw, predicted_path, predicted_path_pub


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

	rate = rospy.Rate(500)

	pose = PoseStamped()
	predicted_path.header.frame_id = 'world'
	pose.header.stamp = rospy.Time.now()
	pose.header.frame_id = 'world'
	pose.pose.position.x = x_in[0]
	pose.pose.position.y = x_in[1]
	pose.pose.position.z = x_in[2]
	predicted_path.poses.append(pose)
	predicted_path_pub.publish(predicted_path)
	
	rate.sleep()

	return ret


def imu_receiver(data):

	global inital_timestamp, roll, pitch, yaw, state_estimator, r_imu, a_imu, w_imu, orientation_list

	orientation_q = data.orientation
	orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
	(roll, pitch, yaw) = euler_from_quaternion(orientation_list)
	

	imu_data = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z, data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])

	timestep = (data.header.stamp.secs + (data.header.stamp.nsecs * 1e-9)) - inital_timestamp

	inital_timestamp = data.header.stamp.secs + (data.header.stamp.nsecs * 1e-9)

	a_imu = np.array([data.linear_acceleration.x, data.linear_acceleration.y, data.linear_acceleration.z])
	w_imu = np.array([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])

	state_estimator.predict(timestep)

	state_estimator.update([9, 10, 11, 12, 13, 14], imu_data, r_imu)



def geodetic_to_ecef(data):
	global state_estimator, r_gps

	lamb = math.radians(data.latitude)
	phi = math.radians(data.longitude)
	s = math.sin(lamb)
	N = a / math.sqrt(1 - e_sq * s * s)

	sin_lambda = math.sin(lamb)
	cos_lambda = math.cos(lamb)
	sin_phi = math.sin(phi)
	cos_phi = math.cos(phi)

	x = (data.altitude + N) * cos_lambda * cos_phi
	y = (data.altitude + N) * cos_lambda * sin_phi
	z = (data.altitude + (1 - e_sq) * N) * sin_lambda

	xNorth, yWest, zUp = ecef_to_enu(x, y, z, 49.860246, 8.687077, 0.0)
	gps_data = [xNorth, yWest, zUp]
	state_estimator.update([0, 1, 2], gps_data, r_gps)


def ecef_to_enu(x, y, z, lat0, lon0, h0):
	lamb = math.radians(lat0)
	phi = math.radians(lon0)
	s = math.sin(lamb)
	N = a / math.sqrt(1 - e_sq * s * s)

	sin_lambda = math.sin(lamb)
	cos_lambda = math.cos(lamb)
	sin_phi = math.sin(phi)
	cos_phi = math.cos(phi)

	x0 = (h0 + N) * cos_lambda * cos_phi
	y0 = (h0 + N) * cos_lambda * sin_phi
	z0 = (h0 + (1 - e_sq) * N) * sin_lambda

	xd = x - x0
	yd = y - y0
	zd = z - z0

	xEast = -sin_phi * xd + cos_phi * yd
	yNorth = (-cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd) - 9  #offset
	zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

	xNorth = yNorth
	yWest = -xEast

	return xNorth, yWest, zUp

def ground_truth(data):
	global true_path, true_path_pub
	true_path.header = data.header
	pose = PoseStamped()
	pose.header = data.header
	pose.pose = data.pose.pose
	true_path.poses.append(pose)
	true_path_pub.publish(true_path)


def main():
	global inital_timestamp, state_estimator, r_imu, R_prev, r_gps, true_path_pub, predicted_path_pub

	inital_timestamp = 0.0
	R_prev = np.zeros((3, 3))

	q = 0.0001 * np.eye(15)   #process noise covariance Q - needs to be updated for adaptive UKF
	
	r_imu = 0.0001 * np.eye(6)   #IMU measurement noise covariance - R - needs to be updated for adaptive UKF

	r_gps = 0.000001 * np.eye(3)	#GPS measurement noise covariance  needs to be updated for adaptive UKF


	state_estimator = UKF(15, q, np.zeros(15), 0.000001 * np.eye(15), 0.001, 0, 2, iterate_x)

	rospy.init_node('mainSim', anonymous = True)
	rospy.Subscriber('raw_imu', Imu, imu_receiver)
	rospy.Subscriber("fix", NavSatFix, geodetic_to_ecef)
	rospy.Subscriber('ground_truth/state', Odometry, ground_truth)
	true_path_pub = rospy.Publisher('true_path', Path, queue_size = 10)
	predicted_path_pub = rospy.Publisher('predicted_path', Path, queue_size = 10)
	rospy.spin()

if __name__ == "__main__":
	main()