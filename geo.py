#!/usr/bin/env python

import rospy
import math
from sensor_msgs.msg import NavSatFix

a = 6378137.0
e_sq = 6.6943799901377997e-3

def geodetic_to_ecef(data):
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

    print("xNorth: ", xNorth, "yWest: ", yWest, "zUp: ", zUp)

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
    yNorth = (-cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd) - 9
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    xNorth = yNorth
    yWest = -xEast

    return xNorth, yWest, zUp


def gpsListener():
	rospy.init_node('geo', anonymous = True)
	rospy.Subscriber("fix", NavSatFix, geodetic_to_ecef)
	rospy.spin()

if __name__ == '__main__':
	gpsListener()