#!/usr/bin/env python

import rospy
from std_msgs.msg import Header, String
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion, Point32
from nav_msgs.srv import GetMap 
from nav_msgs.msg import Odometry
import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix, quaternion_from_euler
import numpy as np
import scipy as sp
from copy import deepcopy
from scipy.stats import cauchy
from matplotlib.path import Path
from scipy.ndimage.interpolation import affine_transform, rotate
from my_occ_field import OccupancyField

from helper_functions import angle_diff

rospy.init_node('pf_burn')

class Particle():
    def __init__(self, x=0, y=0, theta=0, w=1):
	self.x = x
	self.y = y
	self.theta = theta
	self.w = w
	
    def as_pose(self):
	orientation = quaternion_from_euler(0, 0, self.theta)
	return Pose(position=Point(self.x, self.y, 0),
		    orientation=Quaternion(*orientation))

class ParticleFilter():
    """Sets up a particle filter
       some parameters are specified at runtime.
       ---Parameters---
       n_particles: the number of pose hypotheses
       map_frame; str, name of the map frame
       odometry_frame: str, name of the coordinate frame for relative updates
       base_frame = str, name of a body-centered coordinate frame
       d_thresh: float, smallest distance the robot moves before we do an update
       theta_thresh: float, smallest rotation before we do an update
       
       """
    def __init__(self):
	self.n_particles = rospy.get_param('~n_particles', 2000)
	self.map_frame = "map"
	self.odometry_frame = "odom"

	self.d_thresh = rospy.get_param('~d_thresh', 0.1)
	self.theta_thresh = rospy.get_param('~theta_thresh', 0.1)
	self.resample_conf = rospy.get_param('~resample', 0.2)

	self.last_odom_update_pose = PoseStamped(header=Header(stamp=rospy.get_rostime(),
							       frame_id="odom"),
						 pose=Pose(position=Point(0, 0, 0),
							   orientation=Quaternion(*quaternion_from_euler(0, 0, 0))))	

	#We need a map
	rospy.wait_for_service('static_map')
	try:
            map_pxy = rospy.ServiceProxy('static_map', GetMap)
	    self.map = map_pxy().map
	except rospy.ServiceException:
	    rospy.logfatal("No map aquired")
	
	#also some publishers
	#particle cloud
	self.beta_pub = rospy.Publisher('/alpha_pose/beta_list', PoseArray, queue_size=10)

	#best guess
	self.alpha_prime_pub = rospy.Publisher('/alpha_pose/prime', PoseStamped, queue_size=10)
		#Two inputs from the robot need to be handled by the particle filter
	#Position and laser scans
	rospy.Subscriber("/odom", Odometry, self.update_particles_with_odom)
	#rospy.Subscriber("/stable_scan", LaserScan, self.update_particles_with_laser)
    

	#a sprinkle of tf
	self.tf = TransformListener()
	self.tfb = TransformBroadcaster()

	#list of hypothesised poses
	self.betas = []
	self.alpha_prime = None
	
	#init particle filter
	self.occupancy_field = OccupancyField(self.map)
	self.make_transform()
	self.initialize_betas()

    def beta_in_hull(self):
	while True:
	     #square sampling
	    x, y = np.random.uniform(low=np.min(self.hull_pts, axis=0),
				     high=np.max(self.hull_pts, axis=0),
				     size=2)
	    #inside convex hull
	    if not self.poly.contains_point((x, y)):
		continue
	    
	    theta = np.random.uniform(low=0,
				      high=360-1)

	    return Particle(x, y, theta, w=1) #normalize later



    def initialize_betas(self):
	"""Builds the initial beta particle list.
	   Box-samples around the convex hull of the map."""
	#builds a closed polygon from convex hull vertices
	#vertices are indexes
	cvx_hull = self.occupancy_field.convex_hull
	hull_pts = np.array(cvx_hull.points[cvx_hull.vertices])
	
	#close the convex hull polygon with the first vertex
	hull_pts = np.vstack((hull_pts, hull_pts[0]))
	poly = Path(hull_pts)
	self.hull_pts = hull_pts
	self.poly = poly
	
	while len(self.betas) <= self.n_particles:
	    b = self.beta_in_hull()
	    self.betas.append(b)
	   	
	self.normalize_betas()

    def normalize_betas(self):
	"""Normalizes the w attribute of all Particles in self.betas"""
	weights = np.array([p.w for p in self.betas])
	weights = weights/np.sum(weights)
	
	mod_beta = []
	for i, p in enumerate(self.betas):
	    p.w = weights[i]
	    mod_beta.append(p)

	self.betas = mod_beta

    @staticmethod
    def unwrap(obj, attrs):
	"""helper getattr wrapper for data structure unwrap"""
	return [getattr(obj, x) for x in attrs]
	
    def update_particles_with_odom(self, msg):
	"""When the robot moves some relative distance bounded
	   by some parameters, update the particles for that motion

	   This is relative motion, and for small distances, holds (mostly) true.
	   It is based off wheel encoders, so there are several failure modes"""
	msg_loc = np.array(self.unwrap(msg.pose.pose.position, 'xy'))
	last_loc = np.array(self.unwrap(self.last_odom_update_pose.pose.position,'xy'))
	distance = msg_loc - last_loc
	if np.linalg.norm(distance) > self.d_thresh:
	    self.last_odom_update_pose = PoseStamped(header=Header(stamp=rospy.get_rostime(),
								   frame_id='odom'),
						     pose=Pose(position=msg.pose.pose.position,
							       orientation=self.last_odom_update_pose.pose.orientation))
	    new_betas = []
	    for p in self.betas:
		attrs = np.array(self.unwrap(p, 'xy'))
		attrs = attrs + distance
		new_betas.append(Particle(*attrs, theta=p.theta, w=p.w))
	    self.betas = new_betas
		
	msg_theta = np.array(self.unwrap(msg.pose.pose.orientation, 'xyzw'))
	last_theta = np.array(self.unwrap(self.last_odom_update_pose.pose.orientation,'xyzw'))
	#corresponds to 'z' axis rotation
	angle_delta = abs(angle_diff(msg_theta[2], last_theta[2]))
	if angle_delta > self.theta_thresh:
	    self.last_odom_update_pose = PoseStamped(header=Header(stamp=rospy.get_rostime(),
                                                		   frame_id='odom'),
                                                     pose=Pose(position=self.last_odom_update_pose.pose.position,
						     orientation=msg.pose.pose.orientation))
	    new_betas = []
	    for p in self.betas:
		new_betas.append(Particle(p.x, p.y, (p.theta + angle_delta) % 360, p.w))
	    self.betas = new_betas
	self.resample_particles()


    @staticmethod
    def laser_to_cloud(msg):
        scan = msg.ranges[:-1] #the last value is a repeated first value
        angles = np.array(range(len(scan))) * 180/np.pi
        xs = np.cos(angles) * scan
        ys = np.sin(angles) * scan
        points = [Point32(x,y,0) for x,y in zip(xs,ys) if not np.linalg.norm([x,y]) == 0.0] #drop all zero-distance readings
        cloud = PointCloud(header=Header(frame_id="/base_laser_link",
                                         stamp=msg.header.stamp),
                           points=points,
                           channels=ChannelFloat32(name="distance",
                                                   values=[d for d in scan if not d == 0.0]))
        return cloud



    def update_particles_with_laser(self, msg):
	#get scan points in cartesian
	pts = self.tf.transformPointCloud(self.odometry_frame, self.laser_to_cloud(msg)).points
	#coulumn vector
	pts = np.array([(p.x, p.y) for p in pts])
	weights = []
	betas = deepcopy(self.betas)
	for p in betas:
	   tformed_pts = rotate(pts, p.theta) + np.array([p.x, p.y])
	   xs, ys = tformed_pts.T
	   relwt = np.sum([self.occupancy_field.get_distance_to_closest_point(x, y) for x, y, in zip(xs, ys)])/len(xs.tolist())
	   if relwt <= 0:
	        print "Ruh roh"
	   p.w = p.w * relwt
	  
	self.betas = betas
	self.normalize_particle_weights()

    def normalize_particle_weights(self):
	betas = deepcopy(self.betas)
	ws = np.array([p.w for p in betas])
	ws = ws/np.sum(ws)
	for i, p in enumerate(betas):
	    p.w = ws[i]
	self.betas = betas
	self.resample_particles()
	
    def resample_particles(self):
	"""deletes invalid or unlikely particles, and samples new ones"""
	self.normalize_particle_weights()
	self.most_likely_particle()
	betas = deepcopy(self.betas)
	#if it's likely and within the hull
	good_particles = [p for p in betas if p.w > self.resample_conf and self.poly.contains_point((p.x, p.y))]
	#number of leftover particles / whatever's left in probability
	p_prob = (self.n_particles - len(good_particles))/1-np.sum([p.w for p in good_particles])
	while len(good_particles) >= self.n_particles:
	    b = self.beta_in_hull()
	    b.w = p_prob 
	    good_particles.append(b)
	self.betas = good_particles
	  
    def most_likely_particle(self):
	#mode of the distribution
	self.alpha_pose = self.betas[np.argmax([p.w for p in self.betas])].as_pose()
	self.alpha_prime_pub.publish(PoseStamped(header=Header(stamp=rospy.get_rostime(),
							      frame_id=self.map_frame),
						pose=self.alpha_pose))

    def make_transform(self, trans=None, rot=None):
	if trans == None:
	    trans = self.unwrap(self.map.info.origin.position, 'xyz')
	if rot == None:
	    rot = self.unwrap(self.map.info.origin.orientation, 'xyzw')

	self.tfb.sendTransform(trans,
			       rot,
			       rospy.Time.now(),
			       self.map_frame,
			       self.odometry_frame)
    def run(self):
	rate = rospy.Rate(5)
	while not rospy.is_shutdown():
	    self.beta_pub.publish(self.beta_as_posearray(self.betas))
	    self.make_transform()
	    rate.sleep()

	
    def beta_as_posearray(self, betas):
	return PoseArray(header=Header(stamp=rospy.get_rostime(),
				       frame_id=self.map_frame),
			 poses=[p.as_pose() for p in betas])



if __name__ == "__main__":
   pf = ParticleFilter()
   pf.run()
