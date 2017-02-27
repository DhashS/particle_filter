import rospy

import numpy as np
from scipy.spatial import ConvexHull

class OccupancyField(object):
    """ Stores an occupancy field for an input map.  An occupancy field returns the distance to the closest
        obstacle for any coordinate in the map
        Attributes:
            map: the map to localize against (nav_msgs/OccupancyGrid)
            closest_occ: the distance for each entry in the OccupancyGrid to the closest obstacle
    """

    def __init__(self, M):
        self.map = M      # save this for late
	self.array_map = np.array(self.map.data).reshape((self.map.info.width, self.map.info.height))
	self.occupied_cells = np.argwhere(self.array_map > 0) * self.map.info.resolution
	xs = self.occupied_cells.T[1] + self.map.info.origin.position.x
	ys = self.occupied_cells.T[0] + self.map.info.origin.position.y
	self.occupied_cells = np.atleast_2d([xs,ys]).T
	self.convex_hull = ConvexHull(self.occupied_cells)
	print self.convex_hull.vertices

    def get_closest_obstacle_distance(self, x, y):
	return np.min(np.linalg.norm(self.occupied_cells - np.array([x, y]), axis=1))
	
