''' 
class & functions for credit card bounding box validation.
a box is considered valid if its diagonal overlaps with the
line segment representing the keypoints of the head or iris 

'''
import numpy as np

class Validator(object):
    def __init__(self, a1, a2, box) -> None:
        self.a1 = a1
        self.a2 = a2
        self.box = box
        self.b1, self.b2 = self.get_diagonal()

    def get_point(self):
        """ 
        Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([self.a1,self.a2,self.b1,self.b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return (x/z, y/z)

    def get_diagonal(self):
        '''
        uses bounding box to return diagonal points.
        returns topleft and bottom right bounding box coordinates.
        '''
        # bottome right point
        br = self.box[0]
        # top left
        tl = self.box[2]
        return tl, br
    
    def check_intersect(self):
        '''Checks if the path of a tracked object crossed a line.
            b_point: box centroid
            p_b_point: previous box centroid
            line_coordinates: two points representing the analytic line
            intersection: True if the target crossed the line
        '''
        intersection = self.get_point()
        if intersection == ('inf', 'inf'):
            return False
        x_t = [self.b1[0], self.b2[0]]
        y_t = [self.b1[1], self.b2[1]]
        # check intersection against target line
        if intersection[0] >= min(x_t) and intersection[0] <= max(x_t):
            if intersection[1] >= min(y_t) and intersection[1] <= max(y_t):
                return True
            else:
                False