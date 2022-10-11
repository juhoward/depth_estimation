from math import sqrt, dist
import numpy as np

class FaceDet(object):
    '''
    Class representing a detected face & holds the following data:
    --mesh
    --head measurements
    --distance calculations
    --camera parameters
    Must be initialized by calculating the focal length (f) of the camera.
    Right now, f is calculated using the endpoints of a credit card (85.6mm width)
    held 20 (in.) from the webcam.
    '''
    def __init__(self, d_2_obj, points):
        # credit card width (mm)
        self.w_card = 85.6
        # mean human iris diameter (mm)
        self.w_iris = 11.7
        # first, calculate focal length
        # euclidean distance between pts
        self.w_pix = dist(points[:2],  points[2:])
        # converts an initial distance in inches to mm
        self.d_2_obj = self.in_to_mm(d_2_obj)
        # computes focal length of camera
        self.f = self.f_length()
        self.l_iris = {'center': None, 'radius': None}
        self.r_iris = {'center': None, 'radius': None}
        # mediapipe face mesh
        self.mesh = None
        # midiapipe head pts
        self.head_pts = None
        # head width (mm) based on iris diameter (mm)
        self.head_w = 0
        # head pixel width
        self.head_pixw = 0
        # holds head measurements
        self.head_measurements = []
        # subject-to-camera distance (in)
        self.s2c_d = 0
        # s2c (cm)
        self.s2c_ds = []
        # average relative inverse depth (iris, head)
        self.ri_depth = 0
        self.ri_depths = []
        # converted absolute depth
        self.abs_depth = 0
        self.abs_depths = []
        #  errors
        self.error = 0
        self.errors = []


    def f_length(self):
        ''' 
        returns the focal length based on triangle similarity.
        d_2_obj : known distance to the object
        w_card : known width of object in mm
        w_pix : distance in pixels
        TODO: test change in w_card to iris diameter
        '''
        return (self.d_2_obj * self.w_pix) / self.w_card

    def s2c_dist(self, w_object, w_pix):
        '''
        returns the subject-to-camera distance in mm using triangle similarity.
        f : known focal length in mm
        w_object : known width of object in mm
        w_pix : distance in pixels
        '''
        s2c_d = (self.f * w_object) / w_pix
        # transform mm to cm
        s2c_d /= 10
        # log metric distance (cm) for parameter estimation
        self.s2c_ds.append(s2c_d)
        # return distance in ft
        s2c_d = self.cm_to_ft(s2c_d)
        # keep state for reporting
        self.s2c_d = s2c_d


    def get_headw(self, p1, p2, logging=True):
        '''
        takes cheek points from facemesh &
        returns the width (mm) of the head based on the iris detection.
        appends the head width in a list for later use. 
        '''
        self.head_pts = (p1, p2)
        # head width in pixels
        self.head_pixw = dist((p1[0], p1[1]), (p2[0], p2[1]))
        # horizontal distance in mm/pixel units : iris plane
        self.head_w = (self.head_pixw * self.w_iris) / (self.l_iris['radius'] * 2)
        if logging:
            self.head_measurements.append(self.head_w)
    
    def get_depth(self, img):
        '''
        returns the average relative inverse depth of 2 depth pixels.
        '''
        if self.mesh is not None:
            # if face detected, use iris location depth
            l_ctr = list(map(lambda x: int(x), self.l_iris['center']))
            r_ctr = list(map(lambda x: int(x), self.r_iris['center']))
            for idx, (i,j) in enumerate(zip(l_ctr, r_ctr)):
                if idx == 0:
                    l_ctr[idx] = min(img.shape[0]-1, i)
                    r_ctr[idx] = min(img.shape[0]-1, j)
                else:
                    l_ctr[idx] = min(img.shape[1]-1, i)
                    l_ctr[idx] = min(img.shape[1]-1, j)                  
            left = img[l_ctr[0],l_ctr[1]]
            right = img[r_ctr[0], r_ctr[1]]
            ri_depth = (left + right) / 2
            self.ri_depth = ri_depth
            self.ri_depths.append(ri_depth)
        else:
            d_left = img[self.head_pts[0][0], self.head_pts[0][1]]
            d_right = img[self.head_pts[1][0], self.head_pts[1][1]]
            ri_depth = (d_left + d_right) / 2
            self.ri_depth = ri_depth
            self.ri_depths.append(ri_depth)

    def rel2abs_2(self, pred_depths, gt_depths):
        '''
        given dataset of relative inverse depths and gt_depths (cm),
        finds a linear relationship in form pred = mx + b
        returns absolute depth (cm).
        '''
        # invert gt
        gt = list(map(lambda x: 1/x, gt_depths))
        # align prediction based on least squares estimates
        A = np.vstack([gt, np.ones(len(gt))]).T
        self.m, self.b = np.linalg.lstsq(A, pred_depths, rcond=None)[0]
        # transform to ft
        self.abs_depth = self.cm_to_ft(self.ri_depth * self.m + self.b)
    
    def rel2abs(self):
        '''
        a simple linear transformation.
        division by 2.54 to convert to inches.
        '''
        abs_depth = self.ri_depth / 2.54
        self.abs_depth = abs_depth
        self.abs_depths.append(abs_depth)
    
    def rmse(self):
        '''
        returns rmse of converted abs depths and s2c distsances.
        '''
        errors = list(map(lambda x: (self.cm_to_ft(x[0]) - x[1])**2, zip(self.s2c_ds, self.abs_depths)))
        return sqrt((sum(errors)/ len(errors)))

    def mae(self):
        '''
        returns mean absolute error of converted abs depth and s2c distances
        '''
        errors = list(map(lambda x: abs(self.cm_to_ft(x[0]) - x[1]), zip(self.s2c_ds, self.abs_depths)))
        return sum(errors) / len(errors)

    def mm2cm(self, dist):
        return dist/10

    def cm_to_ft(self, dist):
        return round(dist/(2.54*12), 2)

    def in_to_mm(self, dist):
        return round(dist * 2.54 * 10, 2)

    def diameter(self, radius):
        return int(radius * 2)

    def dist_euclid(self, pt1:tuple, pt2:tuple):
        return sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)