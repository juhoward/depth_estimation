from math import sqrt, dist


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
        # holds head measurements
        self.head_measurements = []
        # subject-to-camera distance
        self.s2c_d = 0
        # average depth
        self.av_depth = 0

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
        # return distance in ft
        self.s2c_d = self.cm_to_ft(s2c_d)
    
    def get_headw(self, p1, p2):
        '''
        takes cheek points from facemesh &
        returns the width (mm) of the head based on the iris detection.
        appends the head width in a list for later use. 
        '''
        self.head_pts = (p1, p2)
        # head width in pixels
        head_pixw = dist((p1[0], p1[1]), (p2[0], p2[1]))
        # horizontal distance in mm/pixel units : iris plane
        self.head_w = (head_pixw * self.w_iris) / (self.l_iris['radius'] * 2)
        self.head_measurements.append(self.head_w)
    
    def get_depth(self, img):
        '''
        returns the average of both 2 depth pixels.
        '''
        if self.mesh is not None:
            # if face detected, use iris location depth
            l_ctr = list(map(lambda x: int(x), self.l_iris['center']))
            r_ctr = list(map(lambda x: int(x), self.r_iris['center']))
            for idx, i,j in enumerate(zip(l_ctr, r_ctr)):
                if idx == 0:
                    l_ctr[idx] = min(self.w, l_ctr[idx])
                    r_ctr[idx] = min(self.w, r_ctr[idx])
                else:
                    l_ctr[idx] = min(self.h, l_ctr[idx])
                    l_ctr[idx] = min(self.h, l_ctr[idx])                  
            print(
                f'left: {l_ctr}, right:{r_ctr}, img shape: {img.shape}'
            )
            left = img[l_ctr[0],l_ctr[1]]
            right = img[r_ctr[0], r_ctr[1]]
            self.av_depth = (left + right) / 2
        else:
            d_left = img[self.head_pts[0][0], self.head_pts[0][1]]
            d_right = img[self.head_pts[1][0], self.head_pts[1][1]]
            self.av_depth = (d_left + d_right) / 2

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