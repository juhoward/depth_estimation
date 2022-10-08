import cv2
import mediapipe as mp
import torch
import numpy as np
from statistics import median
from math import sqrt, dist
from depth_midas import DepthEstimator


class PersonDetector(object):
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, face, minDetectionCon=0.2):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """
        # face mesh indices
        self.LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        # horizontal points (left, right), vertical points (top, bottom)
        self.HEAD = [234, 454, 10, 152]
        # body pose points
        self.BODY_HEAD = [7, 8]
        # raw coordinates for card from test data
        self.CARD = [505, 504, 675, 501]
        # mediapipe model config
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpface_mesh = mp.solutions.face_mesh
        # mediapipe model output
        self.results = None
        # image dimensions
        self.w = None
        self.h = None
        self.body_mesh = None
        # face, iris points & measurements
        self.face = face
    
    def findIris(self, img):
        '''
        Detect Irises of a single person in an image. 
        Returns a point mesh. 
        '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self.mpface_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
            ) as face_mesh:
                self.h, self.w = img.shape[:2]
                self.results = face_mesh.process(imgRGB)
                if self.results.multi_face_landmarks:
                    mesh_points = np.array(
                        [
                            np.multiply([p.x, p.y], [self.w, self.h]).astype(int) for p in self.results.multi_face_landmarks[0].landmark
                        ]
                    )
                    self.face.mesh = mesh_points
                    self.face.l_iris['center'], self.face.l_iris['radius'] = cv2.minEnclosingCircle(self.face.mesh[self.LEFT_IRIS])
                    self.face.r_iris['center'], self.face.r_iris['radius'] = cv2.minEnclosingCircle(self.face.mesh[self.RIGHT_IRIS])


    def findBody(self, img):
        '''
        Detect body.
        Returns ear to ear distance in pixels.
        '''
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_pose = mp.solutions.pose
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing = mp.solutions.drawing_utils
        with mp_pose.Pose(
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5) as pose:

            self.results = pose.process(img)
            # img.flags.writeable = True
            mp_drawing.draw_landmarks(
                img,
                self.results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            if self.results.pose_world_landmarks:
                head_pts = []
                for idx, pt in enumerate(self.results.pose_landmarks.landmark):
                    center = np.array(
                        np.multiply([pt.x, pt.y], [self.w, self.h]).astype(int)
                    )
                    if idx in [7,8]:
                        head_pts.append(center)
                        cv2.circle(img, center, 2, (255,0,255), 2, cv2.LINE_AA)
                        message = f"{idx}"
                        cv2.putText(img, message, (center[0], center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1, cv2.LINE_AA)
                # maybe use real 3D positions later. All points relative to center of hip joint
                body_pts = [
                    {
                        'x': int(p.x * self.w),
                        'y': int(p.y * self.h),
                        'z': p.z,
                        'visibility':p.visibility
                    } for p in self.results.pose_world_landmarks.landmark]     
        return img, head_pts

    def visualize(self, img):
        '''
        function that visualized face mesh returned by iris detections.
        '''
        cv2.polylines(img, [self.face.mesh[self.LEFT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
        cv2.polylines(img, [self.face.mesh[self.RIGHT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
        center_left = np.array(self.face.l_iris['center'], dtype=np.int32)
        center_right = np.array(self.face.r_iris['center'], dtype=np.int32)
        cv2.circle(img, center_left, int(self.face.l_iris['radius']), (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(img, center_right, int(self.face.r_iris['radius']), (255,0,255), 2, cv2.LINE_AA)
        cv2.line(img, self.face.mesh[self.HEAD[0]], self.face.mesh[self.HEAD[1]], (0,255,0), 1, cv2.LINE_AA)
        cv2.line(img, self.face.mesh[self.HEAD[2]], self.face.mesh[self.HEAD[3]], (0,255,0), 1, cv2.LINE_AA)
        # credit card points, take these out later
        cv2.circle(img, (505,504), 1, (255,0,255), 2, cv2.LINE_AA)
        cv2.circle(img, (675,501), 1, (255,0,255), 2, cv2.LINE_AA)
        # iris output
        self.frame = img


class VidStream(object):
    '''
    a wrapper for OpenCV that accepts an estimater and a detector.
    '''
    def __init__(self, estimator, detector, face, src=None, output=None):
        self.detector = detector
        self.estimator = estimator
        self.face = face
        self.video = cv2.VideoCapture(src)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # FPS = 1/X, X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        self.w = int(self.video.get(3))
        self.h = int(self.video.get(4))
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter(str(output), self.fourcc, 20, (self.w, self.h))
        self.status = False
        self.i_frame = None
        self.b_frame = None
        self.cnt = 0

    def stream(self):
        while True:
            if self.video.isOpened() == False:
                print('Error opening file.')
            
            if self.video.isOpened():
                self.status, self.frame = self.video.read()
                self.cnt += 1
                print(f'Frame: {self.cnt}')
                if self.cnt >= 230:
                    self.video.release()
                    self.writer.release()
                    cv2.destroyAllWindows()
                    break
                if self.status == True:
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        self.video.release()
                        self.writer.release()
                        break
                    self.face.mesh = None
                    self.detector.findIris(self.frame)
                    if not self.face.mesh is None:
                        self.detector.visualize(self.frame)
                        # calculate distances
                        l_diameter = self.face.l_iris['radius'] * 2
                        # s2c distance
                        self.face.s2c_dist(self.face.w_iris, l_diameter)
                        # head points
                        x1, y1 = self.face.mesh[self.detector.HEAD[0]]
                        x2, y2 = self.face.mesh[self.detector.HEAD[1]]
                        # head width in pixels
                        self.face.get_headw((x1, y1), (x2, y2))
                        message = f"S2C Distance (ft) - iris: {str(self.face.s2c_d)}"
                        # message2 = f"S2C Distance (ft) - head: {str(s2c_d2)}"
                        message3 = f'Head width (in): {str(round((self.face.head_w/10) / 2.54, 2))}'
                        message4 = f'head_w_mm: {str(self.face.head_w)}'
                        message5 = f'focal length: {round(self.face.f, 2)}'
                        # message6 = f'mm / pixel - iris plane: {pix_dist}'
                        messages = [message, message3, message4, message5]
                        for idx, m in enumerate(messages):
                            cv2.putText(self.frame, m, (50, 50 + idx*50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                        self.writer.write(self.frame)
                    else:
                        message = 'Face not detected. Using body pose estimates.'
                        cv2.putText(self.frame, message, (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                        self.frame, head_pts = self.detector.findBody(self.frame)
                        self.face.get_headw(head_pts[0], head_pts[1])
                        self.face.s2c_dist(median(self.face.head_measurements), self.face.head_w)
                        message2 = f'S2C dist (ft): {self.face.s2c_d}'
                        cv2.putText(self.frame, message2, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                        self.writer.write(self.frame)
                    # self.write_output()
                else:
                    self.video.release()
                    self.writer.release()
                    break
            else:
                print(f'Sucessfully read {self.cnt} out of {self.video.get(7)} frames.')
                self.video.release()
                self.writer.release()
                break
        cv2.destroyAllWindows()
    
    def detect_points(self, img):
        detector.findIris(img)
        if not self.face.mesh:
            detector.findBody(img)

    def write_output(self):
        prediction = self.estimator.predict(self.frame)
        output = prediction.astype(np.uint8)
        three_c = cv2.merge([output,output,output])
        self.writer.write(three_c)

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
        # head width (mm) based on iris diameter (mm)
        self.head_w = 0
        # holds head measurements
        self.head_measurements = []
        # subject-to-camera distance
        self.s2c_d = 0

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
        # head width in pixels
        head_pixw = dist((p1[0], p1[1]), (p2[0], p2[1]))
        # horizontal distance in mm/pixel units : iris plane
        self.head_w = (head_pixw * self.w_iris) / (self.l_iris['radius'] * 2)
        self.head_measurements.append(self.head_w)

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

if __name__ == '__main__':
    # load depth estimator
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    vid = "/home/jhoward/facedepth/webcam_video.mp4"
    vid2 = "/home/jhoward/facedepth/card_20_10_5.mp4"
    vid3 = "/home/jhoward/facedepth/10ft.mp4"
    output = '/home/jhoward/facedepth/new_output.avi'
    # raw coordinates for card from test data
    CARD = np.array([505, 504, 675, 501])
    # calculating focal length based on credit card test footage
    # distance to credit card (in)
    d_2_obj = 20
    # assume standard iris diameter of 11.7 mm
    w_real = 11.7
    # face object holds focal length, face data & calculates s2c_dist
    face = FaceDet(d_2_obj, CARD)
    
    
    midas = DepthEstimator(model_type)
    detector = PersonDetector(face)
    video_stream = VidStream(midas, detector, face, vid3, output)
    video_stream.stream()
