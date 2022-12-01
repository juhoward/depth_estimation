import cv2
from face import FaceDet
from detectors import PersonDetector
from statistics import median
from math import dist
import numpy as np
import os



class MonoCalibrator(object):
    def __init__(self,
                 camera_ids:dict,
                 face,
                 dist):
        os.system("bash ./calibration/disable_autofocus.sh")
        self.camera_ids = camera_ids
        self.face = face
        # distance to the camera
        self.dist = dist
        # mediapipe models
        self.detector= PersonDetector(self.face)
        self.cameras = {}
        for idx, (name, id) in enumerate(camera_ids.items()):
            if idx == 0:
                self.cameras[name] = cv2.VideoCapture(id)


    def stream(self):
        print('press and hold "c" when 12 in. from camera ...')
        self.cnt = 0
        f_lengths = []
        w_irises = []
        w_irises2 = []
        while True:
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                if ok:
                    # for finding card points at a new distance by hand
                    # cv2.circle(frame, (257,240), 1, (255,0,255), 2, cv2.LINE_AA)
                    # cv2.circle(frame, (402,240), 1, (255,0,255), 2, cv2.LINE_AA)
                    cv2.imshow('calibration', frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                break
            elif cv2.waitKey(2) & 0xff == ord('c'):

                output = self.detect(self.face, self.detector, frame)
                if output[0] == True:
                    self.cnt +=1
                    f, w_iris, w_iris2 = output[1:]
                    f_lengths.append(f)
                    w_irises.append(w_iris)
                    w_irises2.append(w_iris2)
                    m = f'Captured {self.cnt} out of 40 images.'
                    m2 = f'Iris diameter - {round(w_iris, 2)}'
                    m3 = f'Iris diameter 2 - {round(w_iris2, 2)}'
                    cv2.putText(frame, m, (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, m2, (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, m3, (50,150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    # self.detector.visualize(frame)
                    frame = self.show_iris_kpts(frame)
                    cv2.imshow('capture', frame)
                if self.cnt > 39:
                    # update iris-based focal length 
                    self.face.f_iris = median(f_lengths)
                    # update width of the iris
                    self.face.w_iris = median(w_irises)
                    print(f'focal length: {median(f_lengths)}')
                    print(f'iris diameter: {median(w_irises)}')
                    print(f'iris diameter from kpts: {median(w_irises2)}')
                    cam.release()
                    cv2.destroyAllWindows()
                    break
        return self.face


    def detect(self, face, detector, frame):
        face.mesh = None        
        detector.findIris(frame)
        output = []
        if not face.mesh is None:
            output.append(True)
            # a tuple (left eye, right eye)
            face.xvals = self.xvals(face, detector)
            diameter = self.get_iris_diameter()
            # returns median pixel width of corneas
            diameter2 = self.get_iris_diameter_from_pts()
            # transforms to mm
            diameter2 = self.get_w_iris2(diameter2)
            self.update_iris_width()
            output.append(self.get_f_length(diameter))
            output.append(self.face.w_iris)
            output.append(diameter2)
            return output
        else:
            output.append(False)
            return output

    def show_iris_kpts(self, img):
        left_i = self.face.mesh[self.detector.LEFT_IRIS]
        right_i = self.face.mesh[self.detector.RIGHT_IRIS]
        for idx, pt in enumerate(left_i):
            cv2.circle(img, pt, 1, (255,0,255), 1, cv2.LINE_AA)
            cv2.putText(img, str(self.detector.LEFT_IRIS[idx]), (pt[0],pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
        for idx, pt in enumerate(right_i):
            cv2.circle(img, pt, 1, (255,0,255), 1, cv2.LINE_AA)
            cv2.putText(img, str(self.detector.RIGHT_IRIS[idx]), (pt[0],pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
        return img

    def get_f_length(self, diameter):
        return (self.dist * diameter) / self.face.w_iris

    def get_iris_diameter(self):
        '''
        returns the median iris diameter (pixels) using the 8 iris keypoints
        in the face mesh.
        '''
        # 4 iris points per eye
        kpts = [self.face.mesh[self.detector.LEFT_IRIS],
                self.face.mesh[self.detector.RIGHT_IRIS]]
        measurements = []
        for pts in kpts:
            # 2 euclidean distances per eye
            diameter1 = dist(pts[0], pts[2])
            diameter2 = dist(pts[1], pts[3])
            measurements.append(diameter1)
            measurements.append(diameter2)
        # returns median of the 4 diameters
        return median(measurements)

    def get_w_iris2(self, diameter):
        return (diameter * self.face.w_card) / self.face.w_pix

    def get_w_iris(self):
        '''
        uses width of the credit card to estimate the corneal diameter
        of detected irises.
        face must be self.dist units away from the camera for assumptions to hold.
        '''
        # return pixel width of iris
        dmtr = self.get_iris_diameter()
        # return real iris diameter
        return (dmtr * self.face.w_card) / self.face.w_pix
    
    def update_iris_width(self):
        self.face.w_iris = self.get_w_iris()
    
    def xvals(self, face, detector, iris=True):
        '''
        collect x values from keypoint detections. x values are used for disparity
        '''
        if iris:
            xvals_left_i = list(map(lambda x: x[0], face.mesh[detector.LEFT_IRIS]))
            xvals_right_i = list(map(lambda x: x[0], face.mesh[detector.RIGHT_IRIS]))
            # print(f'Eye points count: {len(xvals_left_i)+len(xvals_right_i)}')
            return xvals_right_i, xvals_left_i
        else:
            # print(f'Head points count: {len(face.head_pts)}')
            xvals = list(map(lambda x: x[0], face.head_pts))
            return xvals

if __name__ == '__main__':
    cameras = {'camL':0}
    # card points at 20 inches
    CARD20 = np.array([315, 240, 402, 240])
    # card points at 12 inches
    CARD12 = np.array([257, 240, 402, 240])
    # card distance (in.)
    d_2_card = 12
    # face object to hold biometrics
    face = FaceDet(d_2_card, CARD12)
    # calibrator, where distance to camera (in.) is transformed (mm)
    monocal = MonoCalibrator(cameras, face, 12*25.4)
    # updated face object with focal length and iris width (mm)
    face = monocal.stream()