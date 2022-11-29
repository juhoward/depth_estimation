import cv2
from face import FaceDet
from detectors import PersonDetector
from statistics import median
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
        self.dist = dist
        # distance to the camera
        self.dist = self.dist
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
        while True:
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                if ok:
                    cv2.imshow('calibration', frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
                cam.release()
                cv2.destroyAllWindows()
                break
            elif cv2.waitKey(2) & 0xff == ord('c'):
                self.cnt +=1
                f = self.detect(self.face, self.detector, frame)
                f_lengths.append(f)
                m = f'Captured {self.cnt} out of 40 images.'
                cv2.putText(frame, m, (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('calibration', frame)
                if self.cnt > 39:
                    self.face.f_iris = median(f_lengths)
                    print(f'focal length: {median(f_lengths)}')
                    cam.release()
                    cv2.destroyAllWindows()
                    break
        return self.face.f_iris


    def detect(self, face, detector, frame):
        face.mesh = None        
        detector.findIris(frame)
        if not face.mesh is None:
            # a tuple (left eye, right eye)
            face.xvals = self.xvals(face, detector)
            l_diameter = face.l_iris['radius'] * 2
            r_diameter = face.r_iris['radius'] * 2
            diameter = (l_diameter + r_diameter) / 2
            return self.get_f_length(diameter)


    def get_f_length(self, diameter):
        return (self.dist * diameter) / self.face.w_iris
    
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
    CARD2 = np.array([315, 240, 402, 240])
    d_2_card = 20
    face = FaceDet(d_2_card, CARD2)
    monocal = MonoCalibrator(cameras, face, 12*25.4)
    f_length_iris = monocal.stream()