import cv2
from face import FaceDet
from detectors import PersonDetector
from statistics import median
from math import dist
import numpy as np
import os
import glob


class MonoCalibrator(object):
    '''
    objective of this object is to estimate the camera's focal length.
    The best method right now is the mono_calibrate, based on Zhang's
    checkerboard calibration and implemented in opencv
    The other two methods use the triangle similarity to estimate f
    based on the assumption of a standard sized credit card and that
    the detected iris is of standard width.
    d_2_obj : the distance from which the iris and card are detected
    face : face object to hold mesh data and measure distances
    detector : person detector that called up iris and body pose models
    camera_ids : the name and bus id of cameras
    points : two points representing the width of a card at the d_2_obj distance
    checker_dims : the dimensions of the checkerboard calibration target
    '''
    def __init__(self,
                 camera_ids:dict,
                 face,
                 d_2_obj,
                 points,
                 checker_dims = (5,8)):
        os.system("bash ./calibration/disable_autofocus.sh")
        self.camera_ids = camera_ids
        self.face = face
        # credit card width (mm)
        self.w_card = 85.6
        # distance to the camera
        self.d_2_obj = d_2_obj
        # euclidean distance between pts
        self.card_w_pix = dist(points[:2],  points[2:])
        # focal lengths
        self.f_card = self.get_f_length(self.card_w_pix, self.w_card)
        self.f_iris = 0
        self.f_monocal = 0
        # mediapipe models
        self.detector= PersonDetector(self.face)
        self.cameras = {}
        for idx, (name, id) in enumerate(camera_ids.items()):
            if idx == 0:
                self.cameras[name] = cv2.VideoCapture(id)
        # Vector for 3D points
        self.points3D= []
        # Vector for 2D points
        self.points2D = []
        # counter
        self.cnt = 0
        # checkerboard dimensions
        self.checker_dims = checker_dims
        # save grayscale frame
        self.grayFrame = None


    def stream(self, from_saved=False):      
        f_lengths = []
        w_irises = []
        if from_saved:
            print(f'reading saved calibration images from : ./calibration/mono_imgs/')
            # reused saved claibration images
            img_reader = self.mono_calibration_data_reader('./calibration/mono_imgs/')
            self.find_corners(img_reader, (5,8))
            camera_intrinsics = self.mono_calibrate()
            # get mean of focal lengths in x and y dimensions
            f = (camera_intrinsics[1][0][0] + camera_intrinsics[1][1][1]) / 2
            self.f_monocal = f
            print(f'Calibration complete. focal length: {self.f_monocal}')
            print('press and hold spacebar when 12 in. from camera to calibrate iris...')
            print('press "n" to end calibration...')
        else:
            print('press and hold "c" when target is near camera ...')
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
                # collect new calibration images
                self.get_mono_calibration_data(frame)
                if len(self.points2D) > 9:
                    # reset img count
                    self.cnt = 0
                    # intrinsic matrix is at index 1 of this list 
                    camera_intrinsics = self.mono_calibrate()
                    # get mean of focal lengths in x and y dimensions
                    f = (camera_intrinsics[1][0][0] + camera_intrinsics[1][1][1]) / 2
                    self.f_monocal = f
                    cv2.destroyAllWindows()
                    # cue to user to move on to next procedure
                    print('Estimating focal length based on iris detector')
                    print('press and hold spacebar when 12 in. from camera ...')

            elif cv2.waitKey(3) & 0xff == ord(' '):
                # estimate focal length
                output = self.detect(self.face, self.detector, frame)
                if output[0] == True:
                    self.cnt +=1
                    f, w_iris = output[1:]
                    f_lengths.append(f)
                    w_irises.append(w_iris)
                    m = f'Captured {self.cnt} out of 40 images.'
                    m2 = f'Iris diameter - {round(w_iris, 2)}'
                    cv2.putText(frame, m, (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    cv2.putText(frame, m2, (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    self.detector.visualize(frame)
                    # frame = self.show_iris_kpts(frame)
                    cv2.imshow('iris capture from 12in.', frame)
                if self.cnt > 29:
                    # update iris-based focal length 
                    self.f_iris = median(f_lengths)
                    # update width of the iris
                    self.face.w_iris = median(w_irises)
                    print(f'credit card-based iris diameter: {median(w_irises)}')
                    for name in self.cameras:
                        self.cameras[name].release()
                    cv2.destroyAllWindows()
                    return self.face
            elif cv2.waitKey(4) & 0xff == ord('n'):
                self.f_iris = 561.64
        print('Focal Lengths:')
        print(f'f_card: {self.f_card}\tf_iris: {self.f_iris}\tf_monocal: {self.f_monocal}')


    def detect(self, face, detector, frame):
        face.mesh = None        
        detector.findIris(frame)
        output = []
        if not face.mesh is None:
            output.append(True)
            # a tuple (left eye, right eye)
            face.xvals = self.xvals(face, detector)
            # returns median pixel width of iris
            w_iris_pix = face.get_iris_diameter()
            # transforms to real width and updates face mesh
            face.update_iris_width(self.w_card, self.card_w_pix)
            # focal length when iris diameter assumed to be 11.7mm
            output.append(self.get_f_length(w_iris_pix, 11.7))
            output.append(face.w_iris)
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

    def get_f_length(self, w_pixels, w_mm):
        '''
        returns focal length based on triangle similarity.

        self.d_2_obj = distance to camera (mm)
        w_pixels = pixel width of detected object
        w_mm = real / assumed width of object (mm)
        '''
        return (self.d_2_obj * w_pixels) / w_mm

    def xvals(self, face, iris=True):
        '''
        collect x values from keypoint detections. x values are used for disparity
        '''
        if iris:
            xvals_left_i = list(map(lambda x: x[0], face.mesh[face.LEFT_IRIS]))
            xvals_right_i = list(map(lambda x: x[0], face.mesh[face.RIGHT_IRIS]))
            # print(f'Eye points count: {len(xvals_left_i)+len(xvals_right_i)}')
            return xvals_right_i, xvals_left_i
        else:
            # print(f'Head points count: {len(face.head_pts)}')
            xvals = list(map(lambda x: x[0], face.head_pts))
            return xvals
    
    def mono_calibration_data_reader(self, img_dir):
        '''
        reads previously collected calibration images.
        '''
        imgs = glob.glob('./calibration/mono_imgs/*.png')
        return (cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY) for im in imgs)

    def find_corners(self, img_reader, checker_dims=(5,8)):
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 3D points real world coordinates
        objectp3d = np.zeros((1, 
                            checker_dims[0] * checker_dims[1],
                            3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:checker_dims[0],
                                    0:checker_dims[1]].T.reshape(-1, 2)
        for img in img_reader:
            ret, corners = cv2.findChessboardCorners(
                            img, checker_dims,
                            cv2.CALIB_CB_ADAPTIVE_THRESH
                            + cv2.CALIB_CB_FAST_CHECK +
                            cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                self.grayFrame = img
                self.points3D.append(objectp3d)
                corners2 = cv2.cornerSubPix(
                            self.grayFrame, corners, (11, 11), (-1, -1), criteria)
                self.points2D.append(corners2)
    
    def get_mono_calibration_data(self, frame, checker_dims=(5,8), save=True):
        '''
        1. uses single camera to collect 9 calibration images.
        2. if checkerboard of check_dims dimensions is detected, 2 and 3d points are collected.
        3. Image with the detected corners labeled is displayed.
        '''
        # stop the iteration when specified
        # accuracy, epsilon, is reached or
        # specified number of iterations are completed.
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 3D points real world coordinates
        objectp3d = np.zeros((1, 
                            checker_dims[0] * checker_dims[1],
                            3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:checker_dims[0],
                                    0:checker_dims[1]].T.reshape(-1, 2)
        self.grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
                        self.grayFrame, checker_dims,
                        cv2.CALIB_CB_ADAPTIVE_THRESH
                        + cv2.CALIB_CB_FAST_CHECK +
                        cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            if save == True:
                self.cnt += 1
                fname = './calibration/mono_imgs/' + str(self.cnt) + '.png'
                print(f'Saving image: {fname}')
                cv2.imwrite(fname, frame)
            self.points3D.append(objectp3d)
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                self.grayFrame, corners, (11, 11), (-1, -1), criteria)
            self.points2D.append(corners2)
            # Draw and display the corners
            image = cv2.drawChessboardCorners(frame,
                                            checker_dims,
                                            corners2, ret)
            cv2.imshow('Target Detected', image)

    def mono_calibrate(self):
        ''' 
        Perform camera calibration by
        passing the 3D points and the corresponding 
        pixel coordinates of the detected corners (points2D)
        '''
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
            self.points3D, self.points2D, self.grayFrame.shape[::-1], None, None)
        print(f'Calibration error:\n{ret}')
        print(f"\nCamera matrix: \n{matrix}")
        print(f"\n Distortion coefficients:\b{distortion}")
        print(f"\n Rotation Vectors:\n{r_vecs}")
        print(f"\n Translation Vectors:\n{t_vecs}")
        return ret, matrix, distortion, r_vecs, t_vecs

if __name__ == '__main__':
    cameras = {'camL':0}
    # card points at 20 inches
    CARD20 = np.array([315, 240, 402, 240])
    # card points at 12 inches
    CARD12 = np.array([257, 240, 402, 240])
    # face object to hold biometrics
    face = FaceDet()
    # calibrator, where distance to camera (in.) is transformed (mm)
    monocal = MonoCalibrator(cameras, face, 12*25.4, CARD12)
    # updated face object with focal length and iris width (mm)
    face = monocal.stream(from_saved=True)
    print('Focal Lengths:')
    print(f'f_card: {monocal.f_card}\tf_iris: {monocal.f_iris}\tf_monocal: {monocal.f_monocal}')