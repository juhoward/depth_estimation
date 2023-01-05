import cv2
import numpy as np
import glob
from statistics import median
from math import dist
from calibration.calibration import Calibrator
from disparity import disparity_mapper

from face import FaceDet
from depth_midas import DepthEstimator
from detectors import PersonDetector
from mono_calibrate import MonoCalibrator
from utils import Results

from time import process_time

class Stereo_VidStream(object):
    '''
    TODO
    include a means to better synchronize the firing of both cameras
    currently, the time between camera fires is not measured,
    it is assumed that synchronization isleading to greater disparities and inaccurate gt distances.
    measure time between camera fires and compare to new test code.
    test this:
        vidStreamL = cv2.VideoCapture(0)
        vidStreamR = cv2.VideoCapture(2)

        for i in range(10):
            vidStreamL.grab()
            vidStreamR.grab()
        _, imgL = vidStreamL.retrieve()
        _, imgR = vidStreamR.retrieve()


    '''
    def __init__(self,
                 camera_ids:dict, 
                 stereo_calibrator,
                 f_lengths,
                 methods,
                 faceL,
                 faceR,
                 depth_estimator,
                 record=False):
        self.camera_ids = camera_ids
        self.faceL = faceL
        self.faceR = faceR
        # mediapipe models
        self.personL= PersonDetector(self.faceL)
        self.personR= PersonDetector(self.faceR)
        self.depth_estimator = depth_estimator
        self.stereo_calibrator = stereo_calibrator
        self.f_lengths = f_lengths
        self.methods = methods
        self.resultsL = Results()
        self.resultsR = Results()
        self.gt = Results()
        self.record = record
        self.cameras = {}
        self.writers = {}
        self.time_logs = {}
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for name, id in camera_ids.items():
            self.cameras[name] = cv2.VideoCapture(id)
            self.time_logs[name] = 0
            w = int(self.cameras[name].get(3))
            h = int(self.cameras[name].get(4))
            for detector in [self.personL, self.personR]:
                detector.w = w
                detector.h = h
            if self.record:
                fname = './' + name + '.mp4'
                self.writers[name] = cv2.VideoWriter(fname, self.fourcc, 20, (w, h))

    def stereo_stream(self, disp_mapper, display=False):
        print('press "q" to exit...')
        frames = {name:None for name, id in self.camera_ids.items()}
        cams = list(frames.keys())
        self.cnt = 0
        while True:
            # capture frames from both cameras to maintain better synchronization
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                if ok:
                    frames[name] = frame
                    self.time_logs[name] = process_time()
                    self.cnt +=1
                    if self.record:
                        self.writers[name].write(frame)
            # when two frames are captured, recitfy them, detect objects, can get distances
            if self.cnt % 2 == 0:
                rectL, rectR = self.stereo_calibrator.stereo_rectify(frames[cams[0]], frames[cams[1]], saved_params=True)
                if display == True:
                    disparity_SGBM = disp_mapper.compute(rectL,rectR)
                    self.visualize_disparity(disparity_SGBM)
                # iris and body detection
                rect_frames = [rectL, rectR]
                self.detect(rect_frames)
            # escape sequence
            if cv2.waitKey(1) & 0xff == ord('q'):
                cam.release()
                if self.record:
                    self.writers[name].release()
                print(f'\nTime between camera fires (single sample): {self.time_logs[cams[1]] - self.time_logs[cams[0]]} (s)')
                # record ground truth observations
                self.gt.write_csv('gt', 'gt')
                # record camera observations
                logs = [self.resultsL, self.resultsR]
                for idx, name in enumerate(self.cameras):
                    for method in self.methods:
                        logs[idx].write_csv(name, method)
                break

    def visualize_disparity(self, disparity_SGBM):
        disp_img = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                        beta=0, norm_type=cv2.NORM_MINMAX)
        disp_img = np.uint8(disp_img)
        disp_img = cv2.applyColorMap(disp_img, cv2.COLORMAP_MAGMA)
        cv2.imshow("Disparity Map", disp_img)

    def get_depth(self, roi, disparity_SGBM, b=3.75*2.54, calibrated=True):
        disp_img = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                        beta=0, norm_type=cv2.NORM_MINMAX)
        disp_img = np.uint8(disp_img)
        disp_img = cv2.applyColorMap(disp_img, cv2.COLORMAP_MAGMA)
        x, y = roi[0]
        w, h = roi[1]
        region = disp_img[y:y+h, x:x+w]
        cv2.imshow("ROI", region)
        roi_disparity = disparity_SGBM[y:y+h, x:x+h]
        median_disp = np.median(roi_disparity)
        if calibrated:
            f = self.stereo_calibrator.get_calibrated_f()
        else:
            f = self.calibrator.get_uncalibrated_f()
        print(f'SGBM method - f:{f}\tb:{b}\tdistance: {(b*f)/median_disp}')

    def detect(self, frames):
        faces = [self.faceL, self.faceR]
        detectors = [self.personL, self.personR]
        logs = [self.resultsL, self.resultsR]
        sides = ['left', 'right']

        for frm, face, detector, log, side in zip(frames, faces, detectors, logs, sides):
            # clear face mesh
            face.mesh = None
            # populate face object with keypoints
            detector.findIris(frm)

            # TODO: maybe put depth back in to continue testing it against other methods.
            # depth_frame = self.depth_estimator.predict(frm)
            # TODO: find a linear transformation that 
            # works from short distances to 10 ft.
            # face.rel2abs()

            # if a face is detected, base 2c distance on iris diameter
            if not face.mesh is None:
                # a tuple (left eye, right eye) for disparity calculation
                face.xvals = self.xvals(face)
                # depth from iris
                # face.get_depth(depth_frame)
                detector.visualize(frm)
                # calculate median iris diameter (pixels)
                i_diameter = face.get_iris_diameter()
                # s2c distances (in) for each focal length estimation method
                # iris diameter is assumed to be 11.7 (mm)
                s2c_dists = list(map(lambda f: self.s2c_dist(f, 11.7, i_diameter), f_lengths))
                # convert from (cm) to (in)
                s2c_dists = list(map(lambda x : x / 2.54, s2c_dists))
                # TODO: get a better credit card detector so assumed iris diameter estimation accuracy 
                #       can be estimated
                # s2c distances (in) for each focal length when iris diameter is estimated
                # s2c_dists_i = list(map(lambda f: self.s2c_dist(f, face.w_iris, i_diameter), f_lengths))
                # s2c_dists_i = list(map(lambda x : x / 2.54, s2c_dists_i))
                # head points
                x1, y1 = face.mesh[detector.HEAD[0]]
                x2, y2 = face.mesh[detector.HEAD[1]]
                # head width in pixels
                face.get_headw((x1, y1), (x2, y2))

                # write output to rgb frame
                message = f"S2C Distances(in): "
                message2 = f"- f_monocal: {round(s2c_dists[0],  2)}"
                message3 = f"- f_card: {round(s2c_dists[1], 2)}"
                message4 = f"- f_iris: {round(s2c_dists[2], 2)}"
                message5 = f'Head width (in): {round(face.head_w / 25.4, 2)}'
                # message6 = f"iris width (mm): {face.w_iris}"
                # message6 = f'focal length (credit card): {round(self.monocal.f_card, 2)}'
                # message7 = f'focal length (iris 11.7mm): {round(self.monocal.f_iris, 2)}'
                # message8 = f'focal length calibrated: {round(self.monocal.f_monocal,2)}'
                # message9 = f'stereo focal length: {round(self.stereo_calibrator.get_calibrated_f(),2)}'
                message10 = f'Frame: {self.cnt}'
                
                messages = [message, message2, message3, message4, message5, message10]
                self.write_messages(messages, frm)

                ########## depth
                # write output to depth image
                # message = f'S2C Distance (ft): {round(face.abs_depth, 2)}'
                # message2 = f'Relative Inverse Depth: {round(face.ri_depth, 2)}'
                # message3 = f'RMSE: {round(face.rmse(), 2)}'
                # message4 = f'MAE: {round(face.mae(), 2)}'
                # messages = [message, message2, message3,  message4]
                # depth_frame = self.to_video_frame(depth_frame)
                # self.write_messages(messages, depth_frame)

            # if no face is detected, use head points from body pose
            # S2C distance is based on median head width relative to iris diameter
            else:
                frm, head_pts = detector.findBody(frm, draw=False)
                if head_pts[0] == True:
                    face.head_pts = head_pts[1:]
                    # get x values to calculate disparity
                    face.xvals = self.xvals(face, iris=False)
                    # face.get_depth(depth_frame)
                    # update depth from head location
                    # face.rel2abs()
                    # don't log head width, won't be accurate w/o iris diameter
                    face.get_headw(head_pts[3], head_pts[4], logging=False)
                    cv2.circle(frm, face.head_pts[3].astype(int), 1, (255,255,255), 2, cv2.LINE_AA)
                    cv2.circle(frm, face.head_pts[4].astype(int), 1, (255,255,255), 2, cv2.LINE_AA)
                    median_head_w = median(face.head_measurements)
                    s2c_dists = list(map(lambda f: self.s2c_dist(f, median_head_w, face.head_pixw), f_lengths))
                    # convert from (cm) to (in)
                    s2c_dists = list(map(lambda x : x / 2.54, s2c_dists))
                    # write output to rgb frame
                    message = 'Face not detected. Using body pose estimates.'
                    cv2.putText(frm, message, (70, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    message2 = f"S2C Distances (in):"
                    message3 = f"- f_monocal: {round(s2c_dists[0],  2)}"
                    message4 = f"- f_card: {round(s2c_dists[1], 2)}"
                    message5 = f"- f_iris: {round(s2c_dists[2], 2)}"
                    message6 = f'median head w (in): {round(median_head_w / (10 * 2.54), 2)}'
                    message7 = f'Frame: {self.cnt}'
                    messages = [message2, message3, message4, message5, message6, message7]
                    self.write_messages(messages, frm)
                # if no head points detected, neither model found a person
                else:
                    # print(f'No detection. Face mesh: {type(self.face.mesh)}\nHead pts: {head_pts}')
                    message = 'Body not detected.'
                    # depth_frame = self.to_video_frame(depth_frame)
                    self.write_messages([message], frm)
            # update logs
            # print(f'updating {side} logs')
            try:
                for dist, method in zip(s2c_dists, self.methods):
                    log.update(dist, method)
            except UnboundLocalError:
                continue
            # self.write_output(depth_frame)
        # report ground truth distance based on detected keypoints
        gt_dist = self.get_gt_distance(faces)
        # gt_dist will be nono when point correspondence isn't achieved
        self.gt.update(gt_dist, 'gt')

        if gt_dist is not None:
            message = f'Distance (in): {round(float(gt_dist), 2)}'
            # self.update_logs(s2c_dists, self.methods)
        else:
            message = 'No point correspondence.'
        combo = np.hstack((frames[0], frames[1]))
        text_coords = (500, 400)
        cv2.putText(combo, message, text_coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('Detections', combo)
    
    def update_log(self, dists:list, methods:list):
        '''
        dists : a list of subject to camera distances
        methods : a list of names to associate with distances
        
        adds distance values and calculates errors for each time step.
        TODO
        add relative inverse depth back into log updates.
        '''
        # record ground truth distance and remove it from lists
        print(f'dists length: {len(dists[:-2])}\tmethods length: {len(methods[:-2])}')
        # for dist, method in zip(dists[:-2], methods[:-2]):          
            # log.rmse(log.history['triangle'], self.gt.history['gt'], 'triangle')
            # log.mae(log.history['triangle'], self.gt.history['gt'], 'triangle')
            # log.rmse(log.history['neural_depth'], self.gt.history['gt'], 'neural_depth')
            # log.mae(log.history['neural_depth'], self.gt.history['gt'], 'neural_depth')
            # print(f"gt: {median(self.gt.history['gt'])}\ntriangle:{median(log.history['triangle'])}\nneural depth: {median(log.history['neural_depth'])}")
            # print(f"\nTriangle\nRMSE: {log.results['triangle_rmse']}\nMAE:{log.results['triangle_mae']}")
            # print(f"\nNeural Depth\nRMSE: {log.results['neural_depth_rmse']}\nMAE:{log.results['neural_depth_mae']}")
    
    def get_gt_distance(self, faces, b = 9.25):
        '''
        faces : the face objects from the left and right cameras
        b : distance in (cm) between the centers of both cameras

        uses the x values from face keypoints to calculate distance to camera.
        median disparity among the keypoints is used to get a more robust 
        distance measurement.
        
        the returned distance is in inches.
        '''

        f = self.stereo_calibrator.f_stereo

        left = faces[0]
        right = faces[1]

        # check if both faces contain iris detections
        if type(left.xvals) == type(tuple()):
            if type(right.xvals) == type(tuple()):
                left_iL = left.xvals[0]
                left_iR = left.xvals[1]
                right_iL = right.xvals[0]
                right_iR = right.xvals[1]
                iL = zip(left_iL, right_iL)
                iR = zip(left_iR, right_iR)
                dispL = [abs(l - r)for l,r in iL]
                dispR = [abs(l - r) for l,r in iR]
                medL = median(dispL)
                medR = median(dispR)
                disparity = (medL + medR) / 2
                return ((f * b) / disparity) / 2.54
            else:
                print('No point correspondence.')
        # check if both faces contain body detections
        elif type(left.xvals) == type(list()):
            if type(right.xvals) == type(list()):
                if len(left.xvals) == len(right.xvals):
                    vals = zip(left.xvals, right.xvals)
                    disp = [abs(l - r) for l,r in vals]
                    disparity = median(disp)
                    # print(f'body disparity: {disparity}')
                    return ((f * b) / disparity) / 2.54
                else:
                    print('No Point correspondence.')
            else:
                print('No point correspondence.')

    def s2c_dist(self, f, w_object, w_pix):
        '''
        returns the subject-to-camera distance in mm using triangle similarity.
        f : focal length in pixels
        w_object : known width of object in mm
        w_pix : object width in pixels
        '''
        # subject to camera distaince (mm)
        s2c_d = (f * w_object) / w_pix
        # transform mm to cm
        s2c_d /= 10
        return s2c_d

    def xvals(self, face, iris=True):
        '''
        collect x values from keypoint detections. x values are used for disparity
        '''
        if iris:
            # returns tuple of 4 left and right iris keypoints
            xvals_left_i = list(map(lambda x: x[0], face.mesh[face.LEFT_IRIS]))
            xvals_right_i = list(map(lambda x: x[0], face.mesh[face.RIGHT_IRIS]))
            # print(f'Eye points count: {len(xvals_left_i)+len(xvals_right_i)}')
            return xvals_right_i, xvals_left_i
        else:
            # returns list of 11 head keypoints from body pose model
            # print(f'Head points count: {len(face.head_pts)}')
            xvals = list(map(lambda x: x[0], face.head_pts))
            return xvals

    def write_messages(self, messages, img):
        for idx, m in enumerate(messages):
            cv2.putText(img, m, (50, 50 + idx*50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    def to_video_frame(self, img):
        ''' 
        transforms midas depth frame to a video frame.
        '''
        output = img.astype(np.uint8)
        # change contrast
        output *= 5
        # brightness
        output += 10
        return cv2.merge([output,output,output])

    def side_by_side(self, img1, img2):
        frame1 = cv2.resize(img1, (self.w//2, self.h//2))
        frame2 = cv2.resize(img2, (self.w//2, self.h//2))
        return np.hstack((frame1, frame2))
    
    def write_output(self, depth_frame):
        combo = self.side_by_side(self.frame, depth_frame)
        self.writer.write(combo)

if __name__ == '__main__':
    # configurations
    cameras = {"camera_l": 0, "camera_r": 2}
    calibration_imgs = './calibration/stereo_imgs/'
    # load stereo camera calibration data?
    from_saved=False

    # select a neural depth estimator
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    # card points at 20 inches
    CARD20 = np.array([315, 240, 402, 240])
    # card points at 12 inches
    CARD12 = np.array([257, 240, 402, 240])
    # face object holds face data
    faceL = FaceDet()
    faceR = FaceDet()
    # Monocalibrator calculates several focal lengths using different metrics, 
    # gets camera intrinsics
    # distance to camera (in.) is transformed (mm)
    monocal = MonoCalibrator(cameras, faceL, 12*25.4, CARD12)
    # updated calibration object with 3 focal lengths and iris width (mm)
    face = monocal.stream(from_saved=True)
    print('Monocular Focal Length Estimates (left camera):')
    print(f'f_card: {monocal.f_card}\tf_iris: {monocal.f_iris}\tf_monocal: {monocal.f_monocal}')
    ################################################################################################
    # get rid of iris calculation for now
    monocal.f_iris = 561.64
    # focal length estimation method names for reporting
    methods = ['f_monocal', 'f_card', 'f_iris']
    # focal length estimates from the monocular calibration step
    f_lengths = [monocal.f_monocal, monocal.f_card, monocal.f_iris]
    ################################################################################################
    # midas
    estimator = DepthEstimator(model_type)
    # stereo calibration
    stereo_calibrator = Calibrator(cameras)
    disp = disparity_mapper(calibrated=False)

    if from_saved:
        # run calibration.py first to save camera parameters
        # TODO make this work.
        stereo_calibrator.get_rectification_params(saved_params=True)
        stereo_calibrator.get_calibrated_f()
        streamer = Stereo_VidStream(cameras, stereo_calibrator, f_lengths, methods, faceL, faceR, estimator)
        streamer.stereo_stream(disp, display=True)
    else:
        img_dir = '/home/digitalopt/proj/depth_estimation/calibration/stereo_imgs/'
        stereo_calibrator.get_camera_params(img_dir)
        stereo_calibrator.stereo_calibrate()
        stereo_calibrator.get_rectification_params()
        stereo_calibrator.get_calibrated_f()
        print('Stereo Focal Length Estimate:')
        print(f'f_stereo: {stereo_calibrator.f_stereo}')
        streamer = Stereo_VidStream(cameras, stereo_calibrator, f_lengths, methods, faceL, faceR, estimator)
        streamer.stereo_stream(disp)                
