import cv2
import numpy as np
import glob
from statistics import median
from calibration.calibration import Calibrator
from disparity import disparity_mapper

from face import FaceDet
from depth_midas import DepthEstimator
from detectors import PersonDetector

class Stereo_VidStream(object):
    def __init__(self,
                 camera_ids:dict, 
                 calibrator,
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
        self.calibrator = calibrator
        self.record = record
        self.cameras = {}
        self.writers = {}
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for name, id in camera_ids.items():
            cv2.namedWindow(name)
            self.cameras[name] = cv2.VideoCapture(id)
            if self.record:
                w = int(self.cameras[name].get(3))
                h = int(self.cameras[name].get(4))
                fname = './' + name + '.mp4'
                self.writers[name] = cv2.VideoWriter(fname, self.fourcc, 20, (w, h))
    def stereo_stream(self, disp_mapper, display=False):
        print('press "q" to exit...')
        frames = {name:None for name, id in self.camera_ids.items()}
        cams = list(frames.keys())
        self.cnt = 0
        while True:
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                if ok:
                    frames[name] = frame
                    self.cnt +=1
                    # cv2.imshow(name, frame)
                    if self.record:
                        self.writers[name].write(frame)
            if self.cnt % 2 == 0:
                rectL, rectR = self.calibrator.stereo_rectify(frames[cams[0]], frames[cams[1]], saved_params=True)
                # iris and body detection
                rect_frames = [rectL, rectR]
                labeled_frames = self.detect(rect_frames)
                disparity_SGBM = disp_mapper.compute(rectL,rectR)
                if display == True:
                    self.visualize_disparity(disparity_SGBM)
                # [(x,y) (w,h)]
                roi = [(disparity_SGBM.shape[1] // 2, disparity_SGBM.shape[0] // 2), (200,200)]
                self.get_depth(roi, disparity_SGBM)

            if cv2.waitKey(1) & 0xff == ord('q'):
                cam.release()
                if self.record:
                    self.writers[name].release()
                for name in self.cameras:
                    cv2.destroyWindow(name)
                break

    def visualize_disparity(self, disparity_SGBM):
        disp_img = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                        beta=0, norm_type=cv2.NORM_MINMAX)
        disp_img = np.uint8(disp_img)
        disp_img = cv2.applyColorMap(disp_img, cv2.COLORMAP_MAGMA)
        cv2.imshow("Disparity Map", disp_img)

    def get_depth(self, roi, disparity_SGBM, b=3.75*2.54):
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
        # print(median_disp)
        f = self.calibrator.get_f()
        print(f'SGBM method - f:{f}\tb:{b}\tdistance: {(b*f)/median_disp}')

    def detect(self, frames):
        faces = [self.faceL, self.faceR]
        detectors = [self.personL, self.personR]
        for frm, face, detector in zip(frames, faces, detectors):
            face.mesh = None
            detector.findIris(frm)
            depth_frame = self.depth_estimator.predict(frm)

            face.rel2abs()
            # if a face is detected, base 2c distance on iris diameter
            if not face.mesh is None:
                # depth from iris
                face.get_depth(depth_frame)
                detector.visualize(frm)
                # calculate distances
                l_diameter = face.l_iris['radius'] * 2
                # s2c distance
                face.s2c_dist(face.w_iris, l_diameter)
                # head points
                x1, y1 = face.mesh[detector.HEAD[0]]
                x2, y2 = face.mesh[detector.HEAD[1]]
                # head width in pixels
                face.get_headw((x1, y1), (x2, y2))

                # write output to rgb frame
                message = f"S2C Distance (ft) - iris: {face.s2c_d}"
                # message2 = f"S2C Distance (ft) - head: {str(s2c_d2)}"
                message3 = f'Head width (in): {round((face.head_w/10) / 2.54, 2)}'
                message4 = f'head_w_mm: {round(face.head_w, 2)}'
                message5 = f'focal length: {round(face.f, 2)}'
                message6 = f'Frame: {self.cnt}'
                # message6 = f'mm / pixel - iris plane: {pix_dist}'
                messages = [message, message3, message4, message5, message6]
                self.write_messages(messages, frm)

                ########## depth
                # write output to depth image
                message = f'S2C Distance (ft): {round(face.abs_depth, 2)}'
                message2 = f'Relative Inverse Depth: {round(face.ri_depth, 2)}'
                message3 = f'RMSE: {round(face.rmse(), 2)}'
                message4 = f'MAE: {round(face.mae(), 2)}'
                messages = [message, message2, message3,  message4]
                depth_frame = self.to_video_frame(depth_frame)
                self.write_messages(messages, depth_frame)
            # if no face is detected, use head points from body pose
            # S2C distance is based on median head width relative to iris diameter
            else: 
                frm, head_pts = detector.findBody(frm)
                if head_pts[0] == True:
                    face.head_pts = head_pts[1:]
                    face.get_depth(depth_frame)
                    # update depth from head location
                    face.rel2abs()
                    # don't log head width, won't be accurate w/o iris diameter
                    face.get_headw(head_pts[1], head_pts[2], logging=False)
                    median_head_w = median(face.head_measurements)
                    face.s2c_dist(median_head_w, face.head_pixw)
                    # write output to rgb frame
                    message = 'Face not detected. Using body pose estimates.'
                    cv2.putText(frm, message, (70, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    message2 = f'S2C dist (ft): {face.s2c_d}'
                    message3 = f'focal length: {round(face.f, 2)}'
                    message4 = f'median head w (in): {round(median_head_w / (10 * 2.54), 2)}'
                    message5 = f'Frame: {self.cnt}'
                    # message6 = f'mm / pixel - iris plane: {pix_dist}'
                    messages = [message2, message3, message4, message5]
                    self.write_messages(messages, frm)
                # if no head points detected, neither model found a person
                else:
                    # print(f'No detection. Face mesh: {type(self.face.mesh)}\nHead pts: {head_pts}')
                    message = 'Body not detected.'
                    depth_frame = self.to_video_frame(depth_frame)
                    self.write_messages([message], frm)
            # self.write_output(depth_frame)
        combo = np.hstack((frames[0], frames[1]))
        cv2.imshow('Detections', combo)
        return frames

    def write_messages(self, messages, img):
        for idx, m in enumerate(messages):
            cv2.putText(img, m, (50, 50 + idx*50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # def detect_points(self, img):
    #     detector.findIris(img)
    #     if not self.face.mesh:
    #         detector.findBody(img)

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
        frame1 = cv2.resize(img1, (self.w //2, self.h//2))
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

    # raw coordinates for card from test data
    CARD = np.array([505, 504, 675, 501])
    CARD2 = np.array([584, 257, 676, 257])
    # calculating focal length based on credit card test footage
    # distance to credit card (in)
    d_2_card = 20
    d_2_card2 = 36
    # assume standard iris diameter of 11.7 mm
    w_real = 11.7

    # face object holds focal length, face data & calculates s2c_dist
    faceL = FaceDet(d_2_card2, CARD2)
    faceR = FaceDet(d_2_card2, CARD2)
    # midas
    estimator = DepthEstimator(model_type)
    # stereo calibration
    calibrator = Calibrator(cameras)
    disp = disparity_mapper(calibrated=False)
    if from_saved:
        # run calibration.py first to save camera parameters
        calibrator.get_rectification_params()
        streamer = Stereo_VidStream(cameras, calibrator, faceL, faceR, estimator)
        streamer.stereo_stream(disp)
    else:
        img_dir = '/home/digitalopt/proj/face_depth/stereo_depth/calibration/stereo_imgs/'
        calibrator.get_camera_params(img_dir)
        calibrator.stereo_calibrate()
        calibrator.get_rectification_params()
        streamer = Stereo_VidStream(cameras, calibrator, faceL, faceR, estimator)
        streamer.stereo_stream(disp)
