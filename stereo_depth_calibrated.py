import cv2
import numpy as np
import glob
from calibration.calibration import Calibrator
from disparity import disparity_mapper

class Stereo_VidStream(object):
    def __init__(self, camera_ids:dict, calibrator, record=False):
        self.camera_ids = camera_ids
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
        cnt = 0
        while True:
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                if ok:
                    frames[name] = frame
                    cnt +=1
                    # cv2.imshow(name, frame)
                    if self.record:
                        self.writers[name].write(frame)
            if cnt % 2 == 0:
                rectL, rectR = self.calibrator.stereo_rectify(frames[cams[0]], frames[cams[1]], saved_params=True)
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
        print(f'f:{f}\tb:{b}\tdistance: {(b*f)/median_disp}')




if __name__ == '__main__':
    # configurations
    cameras = {"camera_l": 0, "camera_r": 2}
    output_dir = './stereo_imgs/'
    from_saved=False
    calibrator = Calibrator(cameras)

    disp = disparity_mapper(calibrated=True)
    if from_saved:
        # run calibration.py first to save camera parameters
        streamer = Stereo_VidStream(cameras, calibrator)
        streamer.stereo_stream(disp)
    else:
        img_dir = './calibration/stereo_imgs/'
        calibrator.get_camera_params(img_dir)
        calibrator.stereo_calibrate()
        calibrator.get_rectification_params()
        streamer = Stereo_VidStream(cameras, calibrator)
        streamer.stereo_stream(disp, display=True)
