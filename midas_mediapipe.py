import cv2
import numpy as np
from statistics import median
from time import process_time
from face import FaceDet
from depth_midas import DepthEstimator
from detectors import PersonDetector



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
        self.writer = cv2.VideoWriter(str(output), self.fourcc, 20, (self.w, self.h // 2))
        self.status = False
        self.cnt = 0
        self.performance = {'iris': list(), 'body': list(), 'depth': list()}

    def stream(self):
        while True:
            if self.video.isOpened() == False:
                print('Error opening file.')
            
            if self.video.isOpened():
                self.status, self.frame = self.video.read()
                self.cnt += 1
                print(f'Frame: {self.cnt}')
                # if self.cnt >= 230:
                #     self.video.release()
                #     self.writer.release()
                #     cv2.destroyAllWindows()
                #     break
                if self.status == True:
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        self.video.release()
                        self.writer.release()
                        break
                    self.face.mesh = None
                    self.start = process_time()
                    self.detector.findIris(self.frame)
                    self.end = process_time()
                    self.performance['iris'].append(self.end - self.start)

                    self.start = process_time()                    
                    depth_frame = self.estimator.predict(self.frame)
                    self.end = process_time() 
                    self.performance['depth'].append(self.end - self.start)
                    # depth from iris
                    self.face.get_depth(depth_frame)
                    self.face.rel2abs()                
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
                        ######################################
                        # write output to rgb frame
                        message = f"S2C Distance (ft) - iris: {str(self.face.s2c_d)}"
                        # message2 = f"S2C Distance (ft) - head: {str(s2c_d2)}"
                        message3 = f'Head width (in): {str(round((self.face.head_w/10) / 2.54, 2))}'
                        message4 = f'head_w_mm: {str(self.face.head_w)}'
                        message5 = f'focal length: {round(self.face.f, 2)}'
                        # message6 = f'mm / pixel - iris plane: {pix_dist}'
                        messages = [message, message3, message4, message5]
                        self.write_messages(messages, self.frame)
                        ######################################
                        depth_frame = self.to_video_frame(depth_frame)
                        # write output to depth image
                        message = f'Relative Inverse Depth: {round(self.face.ri_depth, 2)}'
                        message2 = f'Absolute Depth: {round(self.face.abs_depth, 2)}'
                        message3 = f'RMSE: {round(self.face.rmse(), 2)}'
                        message4 = f'MAE: {round(self.face.mae(), 2)}'
                        messages = [message, message2, message3,  message4]
                        self.write_messages(messages, depth_frame)
                        self.write_output(depth_frame)
                    else:
                        # update depth from head location
                        self.face.rel2abs()
                        # write output to rgb frame
                        message = 'Face not detected. Using body pose estimates.'
                        cv2.putText(self.frame, message, (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                        self.start = process_time()
                        self.frame, head_pts = self.detector.findBody(self.frame)
                        self.end = process_time() 
                        self.performance['body'].append(self.end - self.start)
                        self.face.get_headw(head_pts[0], head_pts[1])
                        self.face.s2c_dist(median(self.face.head_measurements), self.face.head_w)
                        message2 = f'S2C dist (ft): {self.face.s2c_d}'
                        cv2.putText(self.frame, message2, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                        depth_frame = self.to_video_frame(depth_frame)
                        # write output to depth image
                        message = f'Relative Inverse Depth: {round(self.face.ri_depth, 2)}'
                        message2 = f'Absolute Depth: {round(self.face.abs_depth, 2)}'
                        message3 = f'RMSE: {round(self.face.rmse(), 2)}'
                        message4 = f'MAE: {round(self.face.mae(), 2)}'
                        messages = [message, message2, message3,  message4]
                        self.write_messages(messages, depth_frame)
                        self.write_output(depth_frame)
                else:
                    print('Performance stats in FPS:')
                    print(f"Iris: {1//median(self.performance['iris'])}")
                    print(f"Body: {1//median(self.performance['body'])}")
                    print(f"Depth: {1//median(self.performance['depth'])}")
                    break
            else:
                print(f'Sucessfully read {self.cnt} out of {self.video.get(7)} frames.')
                break
        self.video.release()
        self.writer.release()
        cv2.destroyAllWindows()

    def write_messages(self, messages, img):
        for idx, m in enumerate(messages):
            cv2.putText(img, m, (50, 50 + idx*50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    def detect_points(self, img):
        detector.findIris(img)
        if not self.face.mesh:
            detector.findBody(img)

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
    # load depth estimator
    model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    vid = "/home/jhoward/facedepth/webcam_video.mp4"
    vid2 = "/home/jhoward/facedepth/card_20_10_5.mp4"
    vid3 = "/home/jhoward/facedepth/10ft.mp4"
    output = '/home/jhoward/facedepth/midas_output.avi'
    # raw coordinates for card from test data
    CARD = np.array([505, 504, 675, 501])
    # calculating focal length based on credit card test footage
    # distance to credit card (in)
    d_2_obj = 20
    # assume standard iris diameter of 11.7 mm
    w_real = 11.7
    # face object holds focal length, face data & calculates s2c_dist
    face = FaceDet(d_2_obj, CARD)
    
    
    estimator = DepthEstimator(model_type)
    detector = PersonDetector(face)
    video_stream = VidStream(estimator, detector, face, vid3, output)
    video_stream.stream()
