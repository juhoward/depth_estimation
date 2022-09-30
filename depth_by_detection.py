"""
Face & Body Detection Module
"""

import cv2
import mediapipe as mp
import numpy as np
from statistics import median
from math import sqrt


class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, minDetectionCon=0.2):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.mpface_mesh = mp.solutions.face_mesh
        self.results = None
        self.w = None
        self.h = None

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            ih, iw, ic = img.shape
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                bboxInfo = {"id": id, "bbox": bbox, "score": detection.score, "center": (cx, cy)}
                bboxs.append(bboxInfo)
                if draw:
                    img = cv2.rectangle(img, bbox, (255, 0, 255), 2)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, bboxs
    
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
                h, w = img.shape[:2]
                self.results = face_mesh.process(img)
                if self.results.multi_face_landmarks:
                    mesh_points = np.array(
                        [
                            np.multiply([p.x, p.y], [w, h]).astype(int) for p in self.results.multi_face_landmarks[0].landmark
                        ]
                    )
                    return mesh_points
    def findBody(self, img):
        '''
        Detect body.
        Returns ear to ear distance in pixels.
        '''
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_pose = mp.solutions.pose
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing = mp.solutions.drawing_utils
        self.w, self.h = img.shape[:2]
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
                head_pts = [
                    {
                        'x': p.x,
                        'y': p.y,
                        'z': p.z,
                        'visibility':p.visibility
                    } for p in self.results.pose_world_landmarks.landmark]
                
        return img, head_pts


def mm2cm(dist):
    return dist/10

def cm_to_ft(dist):
    return round(dist/(2.54*12), 2)

def in_to_mm(dist):
    return round(dist * 2.54 * 10, 2)

def diameter(radius):
    return int(radius * 2)

def dist_euclid(pt1:tuple, pt2:tuple):
    return sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def f_length(d_2_obj, w_object, w_pix):
    ''' 
    returns the focal length based on triangle similarity.
    d_2_obj : known distance to the object
    w_object : known width of object in mm
    w_pix : distance in pixels
    '''
    return (d_2_obj * w_pix) / w_object

def s2c_dist(f, w_object, w_pix):
    '''
    returns the subject-to-camera distance in mm using triangle similarity.
    f : known focal length in mm
    w_object : known width of object in mm
    w_pix : distance in pixels
    '''
    return (f * w_object) / w_pix


def main():
    vid = "/home/jhoward/facedepth/webcam_video.mp4"
    vid2 = "/home/jhoward/facedepth/10ft.mp4"
    vid3 = "/home/jhoward/facedepth/card_20_10_5.mp4"
    output = '/home/jhoward/facedepth/output.avi'
    video = cv2.VideoCapture(vid2) #cv2.VideoCapture(0)
    print(f'Frame count: {video.get(cv2.CAP_PROP_FRAME_COUNT)}')
    if video.isOpened() == False:
        print('Error opening file')
    w = int(video.get(3))
    h = int(video.get(4))
    writer = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'MJPG'), 20, (w,h))

    detector = FaceDetector()
    # face mesh indices
    LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    # horizontal points (left, right), vertical points (top, bottom)
    HEAD = [234, 454, 10, 152]
    # body pose points
    BODY_HEAD = [7, 8]
    # raw coordinates for card from test data
    CARD = [505, 504, 675, 501]

    cnt = 0

    # calculating focal length based on credit card test footage
    w_pix = dist_euclid((CARD[0], CARD[2]), (CARD[1], CARD[3]))
    d_2_obj = in_to_mm(20)
    f = f_length(d_2_obj, w_object=82.6, w_pix=w_pix)
    # assume standard iris diameter of 11.7 mm
    w_real = 11.7
    head_measurements = []
    while video.isOpened():
        cnt += 1
        print(f'Frame: {cnt} Width: {video.get(3)}\t Height: {video.get(4)}')
        success, img = video.read()
        if success:
            mesh_points = detector.findIris(img)
            if mesh_points is not None:
                cv2.polylines(img, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
                cv2.polylines(img, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv2.LINE_AA)
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv2.circle(img, center_left, int(l_radius), (255,0,255), 2, cv2.LINE_AA)
                cv2.circle(img, center_right, int(r_radius), (255,0,255), 2, cv2.LINE_AA)
                cv2.line(img, mesh_points[HEAD[0]], mesh_points[HEAD[1]], (0,255,0), 1, cv2.LINE_AA)
                cv2.line(img, mesh_points[HEAD[2]], mesh_points[HEAD[3]], (0,255,0), 1, cv2.LINE_AA)
                cv2.circle(img, (505,504), 1, (255,0,255), 2, cv2.LINE_AA)
                cv2.circle(img, (675,501), 1, (255,0,255), 2, cv2.LINE_AA)

                # subject-to-camera distance using iris
                l_diameter = l_radius * 2
                # subject-to-camera distance in cm
                s2c_d = s2c_dist(f, w_real, l_diameter)
                # convert to cm
                s2c_d /= 10
                # subject-to-camera distance in ft
                s2c_d = cm_to_ft(s2c_d)

                # head subject-to-camera distance using iris diameter
                x1, y1 = mesh_points[HEAD[0]]
                x2, y2 = mesh_points[HEAD[1]]
                # head width in pixels
                head_pixw = dist_euclid((x1, y1), (x2, y2))
                # horizontal distance in mm/pixel units : iris plane
                pix_dist = w_real / l_diameter
                head_w_mm = (head_pixw * w_real) / l_diameter
                head_measurements.append(head_w_mm)
                # s2c_d2 = ((f*head_w_mm) / head_pixw) / 10
                s2c_d2 = s2c_dist(f, head_w_mm, head_pixw) / 10
                # convert to ft
                s2c_d2 = cm_to_ft(s2c_d2)

                message = f"S2C Distance (ft) - iris: {str(s2c_d)}"
                message2 = f"S2C Distance (ft) - head: {str(s2c_d2)}"
                message3 = f'Head width (in): {str(round((head_w_mm/10) / 2.54, 2))}'
                message4 = f'head_w_mm: {str(head_w_mm)}'
                message5 = f'focal length: {round(f, 2)}'
                message6 = f'mm / pixel - iris plane: {pix_dist}'
                messages = [message, message2, message3, message4, message5, message6]
                for idx, m in enumerate(messages):
                    cv2.putText(img, m, (50, 50 + idx*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                writer.write(img)
                # if cnt > 149:
                #     break
            # if no face mesh, try face detection
            else:
                message = 'Landmarks not detected. Using face boundaries.'
                cv2.putText(img, message, (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                
                # transition to face detector
                img, bboxes = detector.findFaces(img)
                # getting median head measurement
                head_w_mm = median(head_measurements)
                if bboxes:
                    for box in bboxes:
                        w_pix = box["bbox"][2]
                        print(box['bbox'])
                        s2c_d = s2c_dist(f, w_object=head_w_mm, w_pix=w_pix)
                        s2c_d /= 10
                        s2c_d = cm_to_ft(s2c_d)
                        message = f'Frame: {cnt}'
                        message2 = f'S2C dist (ft): {s2c_d}'
                        message3 = f'head w in pixels: {w_pix}'
                        message4 = f'actual head w (mm): {head_w_mm}'
                        messages = [message, message2, message3, message4]
                        for idx, m in enumerate(messages):
                            cv2.putText(img, m, (50, 100 + idx*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                    writer.write(img)
                # if no face bboxes, try body pose estimator
                else:
                    message = 'Face not detected. Using body pose estimates.'
                    cv2.putText(img, message, (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    img, head_pts = detector.findBody(img)
                    # head_pts = head_pts[[7,8]]
                    # print(head_pts)
                    writer.write(img)

                # cv2.imshow('Output', img)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break             
        else:
            print('No access to video feed. Exiting...')
            break
    video.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
