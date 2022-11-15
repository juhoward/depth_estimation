import cv2
import numpy as np


class Stereo_VidStream(object):
    def __init__(self, camera_ids:dict, record=False):
        self.camera_ids = camera_ids
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

    def stereo_stream(self):
        print('press "q" to exit...')
        while True:
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                if ok: 
                    cv2.imshow(name, frame)
                    if self.record:
                        self.writers[name].write(frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                cam.release()
                if self.record:
                    self.writers[name].release()
                break
        for name in self.cameras:
            cv2.destroyWindow(name)
    
    def rectify(self, method='ORB', f=1194.5, b=3.75*2.54):
        print('press any key to continue...')
        if method == 'SIFT':
            detector = cv2.SIFT_create()
            FLANN_INDEX_KDTREE = 1
            index_params = {'algorithm':FLANN_INDEX_KDTREE, 'trees':5}
        else:
            detector = cv2.ORB_create()
            FLANN_INDEX_LSH = 6
            index_params = {'algorithm':FLANN_INDEX_LSH, 
                            'table_number': 6, #12,
                            'key_size': 12, #20,
                            'multi_probe_level': 1} #2
        search_params ={'checks':50}
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        descriptors = {name:None for name, id in self.camera_ids.items()}
        frames = {name:None for name, id in self.camera_ids.items()}
        gray = {name:None for name, id in self.camera_ids.items()}
        cnt = 0
        while True:
            for name, cam in self.cameras.items():
                ok, frame = cam.read()
                frames[name] = frame
                if ok:
                    cnt += 1
                    # cv2.imshow(name, frame)
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray[name] = img
                    # returns tuple (keypoints, descriptors)
                    features = detector.detectAndCompute(img, None)
                    descriptors[name] = features

            # keypts = cv2.drawKeypoints(
            #     frame, features[0], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            # )
            # cv2.imshow("Keypoints", keypts)
            # cv2.waitKey(0)
            if cnt % 2 == 0:
                names = list(self.camera_ids.keys())
                # match keypoint descriptors from each camera
                matches = flann.knnMatch(descriptors[names[0]][1], descriptors[names[1]][1], k=2)
                # Need to draw only good matches, so create a mask
                matchesMask = [[0,0] for i in range(len(matches))]
                # hold on to points for later
                pts1 = []
                pts2 = []
                try:
                    # ratio test as per Lowe's paper
                    for i,(m,n) in enumerate(matches):
                        if m.distance < n.distance: # 0.7, 0.8
                            matchesMask[i]=[1,0]
                            # keypoints for camera 2
                            pts2.append(descriptors[names[1]][0][m.trainIdx].pt)
                            # indices for camera 1
                            pts1.append(descriptors[names[0]][0][m.queryIdx].pt)
                    # draw_params = { 'matchColor':(0,255,0),
                    #                 'singlePointColor' : (255,0,0),
                    #                 'matchesMask' : matchesMask,
                    #                 'flags':cv2.DrawMatchesFlags_DEFAULT}
                    # img_matches = cv2.drawMatchesKnn(frames[names[0]],
                    #                             descriptors[names[0]][0],
                    #                             frames[names[1]],
                    #                             descriptors[names[1]][0],
                    #                             matches,None,**draw_params)
                    # cv2.imshow('Keypoint Matches', img_matches)
                except ValueError:
                    print('Value error: not enough value to unpack from matches, line 89.')
                    continue

                # ------------------------------------------------------------
                # STEREO RECTIFICATION

                # Calculate the fundamental matrix for the cameras
                # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
                pts1 = np.int32(pts1)
                pts2 = np.int32(pts2)
                fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
                print(f'Fundamental Matrix:\n{fundamental_matrix}')
                # We select only inlier points
                pts1 = pts1[inliers.ravel() == 1]
                pts2 = pts2[inliers.ravel() == 1]
                # Visualize epilines
                # Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
                def drawlines(img1src, img2src, lines, pts1src, pts2src, color=False):
                    ''' img1 - image on which we draw the epilines for the points in img2
                        lines - corresponding epilines '''
                    # use the same random seed so that two images are comparable!
                    np.random.seed(0)
                    # get row, col range
                    r, c = img1src.shape[:2]
                    if color:
                        img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
                        img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)

                    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        x0, y0 = map(int, [0, -r[2]/r[1]])
                        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
                        img1color = cv2.line(img1src, (x0, y0), (x1, y1), color, 1)
                        img1color = cv2.circle(img1src, tuple(pt1), 5, color, -1)
                        img2color = cv2.circle(img2src, tuple(pt2), 5, color, -1)
                    return img1src, img2src


                # Find epilines corresponding to points in right image (second image) and
                # drawing its lines on left image
                lines1 = cv2.computeCorrespondEpilines(
                    pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
                lines1 = lines1.reshape(-1, 3)
                img5, img6 = drawlines(frames[names[0]], frames[names[1]], lines1, pts1, pts2)

                # Find epilines corresponding to points in left image (first image) and
                # drawing its lines on right image
                lines2 = cv2.computeCorrespondEpilines(
                    pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
                lines2 = lines2.reshape(-1, 3)
                img3, img4 = drawlines(frames[names[1]], frames[names[0]], lines2, pts2, pts1)
                epilines = np.hstack((img5, img3))
                # cv2.imshow("Epilines in both images", epilines)
                cv2.imshow("epilines, img5, img3", epilines)
                # Stereo rectification (uncalibrated variant)

                # Adapted from: https://stackoverflow.com/a/62607343
                h, w = gray[names[0]].shape[:2]
                # H1, H2 are homogrpahy matrices
                _, H1, H2 = cv2.stereoRectifyUncalibrated(
                    np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w, h)
                )
                # Undistort (rectify) the images and save them
                # Adapted from: https://stackoverflow.com/a/62607343
                # img1_rectified = cv2.warpPerspective(frames[names[0]], H1, (w, h))
                # img2_rectified = cv2.warpPerspective(frames[names[1]], H2, (w, h))
                # combo = np.hstack((img1_rectified, img2_rectified))
                # cv2.imshow("rectified", combo)
                img1_rectified = cv2.warpPerspective(gray[names[0]], H1, (w, h))
                img2_rectified = cv2.warpPerspective(gray[names[1]], H2, (w, h))
                combo = np.hstack((img1_rectified, img2_rectified))
                cv2.imshow("rectified", combo) 
                # ------------------------------------------------------------
                # CALCULATE DISPARITY (DEPTH MAP)
                # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
                # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

                # StereoSGBM Parameter explanations:
                # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

                # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
                block_size = 7
                min_disp = -128
                max_disp = 128
                # Maximum disparity minus minimum disparity. The value is always greater than zero.
                # In the current implementation, this parameter must be divisible by 16.
                num_disp = max_disp - min_disp
                # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
                # Normally, a value within the 5-15 range is good enough
                uniquenessRatio = 5
                # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
                # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
                speckleWindowSize = 200
                # Maximum disparity variation within each connected component.
                # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
                # Normally, 1 or 2 is good enough.
                speckleRange = 1
                disp12MaxDiff = 0

                stereo = cv2.StereoSGBM_create(
                    minDisparity=min_disp,
                    numDisparities=num_disp,
                    blockSize=block_size,
                    uniquenessRatio=uniquenessRatio,
                    speckleWindowSize=speckleWindowSize,
                    speckleRange=speckleRange,
                    disp12MaxDiff=disp12MaxDiff,
                    P1=8 * block_size**2,
                    P2=32 * block_size**2,
                )
                disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)
                print(f'Min disparity: {np.min(disparity_SGBM)}\nMax disparity: {np.max(disparity_SGBM)}')
                # min max normalization
                disparity_norm = (disparity_SGBM - np.min(disparity_SGBM)) / (np.max(disparity_SGBM) - np.min(disparity_SGBM))
                disp_range = np.max(disparity_norm) - np.min(disparity_SGBM)
                disp_shift = disparity_SGBM + disp_range
                print(f'Shifted\nMin disparity: {np.min(disp_shift)}\nMax disparity: {np.max(disp_shift)}\n')
                print(f'Disparity range: {disp_range}')
                distance_map = (b * f) / disp_shift
                distance_map = np.uint8(np.clip(distance_map, 0, 255))
                # distance_map = distance_map / disparity_norm
                # cv2.imshow('Raw Disparity', disparity_SGBM)
                # Normalize the values to a range from 0..255 for a grayscale image
                disp_img = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                            beta=0, norm_type=cv2.NORM_MINMAX)
                disp_img = np.uint8(disp_img)
                disp_img = cv2.applyColorMap(disp_img, cv2.COLORMAP_MAGMA)
                cv2.imshow("Disparity Map", disp_img)
                cv2.imshow("Distance Map", distance_map)
                print(f'Min distance: {np.min(distance_map)}\nMax distance: {np.max(distance_map)}')
                print(f'Median distance: {np.median(distance_map)}') 

        for name in self.cameras:
            cv2.destroyWindow(name)


if __name__ == "__main__":
    cameras = {"Camera 1": 0, "Camera 2": 2}
    streamer = Stereo_VidStream(cameras)
    # streamer.stereo_stream()
    streamer.rectify(method='SIFT')