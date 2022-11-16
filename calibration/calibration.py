import numpy as np
import cv2
import glob
import os



class Calibrator(object):
    def __init__(self, camera_ids):
        os.system("bash disable_autofocus.sh")
        # represents the inside corners along both edges
        self.chessboardSize = (9,6)
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
        self.flags = 0
        self.camera_ids = camera_ids
        if type(camera_ids) == type(dict()):
            cam_names = list(camera_ids.keys())
            cam_names = [''.join(x.split(' ')).lower() for x in cam_names]
        self.cam_nms = cam_names


    def get_camera_params(self, img_dir, display=False):
        ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################
        print('calibrating...')
        img_maps = {name: sorted(glob.glob(img_dir + name + '/*.png')) for name in list(self.cam_nms)}
        camL = dict()
        camR = dict()
        # two dictionaries to hold camera parameters
        params = [camL, camR]

        # Arrays to store object points and image points from all the images.
        # 2d points in image plane.
        self.imgpoints = {name: list() for name in self.cam_nms}
        # 3d point in real world space
        self.objpoints = {name: list() for name in self.cam_nms}

        for param_dict, cam_name in zip(params, self.cam_nms):
            print(f'\nreading images for: {cam_name}\n\n')
            pathlist = img_maps[cam_name]
            for idx, path in enumerate(pathlist):
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.img_dims = gray.shape[::-1]
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, self.chessboardSize, None)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    # overwrite first corners with subpixel corners
                    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                    # add chessboard point, where z=0
                    self.objpoints[cam_name].append(self.objp)
                    # add its pixel coordinate
                    self.imgpoints[cam_name].append(corners)
                    if display:
                        # Draw and display the corners
                        cv2.drawChessboardCorners(img, self.chessboardSize, corners, ret)
                        cv2.imshow(path, img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows() 
                else:
                    print(f'Error processing image : {path}')
                    continue

            ############## CALIBRATION #######################################################
            width, height = self.img_dims
            # takes whole list of obj and image points from the dataset and calibrates the camera
            rmse, cameraMatrix, distortion, rotation, translation = cv2.calibrateCamera(self.objpoints[cam_name], 
                                                                        self.imgpoints[cam_name], (width, height), None, None)
            print(f'Camera: {cam_name}\nRMSE: {rmse}\n')

            if display == True:
                # refine camera matrix and return a region of interest
                # if alpha = 0, it returns undistorted image without some pixels.
                newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortion, (width, height), 1, (width, height))
                # simple undistortion
                undist = cv2.undistort(img, cameraMatrix, distortion, None, newCameraMatrix)
                x, y, w, h = roi
                undist_c = undist[y:y+h, x:w+w]
                # cv2.imshow('Simple undistortion', undist)

                # using remapping
                mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distortion, None, newCameraMatrix, (width,height), 5)
                undist2 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                # crop the image
                x, y, w, h = roi
                undist2_c = undist2[y:y+h, x:x+w]
                # cv2.imshow('Undistort, Rectify, Remap', undist2_c)
                # for i, j in zip(['img', 'undist', 'undist2'], [img, undist_c, undist2_c]):
                #     print(i, j.shape)
                merged = np.hstack([img, undist, undist2])
                merged = cv2.resize(merged, (1920//2, 1080//2))
                cv2.imshow('Original Img, Simple Undistortion, Remapped', merged)
            
            param_dict['intrinsic'] = cameraMatrix
            param_dict['distortion'] = distortion
            param_dict['RMSE'] = [rmse]

        cam_params = {name:p for name, p in zip(self.cam_nms, params)}
        for k in cam_params.keys():
            print(f'\n\nCamera parameters identified for {k}\n')
            print(f"Intrinsic matrix\n {cam_params[k]['intrinsic']}")
            print(f"Distortion coefficients\n {cam_params[k]['distortion']}")
            print(f"RMSE: {cam_params[k]['RMSE']}\n\n")
        self.cam_params = cam_params

    def capture_stereo_images(self, output_dir='./stereo_imgs/', record_vid=False):
        # create capture and writer objects
        self.cameras = {}
        for name, id in self.camera_ids.items():
            cv2.namedWindow(name)
            self.cameras[name] = cv2.VideoCapture(id)
            if record_vid == True:
                self.writers = {}
                self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                w = int(self.cameras[name].get(3))
                h = int(self.cameras[name].get(4))
                fname = './' + name + '.mp4'
                self.writers[name] = cv2.VideoWriter(fname, self.fourcc, 20, (w, h))
        for cam in self.camera_ids.keys():
            print(f'making directory- {output_dir + cam}')
            os.makedirs(output_dir + cam, exist_ok=True)
        # original color frames
        rgb_frms = {name:None for name in self.cameras.keys()}
        cnt = 0
        while True:
            for name, cam in self.cameras.items():
                status, rgb_frms[name] = cam.read()
                if status == True:
                    cv2.imshow(name, rgb_frms[name])
            if cv2.waitKey(1) & 0xff == ord('r'):
                cnt += 1
                for name, frm in rgb_frms.items():
                    filename = output_dir + name + '/' + str(cnt).zfill(2) + '.png'
                    print(f'Writing image : {filename}')
                    cv2.imwrite(filename,frm)
            if cv2.waitKey(1) & 0xff == ord('q'):
                for name in self.cameras:
                    cv2.destroyWindow(name)
                    self.cameras[name].release()

    def stereo_calibrate(self):
        ########## Stereo Vision Calibration #############################################
        # self.flags |= cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_FOCAL_LENGTH + cv2.CALIB_FIX_PRINCIPAL_POINT
        self.flags |= cv2.CALIB_FIX_INTRINSIC
        # Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
        # Hence intrinsic parameters are the same 

        criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        cams = list(self.cam_params.keys())
        # two dictionaries to hold camera parameters
        camL = dict()
        camR = dict()
        params = [camL, camR]
        stereo_params = {name: param for name, param in zip(cams, params)}
        # This step is performed to transformation between the two cameras,
        # and calculate Essential and Fundamental matrix
        retStereo, newCameraMatrixL, distL, newCameraMatrixR, \
        distR, rot, trans, essentialMatrix, fundamentalMatrix = cv2.stereoCalibrate(self.objpoints[cams[0]],
                                                                                    self.imgpoints[cams[0]],
                                                                                    self.imgpoints[cams[1]],
                                                                                    self.cam_params[cams[0]]['intrinsic'],
                                                                                    self.cam_params[cams[0]]['distortion'],
                                                                                    self.cam_params[cams[1]]['intrinsic'],
                                                                                    self.cam_params[cams[1]]['distortion'],
                                                                                    self.img_dims,
                                                                                    criteria_stereo, 
                                                                                    flags=self.flags)
        if retStereo:
            if retStereo > 1:
                print('High Stereo Re-Projection error.')
                print('Check stereo image pairs.')
            # unique parameters
            stereo_params[cams[0]]['intrinsic'] = newCameraMatrixL
            stereo_params[cams[0]]['distortion'] = distL
            stereo_params[cams[1]]['intrinsic'] = newCameraMatrixR
            stereo_params[cams[1]]['distortion'] = distR
            # shared parameters
            stereo_params['rotation'] = rot
            stereo_params['translation'] = trans
            stereo_params['e_Mat'] = essentialMatrix
            stereo_params['f_Mat'] = fundamentalMatrix
            self.stereo_params = stereo_params
            print(f'Stereo Calibration Error (RMSE): {retStereo}')
            print(f'Rotation Matrix \n{rot}')
            print(f'Translation Matrix \n{trans}')
            print(f"{cams[0]} Matrix \n{stereo_params[cams[0]]['intrinsic']}")
            print(f"{cams[1]} Matrix \n{stereo_params[cams[1]]['intrinsic']}")
        else:
            print('Stereo Calibration failure...')

    def get_rectification_params(self, save_params=True):
        ########## Stereo Rectification #################################################
        # list of camera names to serve as keys
        cams = list(self.cam_params.keys())
        # two dictionaries to hold camera parameters
        camL = dict()
        camR = dict()
        params = [camL, camR]
        stereo_maps= {name: param for name, param in zip(cams, params)}
        print(f'Rectifying cams: {cams}')
        rectifyScale= 1
        rectL, rectR, projMatrixL, \
        projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(self.stereo_params[cams[0]]['intrinsic'],
                                                        self.stereo_params[cams[0]]['distortion'],
                                                        self.stereo_params[cams[1]]['intrinsic'],
                                                        self.stereo_params[cams[1]]['distortion'],
                                                        self.img_dims,
                                                        self.stereo_params['rotation'],
                                                        self.stereo_params['translation'],
                                                        # flags=cv2.CALIB_ZERO_DISPARITY)
                                                        rectifyScale,
                                                        (0,0))
        results = (rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R)
        
        for idx, cam in enumerate(cams):
            stereo_maps[cam] = cv2.initUndistortRectifyMap(self.stereo_params[cam]['intrinsic'],
                                                        self.stereo_params[cam]['distortion'],
                                                        results[0+idx],
                                                        results[2+idx],
                                                        self.img_dims,
                                                        cv2.CV_16SC2)

        self.stereoMaps = stereo_maps
        
        if save_params:
            print("Saving parameters!")
            cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)
            cv_file.write('stereoMap_camera_l_x',self.stereoMaps[cams[0]][0])
            cv_file.write('stereoMap_camera_l_y',self.stereoMaps[cams[0]][1])
            cv_file.write('stereoMap_camera_r_x',self.stereoMaps[cams[1]][0])
            cv_file.write('stereoMap_camera_r_y',self.stereoMaps[cams[1]][1])
            cv_file.release()
        return stereo_maps

    def stereo_rectify(self, frameL, frameR, saved_params=False, display=False):
        cams = self.cam_nms
        if saved_params:
            # Load camera parameters to undistort and rectify images
            cv_file = cv2.FileStorage()
            cv_file.open('./calibration/stereoMap.xml', cv2.FileStorage_READ)

            stereoMapL_x = cv_file.getNode('stereoMap_' + cams[0] + '_x').mat()
            stereoMapL_y = cv_file.getNode('stereoMap_' + cams[0] + '_y').mat()
            stereoMapR_x = cv_file.getNode('stereoMap_' + cams[1] + '_x').mat()
            stereoMapR_y = cv_file.getNode('stereoMap_' + cams[1] + '_y').mat()
        else:
            stereoMapL_x = self.stereoMaps[cams[0]][0]
            stereoMapL_y = self.stereoMaps[cams[0]][1]
            stereoMapR_x = self.stereoMaps[cams[1]][0]
            stereoMapR_y = self.stereoMaps[cams[1]][1]
        # Undistort and rectify images
        rectL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        rectR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        if display:
            merged = np.hstack([rectL, rectR])
            h, w = merged.shape[:2]
            interval = h // 4
            y = 0
            for x in range(4):
                y += interval
                pt1 = (0, y)
                pt2 = (w, y)
                cv2.line(merged, pt1, pt2, color=(0,255,0))
            cv2.imshow('Stereo Rectified', merged)
            cv2.waitKey(0)
        return rectR, rectL

    def get_f(self):
        # assumes cameras have similar focal lengths
        if self.stereo_params:
            f_xs = 0
            f_ys = 0
            for cam in self.stereo_params.keys():
                if cam in ['camera_l', 'camera_r']:
                    fx = self.stereo_params[cam]['intrinsic'][0][0]
                    fy = self.stereo_params[cam]['intrinsic'][1][1]
                    f_xs += fx
                    f_ys += fy
            return (f_xs + f_ys) // 4
        else:
            print('Cameras not calibrated!')
    
    def re_projection_error(self, cam, stereo=False):
        if not stereo:
            mean_error = 0
            for i in range(len(self.objpoints[cam])):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[cam][i],
                                                self.cam_params[cam]['rotation'][i],
                                                self.cam_params[cam]['translation'][i],
                                                self.cam_params[cam]['intrinsic'],
                                                self.cam_params[cam]['distortion'])
                error = cv2.norm(self.imgpoints[cam][i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error
            print( f"Individual camera RMSE: {mean_error/len(self.objpoints[cam])}")
        # Stereo calibrate automatically returns this value, not needed
        if stereo:
            cams = list(self.cam_params.keys())
            for cam in cams:
                mean_error = 0
                for i in range(len(self.objpoints[cam])):
                    imgpoints2, _ = cv2.projectPoints(self.objpoints[cam][i],
                                                    self.stereo_params['rotation'],
                                                    self.stereo_params['translation'],
                                                    self.stereo_params[cam]['intrinsic'],
                                                    self.stereo_params[cam]['distortion'])
                    error = cv2.norm(self.imgpoints[cam][i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                    mean_error += error
                print(f"{cam} stereo RMSE: {mean_error/len(self.objpoints[cam])}")

if __name__ == '__main__':
    # configurations
    cameras = {"camera_l": 0, "camera_r": 2}
    output_dir = './stereo_imgs/'
    stereo_capture = False
    thing = Calibrator(cameras)
    if stereo_capture:
        thing.capture_stereo_images(output_dir)
    thing.get_camera_params(output_dir)
    thing.stereo_calibrate()
    # rig characteristics
    # focal length

    f = thing.get_f()
    # b (cm)
    b = 3.75*2.54

    thing.get_rectification_params()
    # test recftification on calibration images
    img_maps = {name: sorted(glob.glob('./stereo_imgs/' + name + '/*.png')) for name in thing.cam_nms}
    # stereo disparity solver configuration
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
    for pathL, pathR in zip(img_maps[thing.cam_nms[0]], img_maps[thing.cam_nms[1]]):
        imgL = cv2.imread(pathL)
        imgR = cv2.imread(pathR)
        rectL, rectR = thing.stereo_rectify(imgL, imgR, display=True)
        # imgL = cv2.imread(pathL, 0)
        # imgR = cv2.imread(pathR, 0)
        disparity_SGBM = stereo.compute(rectL,rectR)
        disp_img = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                            beta=0, norm_type=cv2.NORM_MINMAX)
        disp_img = np.uint8(disp_img)
        disp_img = cv2.applyColorMap(disp_img, cv2.COLORMAP_MAGMA)
        cv2.imshow("Disparity Map", disp_img)
        # cv2.imshow('Raw Disparity', disparity_SGBM)
        depth_map = (b*f) / (disparity_SGBM + 1)
        # cv2.imshow('Raw Depth', depth_map)
        