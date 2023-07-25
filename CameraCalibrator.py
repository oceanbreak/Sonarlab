import numpy as np
import cv2
import glob
import os
from Sonarlab.SonarImaging import eqHist

#### Module that provides tools for camera calibration ######

class CameraCalibrator:


    def __init__(self, images_names, grid_shape):
        self.__scale_factor = 0.5
        self.__criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.__img_list = images_names
        self.__output_path = os.path.split(images_names[0])[0]

        # Chessboard shape
        self.__grid_shape = grid_shape
        pass

    def setGridShape(self, new_shape):
        if len(new_shape) == 2:
            self.__grid_shape = tuple(new_shape)
        else:
            raise ValueError('Grid shape must be array of 2')
        
    def setScaleFractor(self, new_scale_factor):
        self.__scale_factor = new_scale_factor

    def setOutputPath(self, new_path):
        self.__output_path = new_path

    def calibrate(self):
        xshape, yshape = self.__grid_shape[0], self.__grid_shape[1]
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((xshape*yshape,3), np.float32)
        objp[:,:2] = np.mgrid[0:xshape,0:yshape].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for fname in self.__img_list:
            img = cv2.imread(fname)
            print('Reading %s' % fname)
            img = cv2.resize(img, (int(img.shape[1] * self.__scale_factor), 
                                   int(img.shape[0] * self.__scale_factor)))
            img = eqHist(img)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.__grid_shape, None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1), self.__criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, self.__grid_shape, corners2, ret)
                im_show = cv2.resize(img, (960, 740))
                cv2.imshow('img',im_show)
                cv2.waitKey(500)

        cv2.destroyAllWindows()


        print('Performing calibration...')

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        # Test Calibration
        img = cv2.imread(self.__img_list[0])
        img = cv2.resize(img, (int(img.shape[1] * self.__scale_factor), 
                               int(img.shape[0] * self.__scale_factor)))

        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        print('Undistorting image...')
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        cv2.imshow('Distorted', img)
        cv2.imshow('Undistorted', dst)
        cv2.waitKey()

        dist_large = dist
        mtx_large = mtx * self.__scale_factor
        mtx_large[-1, -1] = 1.0

        np.save(self.__output_path + 'mtx.npy', mtx_large)
        np.save(self.__output_path + 'dst.npy', dist_large)
        print('Saved')


        print('Handling full sized image')
        img = cv2.imread(glob.glob('data/calibrate/BD3570/*.JPG')[8])
        h,  w = img.shape[:2]
        newcameramtx_large, roi=cv2.getOptimalNewCameraMatrix(mtx_large,dist_large,(w,h),1,(w,h))
        dst = cv2.undistort(img, mtx_large, dist_large, None, newcameramtx_large)

        dst_show = cv2.resize(dst, (960, 720))
        img_show = cv2.resize(img, (960, 720))

        cv2.imshow('Distorted', img_show)
        cv2.imshow('Undistorted', dst_show)
        cv2.waitKey()

