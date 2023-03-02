"""
This module provides tools for rectification of videhosnapshots of planar surfaces
based on fundamental matrix.
Workflow is such: to consequent images are chosen, matching feauters are found,
then essential matrix os found and decomposed into R and T matricies.
Based on R and T triangulations for found points is made, so points in 3D are found.
Then a plane that fits all the points is found. Then original image is rotated based on direction 
of the normal of the found plane, so then image becomes planar and one may accurately calculate linear sizes.
"""

from Sonarlab.SonarImaging import drawPoints
from Sonarlab.SonarImaging import undistortImage
from Sonarlab.SonarImaging import detectKeypoints, drawMatches, eqHist, matchKeypoints, estimateInliers
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters, transform
import Sonarlab.Utils as ut
import pandas as pd 
import open3d as o3d

# Util functions

def computeTranslationVector(kpsA, kpsB, mtx):
    """
    Calculation of translation vetor of camera between two images,
    assuming camera tranlsates only, not rotates
    """

    # Intrinsic parameters
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, -1]
    cy = mtx[1, -1]

    vec_sign = 0


     # Matrix for vanihing point calc
    eq_motion_v_matrix = []
    for X1, X2 in zip(kpsA, kpsB):
        x1, y1, x2, y2 = [*X1, *X2]

        # Count all vectors for y direcion for making sure motion is positive
        vec_sign += y1 - y2

        # Calculate coefficients for translation vector calc
        xx1 = x1 - cx
        xx2 = x2 - cx
        yy1 = y1 - cy
        yy2 = y2 - cy
        tx_coef = yy1*fx - yy2*fy
        ty_coef = xx2*fx - xx1*fy
        tz_coef = yy2*xx1 - yy1*xx2
        # print(tz_coef, xx1, xx2, yy1, yy2)
        eq_motion_v_matrix.append([tx_coef, ty_coef, tz_coef])

    # Calculate motion vector
    # print("EQ MATRIX:", eq_motion_v_matrix)
    eq_motion_v_matrix = np.array(eq_motion_v_matrix)
    u, d, v = np.linalg.svd(eq_motion_v_matrix)
    v = v.transpose()

    # Ensure x vector results as positive motion
    if v[1,-1] * vec_sign > 0:
        tx = v[0,-1]
        ty = v[1,-1]
        tz = v[2,-1]
    else:
        tx = -v[0,-1]
        ty = -v[1,-1]
        tz = -v[2,-1]
    print('TRANSLATION:', [tx,ty,tz])

    tAbs = np.sqrt(tx*tx + ty*ty + tz*tz)

    # Visualize
    print('VISUALIZING TRANSLATION')
    kanvas = np.ones((128,128,3)).astype('uint8')*255
    pt0 = (64,64)
    v_x = int(64 * tx / tAbs)
    v_y = int(64 * ty / tAbs)
    pt1 = (64 + v_x, 64 + v_y)
    cv2.line(kanvas, pt0, pt1, (255, 0, 0), 2)
    cv2.circle(kanvas, pt0, 5, (0,255,0), 2)
    cv2.circle(kanvas, pt1, 5, (255,0,0), 2)
    cv2.imshow('DIR', kanvas)
    cv2.waitKey(10)

    return np.float32([tx, ty, tz])


def computeTranslationVector2(kpsA, kpsB, mtx):
    """
    Calculation of translation vetor of camera between two images,
    assuming camera tranlsates only, not rotates
    """

    # Intrinsic parameters
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    f = (fx + fy)/2
    cx = mtx[0, -1]
    cy = mtx[1, -1]

    vec_sign = 0


     # Matrix for vanihing point calc
    equation_matrix = []
    for X1, X2 in zip(kpsA, kpsB):
        x1, y1, x2, y2 = [*X1, *X2]
        x1 = x1 - cx
        x2 = x2 - cx
        y1 = y1 - cy
        y2 = y2 - cy

        # Count all vectors for y direcion for making sure motion is positive
        vec_sign += y1 - y2

        # Calculate coefficients for translation vector calc
        # Params for VP
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        equation_matrix.append([a, b, c])

    # Calculate motion vector
    # print("EQ MATRIX:", eq_motion_v_matrix)

    equation_matrix = np.array(equation_matrix)
    (u, d, v) = np.linalg.svd(equation_matrix)
    v = v.transpose()
    x = v[0,-1]
    y = v[1,-1]
    z = v[2,-1]
    print( 'Vanishing point: ' + str([x,y,z]))
    vanish_point = (int(x/z), int(y/z))
    print(vanish_point)
    sqr = np.sqrt(x*x + y*y + z*z)

    # Estimate sign of T-vector
    directions = []
    x_dirs = []
    y_dirs = []
    for X1, X2 in zip(kpsA, kpsB):
        x1, y1, x2, y2 = [*X1, *X2]
        directions.append( (x2-x1)*(vanish_point[0] - x1) + (y2-y1)*(vanish_point[1] - x1) )
        x_dirs.append(x2-x1)
        y_dirs.append(y2-y1)
        
    x_sign = np.sign(np.average(x_dirs))
    y_sign = np.sign(np.average(y_dirs))
    z_sign = np.sign(np.average(directions))

    print('SIGN OF X TRANSLATION =', x_sign)
    print('SIGN OF Y TRANSLATION =', y_sign)
    print('SIGN OF Z TRANSLATION =', z_sign)


    # Calculate T-vector
    tx = x/sqr
    ty = y/sqr
    tz=  z*f/sqr

    # Place right sign with respect to most component
    if tz**2/(tx*tx + ty*ty) > 1:
        # Replace with Z sign
        if np.sign(tz) != z_sign:
            tx = -tx
            ty = -ty
            tz = -tz
    elif np.sign(tx) != x_sign:
        # Replace with X sign
        if np.sign(tx) * x_sign < 0:
            tx = -tx
            ty = -ty
            tz = -tz
    else:
        # Replace with Y sign
        if np.sign(ty) != y_sign:
            tx = -tx
            ty = -ty
            tz = -tz  

    tAbs = np.sqrt(tx*tx + ty*ty + tz*tz)

    # Visualize
    print('VISUALIZING TRANSLATION:', tx, ty, tz)
    kanvas = np.ones((128,128,3)).astype('uint8')*255
    pt0 = (64,64)
    v_x = int(64 * tx / tAbs)
    v_y = int(64 * ty / tAbs)
    pt1 = (64 + v_x, 64 + v_y)
    cv2.line(kanvas, pt0, pt1, (255, 0, 0), 2)
    cv2.circle(kanvas, pt0, 5, (0,255,0), 2)
    cv2.circle(kanvas, pt1, 5, (255,0,0), 2)
    cv2.imshow('DIR', kanvas)
    cv2.waitKey(10)

    return np.float32([tx, ty, tz])


def computePlaneNormal(kpsA, kpsB, mtx, trans_vector):
    # Intrinsic parameters
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    f = (fx+fy)/2
    cx = mtx[0, -1]
    cy = mtx[1, -1]
    TX, TY, TZ = trans_vector

    vec_sign = 0


     # Matrix for vanihing point calc
    eq_matrix_pure_plane =  []
    for X1, X2 in zip(kpsA, kpsB):
        x0, y0, x1, y1 = [*X1, *X2]

        A = f*TX*x0 - x1*x0*TZ
        B = f*TX*y0 - x1*y0*TZ
        C = f*x1 - f*x0
        D = x1*TZ*f - f*f*TX

        A1 = f*TY*x0 - y1*x0*TZ
        B1 = f*TY*y0 - y1*y0*TZ
        C1 = f*y1 - f*y0
        D1 = y1*TZ*f - f*f*TY

        if all([A,B,C,D,A1,B1,C1,D1]) > 0.001:
            eq_matrix_pure_plane.append([A,B,C,D])
            eq_matrix_pure_plane.append([A1,B1,C1,D1])

    eq_matrix_pure_plane = np.array(eq_matrix_pure_plane)
    (u, d, v) = np.linalg.svd(eq_matrix_pure_plane)
    v = v.transpose()

    # Plane coefs
    aa = v[-1,0] / v[-1,-1]
    bb = v[-1,1] / v[-1,-1]
    dd = v[-1,2] / v[-1,-1]
    cc = -1.0

    return np.array([aa,bb,cc,dd])


def compute_essential_matrix(matches, status, kpsA, kpsB, intrinsic, method=cv2.FM_RANSAC, threshold=0.2, prob=0.6):
    """Use the set of good mathces to estimate the Essential Matrix.

    See  https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
    for more info.
    """
    pts1, pts2 = [], []
    essential_matrix, inliers = None, None
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            ptA = (kpsA[queryIdx][0], kpsA[queryIdx][1])
            ptB = (kpsB[trainIdx][0], kpsB[trainIdx][1])
            pts1.append(ptA)
            pts2.append(ptB)
    if pts1 and pts2:
        essential_matrix, inliers = cv2.findEssentialMat(
            np.float32(pts1),
            np.float32(pts2),
            intrinsic,
            method=method,
            threshold = threshold,
            prob = prob
#             threshold = 3
        )

    ptsA = [pts1[i] for i in range(len(pts1)) if inliers[i]]
    ptsB = [pts2[i] for i in range(len(pts2)) if inliers[i]]
    
    return essential_matrix, inliers, ptsA, ptsB



def getGoodKps(kpsA, kpsB, matches, status):
    """
    Filtes keypoints of matched images based on status"""
    pts1, pts2 = [], []
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            ptA = (kpsA[queryIdx][0], kpsA[queryIdx][1])
            ptB = (kpsB[trainIdx][0], kpsB[trainIdx][1])
            pts1.append(ptA)
            pts2.append(ptB)
    return np.array(pts1), np.array(pts2)


def rotMatrixFromNormal(a,b,c):
    # a,b,c - coordinates of Z component of rotation
    # Get Z component of rotation
    len_Z = np.sqrt(a*a + b*b + c*c)
    a = a / len_Z
    b = b / len_Z
    c = c / len_Z

    Z = (a, b, c) if c > 0 else (-a, -b, -c)
    
    # Get Y component of rotation from assumption
    # that Y perpendicular to Z and x component of Y is zero
    yy = np.sqrt(1 / (1 + b*b / (c*c)))
    xy = 0.0
    zy = -b * yy / c
    Y = (xy, yy, zy)
    
    # Get X component of rotation
    X = np.cross(Y, Z)
    
    ret = np.vstack((X,Y,Z)).transpose()
    
    return ret


def rotateImagePlane(img, K, R):
    """
    Rotation of image plane based on formula
    X' = RX - RD + D
    where X = K_inv*(u,v,1)
    (u', v', 1) = KX'
    """
    
    # Load intrinsic and invert it
    K_inv = np.linalg.inv(K)

    # Augmented inverse for further mtx multiplication
    K_inv1 = np.vstack((K_inv, np.array([0,0,1])))

    # Z distance constant, 1
    d = np.array([0,0,1]).transpose()

    # Calculate translation vector
    t = (d - R.dot(d)).reshape((3,1))
    R1 = np.hstack((R, t))

    # Calc result homography
    matrix = K @ R1 @ K_inv1
    
    # Rotate image
    tf_img = cv2.warpPerspective(img, np.linalg.inv(matrix), (img.shape[1], img.shape[0]))
    
    # return tf_img
    return cv2.normalize(tf_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    



def RectfyImage(img1, img2, mtx_in, dst_in, SCALE_FACTOR=1.0, lo_ratio=0.5,
                ransac_thresh=5,
                filter_step=10,
                crop=True,
                show_point_cloud=False,
                show_images = True,
                NORMALS_ONLY=False,
                MOTION_ONLY=False):
    """
    frame1 and frame2 are colored images of same size
    mtx and dst - are intrinsic matrix and distortion coefficients
    SCALE_FACTOR is a multipliter for image size
    """
    # Motion in frame array
    motion_in_frame = 0.0

    # Resize input images if scalea-fractor is specified
    new_shape = (int(img1.shape[1] * SCALE_FACTOR), int(img1.shape[0] * SCALE_FACTOR))
    mtx = mtx_in * SCALE_FACTOR
    mtx[-1,-1] = 1.0


    frame1 = undistortImage(img1, mtx_in, dst_in, False)
    frame2 = undistortImage(img2, mtx_in, dst_in, False)

    frame1 = cv2.resize(frame1, new_shape)
    frame2 = cv2.resize(frame2, new_shape)

    frame1 = eqHist(frame1)
    frame2 = eqHist(frame2)

    h,w  = frame1.shape[:2]
    

    # Detect featers and estimate inliers
    kps1, ds1 = detectKeypoints(frame1)
    kps2, ds2 = detectKeypoints(frame2)

    matcher = matchKeypoints(ds1, ds2, lo_ratio)
    ret = estimateInliers(matcher, kps1, kps2, ransac_thresh)
    if ret is not None:
        matches, H, status = ret
        # print(f'{len(matches)} matches found')
    else:
        print('No good features found')
        return frame1, np.float32([0, 0, 0]), []

    if show_images:
        sh_f = drawMatches(frame1, frame2, kps1, kps2, matches, status)
        sh_f = cv2.resize(sh_f, (1024, 400))
        shifted_frame = cv2.warpPerspective(frame2, H, (w,h))
        cv2.imshow("WARP", frame1)
        cv2.waitKey(1000)
        cv2.imshow("WARP",shifted_frame)
        cv2.waitKey(1000)
        cv2.imshow('Matches', sh_f)
        cv2.waitKey(10)

    ptsA, ptsB = getGoodKps(kps1, kps2, matches, status)
    abs_motion = np.average(np.linalg.norm(ptsB - ptsA, axis=1))
    print('POINTS SHAPE:', ptsA.shape, 'ABS MOTION:', abs_motion)
    if MOTION_ONLY:
        return abs_motion
    # Computer essential matrix and recover R and T
    try:

        # COMPUTE TRANSLATION DIRECTLY
        
        R2 = np.identity(3)
        translate = computeTranslationVector2(ptsA, ptsB, mtx)
        # with open('trans_vector.csv', 'a') as fr:
        #     fr.write(';'.join([str(e) for e in translate]) + '\n')

    except (TypeError, np.linalg.LinAlgError):
        # print('Failed to calculate essntial matrix')
        return frame1, np.float32([0, 0, 0]), ([], 0.0)

    # Calc abs motion
    

    # Build projection matricies and triangulate points
    ptsA = np.float32(ptsA)
    ptsB = np.float32(ptsB)
    proj_mtx01 = np.zeros((3,4))
    proj_mtx01[:3,:3] = np.identity(3)
    proj_mtx01 = mtx @ proj_mtx01

    proj_mtx02 = np.zeros((3,4))
    proj_mtx02[:3,:3] = R2
    proj_mtx02[:, -1] = translate.transpose()
    proj_mtx02 = mtx @ proj_mtx02

    points3d = cv2.triangulatePoints(proj_mtx01, proj_mtx02,
                                    ptsA.transpose(),
                                    ptsB.transpose())

    
    # Visualize point cloud
    if show_point_cloud:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(calib_points)
        o3d.visualization.draw_geometries([pcd])

    # Estimating plane for 3d points
    calib_points = (points3d / points3d[-1, :]).transpose()

    _, _, v = np.linalg.svd(calib_points)
    v = v.transpose()
    a, b, c, d = v[:, -1]
    if c < 0:
        a, b, c, d = [-element for element in (a,b,c,d)]
    nomal_abs = np.sqrt(a*a + b*b + c*c)

    print("Coefficients of the plane found: ", a,b,c,d)
    # print('Another ones:', computePlaneNormal(ptsA, ptsB, mtx, translate))
    # a,b,c,d = computePlaneNormal(ptsA, ptsB, mtx, translate)

    # Returnb only plane normal, if flag says
    if NORMALS_ONLY:
        return None, np.float32([a/nomal_abs, b/nomal_abs, c/nomal_abs]), []

    # Calcilate distances CALIB from points to plane
    distances = []
    sqr = np.sqrt(a*a + b*b + c*c)
    for pt in calib_points:
        x, y, z, _ = pt
        dist = (a*x + b*y + c*z + d) / sqr
        distances.append(dist)

    # Rotate 1st imput frame
    rotation_mtx = rotMatrixFromNormal(a,b,c)
    frame = undistortImage(img1, mtx_in, dst_in, crop=crop)
    output = rotateImagePlane(frame, mtx_in, rotation_mtx)
    print('Rotated image figured\n\n')
    

    return output, np.float32([a/nomal_abs, b/nomal_abs, c/nomal_abs]), (distances, abs_motion)



if __name__ == "__main__":

    from tkinter import filedialog
    import os
    import glob
    from SonarImaging import VideoPlayer
    # from scipy.spatial.transform import Rotation as Rot

    # images = glob.glob('*.jpg')
    # print(images)
    npy_files = glob.glob('*.npy')
    print(npy_files)
    dst, mtx = [np.load(y) for y in npy_files]

    PATH = 'D:\DATA'
    video_files = glob.glob(os.path.join(PATH, '*.mp4')) + \
            glob.glob(os.path.join(PATH, '*.avi'))

    for video_file in video_files:
        FILE = os.path.split(video_file)[-1]
        print(FILE)
        v = VideoPlayer()
        v.openVideoFile(video_file)
        v.setScaleFactor(4)

        v.getNextFrame()

        # Set how many frames to skip
        v.setFrameStep(5)

        frame_count = 0
        while v.playing:

            print(f'Frame {v.cur_frame_no} of {v.vid_frame_length}')
            frame1 = v.getCurrentFrame()
            h, w, _ = frame1.shape
            new_shape = (w//4, h//4)
            # key = v.waitKeyHandle()
            prev_normal = [0.0, 0.0, 1.0]
            # if key == ord('F'):
            if v.playing:

                c_key = 0
                normals = []
                motions = []
                iteration = 10
                iteration_step = 5
                rec_step=2
                max_iter = 50
                cosine = 0.0
                motion = 1
                
                while iteration < max_iter and motion < 35 :   

                    print(f"------ITERATION # {iteration}--------")
                    print(f"Cosine figured =  {cosine}")
                    iteration += iteration_step

                    if v.cur_frame_no +  rec_step *  iteration < v.vid_frame_length:
                        print('ATTEMPT FRAME PLUS', v.cur_frame_no + rec_step *  iteration)
                        v.video.set(1, v.cur_frame_no + rec_step * iteration)
                        ret, frame2 = v.video.read()
                        v.video.set(1, v.cur_frame_no)
                    else:
                        print('ATTEMPT FRAME MINUS', v.cur_frame_no - rec_step  *  iteration)
                        v.video.set(1, v.cur_frame_no - rec_step  *  iteration)
                        ret, frame2 = v.video.read()
                        v.video.set(1, v.cur_frame_no)
                    # v.getNextFrame()
                    # frame2 = v.getCurrentFrame()
                    

                    # sh_f2 = cv2.resize(frame2, new_shape)
                    # # cv2.imshow('frame2', sh_f2)

                    rec_ret = RectfyImage(frame1, frame2, mtx, dst, SCALE_FACTOR=0.2, filter_step=1,
                    lo_ratio=0.7,
                    ransac_thresh=3,
                    show_images=False)
                    
                    if rec_ret is None:
                        pass
                    else:
                        rectified, coeffs, extra_info = rec_ret
                        motion = extra_info[1]
                        motions.append(motion)
                        # cosine = coeffs[-1] / ((coeffs[0]**2 + coeffs[1]**2 + coeffs[2] **2) ** 0.5)

                
                # CHECK FOR COSINE TO BE CLOSE TO 1
                print('PLANE normal: ', coeffs)
                cosine = coeffs[-1] / ((coeffs[0]**2 + coeffs[1]**2 + coeffs[2] **2) ** 0.5)
                if cosine > 0.6:
                    normals.append(coeffs)
                    prev_normal = coeffs
                    normal = coeffs
                else:
                    print(' ----  COSINE BAD -----')
                    normals.append(prev_normal)
                    normal = prev_normal

                rec_show = cv2.resize(rectified, new_shape)
                cv2.imshow('Recified frame', rec_show)



                # # Save normals to file
                # fname = f'normals_{FILE[:-4]}.csv'
                # with open(os.path.join(PATH, fname), 'a') as f:
                #     f.write(";".join([str(v.cur_frame_no)] + [str(a) for a in normal]) + '\n')


                c_key = cv2.waitKey(1)


                # cv2.imshow('Recified frame', output)
                # cv2.imwrite(f'2/frame{frame_count:0>3}.jpg', output)
                frame_count += 1
                # plt.close()
                v.playStepForwards()
                # v.show()
        fig, ax = plt.subplots()
        ax.plot(motions)
        plt.show()
