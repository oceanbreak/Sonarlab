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
                show_point_cloud=False,
                show_images = True,
                NORMALS_ONLY=False):
    """
    frame1 and frame2 are colored images of same size
    mtx and dst - are intrinsic matrix and distortion coefficients
    SCALE_FACTOR is a multipliter for image size
    """

    # Resize input images if scalea-fractor is specified
    new_shape = (int(img1.shape[1] * SCALE_FACTOR), int(img1.shape[0] * SCALE_FACTOR))
    mtx = mtx_in * SCALE_FACTOR
    dst = dst_in
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

    # Computer essential matrix and recover R and T
    try:

        # Ess, inliers, ptsA, ptsB = compute_essential_matrix(matches, status, kps1, kps2, mtx)
        # _, R2, translate,  _ = cv2.recoverPose(Ess, np.float32(ptsA), np.float32(ptsB), mtx)


        # COMPUTE TRANSLATION DIRECTLY
        ptsA, ptsB = getGoodKps(kps1, kps2, matches, status)
        R2 = np.identity(3)
        translate = computeTranslationVector(ptsA, ptsB, mtx)
        # with open('trans_vector.csv', 'a') as fr:
        #     fr.write(';'.join([str(e) for e in translate]) + '\n')

    except (TypeError, np.linalg.LinAlgError):
        # print('Failed to calculate essntial matrix')
        return frame1, np.float32([0, 0, 0]), []


    #TRY ZEERO OUT ROATTION
    # R2 = np.identity(3)


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

    # calib_points = (points3d / points3d[-1, :]).transpose()

    # msk1 = ut.goRansac(calib_points[:, 0].reshape(-1, 1), 
    #             calib_points[:, 2].reshape(-1, 1), 
    #             thresh = 2,
    #             show_plot=True)
    # msk2 = ut.goRansac(calib_points[:, 0].reshape(-1, 1), 
    #             calib_points[:, 1].reshape(-1, 1), 
    #             thresh = 2,
    #             show_plot=True)
    
    # mask = msk1 * msk2

    # calib_points = np.float32([calib_points[i, :] for i in range(calib_points.shape[0]) if mask[i]])


    # print('Triangulation for 3d points figured')

    # Estimating plane for 3d points
    calib_points = (points3d / points3d[-1, :]).transpose()

    _, _, v = np.linalg.svd(calib_points)
    v = v.transpose()
    a, b, c, d = v[:, -1]
    if c < 0:
        a, b, c, d = [-element for element in (a,b,c,d)]
    nomal_abs = np.sqrt(a*a + b*b + c*c)

    print("Coefficients of the plane found: ", a,b,c,d)

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
    frame = undistortImage(img1, mtx_in, dst_in, True)
    output = rotateImagePlane(frame, mtx_in, rotation_mtx)
    print('Rotated image figured\n\n')
    

    return output, np.float32([a/nomal_abs, b/nomal_abs, c/nomal_abs]), distances



if __name__ == "__main__":

    from tkinter import filedialog
    import glob
    from SonarImaging import VideoPlayer
    # from scipy.spatial.transform import Rotation as Rot

    # images = glob.glob('*.jpg')
    # print(images)
    npy_files = glob.glob('*.npy')
    print(npy_files)
    dst, mtx = [np.load(y) for y in npy_files]

    PATH = 'D:\DATA\Videomodule video samples/shaky_clean/'
    FILE = 'shaky_clean_7494_01.mp4'

    video_file =  PATH + FILE
    print(video_file)
    v = VideoPlayer()
    v.openVideoFile(video_file)
    v.setScaleFactor(4)

    v.getNextFrame()

    # Set how many frames to skip
    v.setFrameStep(1)

    frame_count = 0
    while v.playing:

        print(f'Frame {v.cur_frame_no} of {v.vid_frame_length}')
        frame1 = v.getCurrentFrame()
        h, w, _ = frame1.shape
        new_shape = (w//4, h//4)
        # key = v.waitKeyHandle()
        # if key == ord('F'):
        if v.playing:

            c_key = 0
            normals = []
            iteration = 0
            rec_step=18
            
            while iteration < 1:   

                print(f"------ITERATION # {iteration}--------")
                iteration += 1

                try:
                    v.video.set(1, v.cur_frame_no + rec_step)
                except:
                    v.video.set(1, v.cur_frame_no - rec_step)
                # v.getNextFrame()
                # frame2 = v.getCurrentFrame()
                ret, frame2 = v.video.read()

                # sh_f2 = cv2.resize(frame2, new_shape)
                # # cv2.imshow('frame2', sh_f2)

                rec_ret = RectfyImage(frame1, frame2, mtx, dst, SCALE_FACTOR=0.2, filter_step=1,
                lo_ratio=0.7,
                ransac_thresh=3,
                show_images=False)
                if rec_ret is None:
                     break

                rectified, coeffs, distances = rec_ret
                normals.append(coeffs)

                rec_show = cv2.resize(rectified, new_shape)
                cv2.imshow('Recified frame', rec_show)


                # Save normals to file
                with open(f'normals_{FILE[:-4]}.csv', 'a') as f:
                    f.write(";".join([str(v.cur_frame_no)] + [str(a) for a in coeffs]) + '\n')


                c_key = cv2.waitKey(1)


            # cv2.imshow('Recified frame', output)
            # cv2.imwrite(f'2/frame{frame_count:0>3}.jpg', output)
            frame_count += 1
            # plt.close()
            v.playStepForwards()
            # v.show()
