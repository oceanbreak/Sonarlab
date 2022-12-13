"""
This module provides tools for rectification of videhosnapshots of planar surfaces
based on fundamental matrix.
Workflow is such: to consequent images are chosen, matching feauters are found,
then essential matrix os found and decomposed into R and T matricies.
Based on R and T triangulations for found points is made, so points in 3D are found.
Then a plane that fits all the points is found. Then original image is rotated based on direction 
of the normal of the found plane, so then image becomes planar and one may accurately calculate linear sizes.
"""

from SonarImaging import drawPoints
from SonarImaging import undistortImage
from SonarImaging import detectKeypoints, drawMatches, eqHist, matchKeypoints, estimateInliers
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, filters, transform

# Util functions

def compute_essential_matrix(matches, status, kpsA, kpsB, intrinsic, method=cv2.FM_RANSAC):
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
#             threshold = 3
        )
    return essential_matrix, inliers, pts1, pts2



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
    return np.array(pts1), np.array(pts1)


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
    tform = transform.ProjectiveTransform(matrix=matrix)
    tf_img = transform.warp(img, tform)
    
    return cv2.normalize(tf_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)    


def RectfyImage(img1, img2, mtx_in, dst_in, SCALE_FACTOR=1.0):
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
    frame1 = cv2.resize(img1, new_shape)
    frame2 = cv2.resize(img2, new_shape)

    # Detect featers and estimate inliers
    kps1, ds1 = detectKeypoints(frame1)
    kps2, ds2 = detectKeypoints(frame2)

    matcher = matchKeypoints(ds1, ds2, 0.4)
    ret = estimateInliers(matcher, kps1, kps2, 3)
    if ret is not None:
        matches, H, status = ret
    else:
        print('No good features found')
        return None

    # Computer essential matrix and recover R and T
    Ess, inliers, ptsA, ptsB = compute_essential_matrix(matches, status, kps1, kps2, mtx)
    _, R2, translate,  _ = cv2.recoverPose(Ess, np.float32(ptsA), np.float32(ptsB), mtx)

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

    # Estimating plane for 3d points
    calib_points = (points3d / points3d[-1, :]).transpose()
    u, dd, v = np.linalg.svd(calib_points)
    v = v.transpose()
    a, b, c, d = v[:, -1]

    print("Coefficients of the plane found: ", a,b,c,d)

    # Calcilate distances CALIB from points to plane
    distances = []
    sqr = np.sqrt(a*a + b*b + c*c)
    for pt in calib_points:
        x, y, z, _ = pt
        dist = (a*x + b*y + c*z + d) / sqr
        distances.append(dist)
        
    # plt.scatter(np.arange(len(distances)), distances)
    # plt.show()
    

    # Rotate 1st imput frame
    rotation_mtx = rotMatrixFromNormal(a,b,c)
    frame = undistortImage(img1, mtx_in, dst_in, True)
    output = rotateImagePlane(frame, mtx_in, rotation_mtx)
    
    return output



if __name__ == "__main__":

    from tkinter import filedialog
    import glob

    images = filedialog.askopenfilenames(title="SELECT PHOTOS", filetypes=(('JPEG files', '*.jpg'), ('PNG files', '*.png')))
    print(images)
    npy_files = glob.glob('*.npy')
    mtx = np.load(npy_files[1])
    dst = np.load(npy_files[0])

    if len(images) < 2:
        print('Select at least 2 images')
        quit()

    for i in range(1, len(images)):
        img1 = cv2.imread(images[i-1])
        img2 = cv2.imread(images[i])
        cv2.imshow('img1', img1)
        cv2.imshow('img2', img2)
        cv2.waitKey(100)
        output_image = RectfyImage(img1, img2, mtx, dst, 0.25)

    cv2.imshow('output', output_image)
    cv2.imwrite('result.png', output_image)

