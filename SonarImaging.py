import cv2
import numpy as np
from matplotlib import pyplot as plt
from functools import partial
import math


### FUNCTIONS ###

def eqHist(image, clache=True, gray_only=False):
    """
    Equalization of image, by default based on CLACHE method
    """
    if gray_only:
        L = image
    else:
        imgHLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        L = imgHLS[:,:,1]
    if clache:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equ = clahe.apply(L)
    else:
        equ = cv2.equalizeHist(L)

    if gray_only:
        return equ

    imgHLS[:,:,1] = equ
    res = cv2.cvtColor(imgHLS, cv2.COLOR_HLS2BGR)
    return res


def equalSize(img_photo, img_video):
    """
    Function crops video to match photo ratio.
    Then it scales photo to video size.
    :param img_photo - RGB photo image
    :param img_video - RGB videi omage
    :return: 2 images of equal size, scale and crop factor
    """
    height_ph, width_ph = img_photo.shape[:2]
    height_v, width_v = img_video.shape[:2]

    v_rate = width_v / height_v
    ph_rate = width_ph / height_ph

    # Crops video according to it's difference in ratio with photo
    delta_x = delta_y = 0
    if ph_rate > v_rate:
        delta_y = int(0.5 * (height_v - width_v / ph_rate))
        if delta_y == 0:
            new_video = img_video
        else:
            new_video = img_video[delta_y : -delta_y, :, :]
    elif ph_rate < v_rate:
        delta_x = int(0.5 * (width_v - height_v * ph_rate))
        if delta_x == 0:
            new_video = img_video
        else:
            new_video = img_video[:, delta_x : -delta_x, :]
    else:
        new_video = img_video

    scale_factor = img_photo.shape[0] / new_video.shape[0]
    new_photo = cv2.resize(img_photo, (new_video.shape[1], new_video.shape[0]))

    return (new_photo, new_video, scale_factor)


def drawPoints(image, *points, radius=5, color=(0,0,255), thickness=2, scaler=1):
    """
    Draw points on image preserving scale
    """
    for pt in points:
        pt = tuple([int(x*scaler) for x in pt])
        cv2.circle(image, pt, radius, color, thickness)


def calcDistance(pt1, pt2):
    y1, x1 = pt1
    y2, x2 = pt2
    return math.sqrt((y1 - y2)**2 + (x1 - x2)**2)


def undistortImage(img, mtx, dist, crop=False):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    ret = cv2.undistort(img, mtx, dist, None, newcameramtx)
    if crop:
        x, y, w1, h1 = roi
        ret = ret[y:y+h1, x:x+w1]
        ret = cv2.resize(ret, (w,h))
    return ret


def sliceImage(image, ptA, ptB):
    rr, cc = cv2.line(*ptA, *ptB)
    values = image[rr, cc]
    return (values, rr, cc)


############# Detection and descriptpors section #################

def detectKeypoints(image, template=None):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    try:
        sift = cv2.xfeatures2d.SIFT_create()
    except:
        sift = cv2.ORB_create()
    kps, descript = sift.detectAndCompute(gray, None)
    kps = np.float32([kp.pt for kp in kps])

    if template is None:
        pass
    elif template.shape[:2] != image.shape[:2]:
        print('Dimensions of image and template dont match')
        pass
    else:
        print('Applying template')
        new_kps = []
        new_descriptors = []
        for pt, descriptor in zip(kps, descript):
            x, y = pt
            if template[int(y),int(x),0] > 50:
                new_kps.append(pt)
                new_descriptors.append(descriptor)
        kps = np.array(new_kps)
        descript = np.array(new_descriptors)
    return (kps, descript)

def matchKeypoints(des1, des2, ratio):
    # print('matchKeypoints')
    matcher = cv2.DescriptorMatcher_create('BruteForce')
    try:
        rawMatches = matcher.knnMatch(des1, des2, 2)
    except:
        return None

    # Lowe's ratio test
    matches = []
    for m in rawMatches:
        if len(m) ==2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        return np.array(matches)
    else:
        return None

def estimateInliers(matches, kpsA, kpsB, reprojThresh):
    if matches is not None:
        ptsA = np.float32([kpsA[i] for i in matches[:,1]])
        ptsB = np.float32([kpsB[i] for i in matches[:,0]])
        (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    else:
        return None

def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    # loop over the matches
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    # return the visualization
    return vis

####################################################

### CLASSES ###
class VideoPlayer:
    """
    This class opens video file and playback it.
    It has some keyboard controls listed below. Also it can return current frame and has an ability to extend its functionality.
    Controls:
    # PLAYER window
    w - play forward
    s - play backward
    f - scale +
    g - scale -
    Esc - quit

    Controls of time step between frames:
    1 - one frame
    2 - 0.5 sec
    3 - 1 sec
    4 - 10 sec
    5 - 30 sec
    6 - 1 min

    """

    def __init__(self):

        #### VARIABLES ####
        self.video = None
        self.video_name = None
        self.frame_step = 1
        self.vid_frame_length = 0
        self.cur_frame_no = 0
        self.fps = 0
        self.scale_factor = 1.0
        self.current_frame = None

        #### FLAGS #####
        self.playing = False
        self.is_loaded = False

        ### FUNCTIONS for key interrupt handling ####
        self.func_set = {'w' : self.playStepForwards, 
                         's' : self.playStepBackwards,
                         'f' : self.scaleUp,
                         'g' : self.scaleDown, 
                         '1' : partial(self.frameStepChange, 0),
                         '2' : partial(self.frameStepChange, 0.5),
                         '3' : partial(self.frameStepChange, 1.0),
                         '4' : partial(self.frameStepChange, 5.0),
                         '5' : partial(self.frameStepChange, 10.0),
                         '6' : partial(self.frameStepChange, 30.0)}


    def addFunction(self, keyword, func, *args):
        """
        Adds function fo self.func_set with keyboard button"""
        if keyword not in self.func_set.keys():
            if len(args) == 0:
                self.func_set.update({keyword : func})
            else:
                self.func_set.update({keyword : partial(func, *args)})
        else:
            print(f' ERROR" Keyword {keyword} reserved')


    def openVideoFile(self, video_file):
        """
        Loads video, checks if it's length > 0, then returns a video object
        """
        try:
            self.video = cv2.VideoCapture(video_file)
            self.vid_frame_length = self.video.get(7)
            self.fps = self.video.get(5)
            self.is_loaded = True
            self.video_name = video_file
            self.playing = False # Flag that says that video is not playing yet
            print(f'INFO: {video_file} loaded successfully')

        except TypeError:
            print('ERROR: %s is not a video' % video_file)
            self.video = None
        if self.vid_frame_length == 0.0:
            self.video = None
            print('ERROR: Zero length video loaded')


    def setScaleFactor(self, value):
        self.scale_factor = value


    def setFrameStep(self, value):
        """
        Value is number of Frames in step
        """
        if value >= 1:
            self.frame_step = int(value)


    def getNextFrame(self):
        """
        Function reads frame from video and returns it
        Returns None if frame is corrupt
        """
        if self.video is None:
            print('ERROR: Video is not opened')
            return None
        try:
            ret, frame = self.video.read()
            # frame = undistortImage(frame, BEWARD_MATRIX, BEWARD_DIST)
        except Exception:
            print('End of video')
        if ret:
            self.current_frame = frame
            self.playing = True
            return frame
        else:
            self.current_frame = None
            return None


    def getCurrentFrame(self):
        return self.current_frame


    def getScaledCurFrame(self):
        """
        Returns a scaled version of current frame
        """
        cur_frame = self.current_frame
        new_size = (int(cur_frame.shape[1] * self.scale_factor), 
                    int(cur_frame.shape[0] * self.scale_factor))
        self.cur_scaled_size = new_size
        return cv2.resize(cur_frame, new_size)


    def show(self):
        """
        Shows current frame if it is not corrupt
        """
        if self.current_frame is not None:
            show_frame = self.getScaledCurFrame()
            cv2.imshow('Player', show_frame)
        else:
            print('WARNING: Corrupt frame')


    def jumpToFrame(self, frame_no):
        """
        Jump to specified frame number
        """
        if self.vid_frame_length > frame_no > 0:
            self.cur_frame_no = frame_no
            self.video.set(1, self.cur_frame_no)
            self.getNextFrame()
        else:
            print(f'{frame_no} is out of bounds for video of length {self.vid_frame_length}')

    
    def playStepForwards(self):
        """
        Updates self.current_frame one step forward N frames, where N = self.frame_step
        """
        print('Forward %i frames' % self.frame_step)
        delta = self.cur_frame_no + self.frame_step
        if delta < self.vid_frame_length:
            self.cur_frame_no += self.frame_step
            self.video.set(1, self.cur_frame_no)
            self.getNextFrame()
        else:
            self.video.set(1, self.vid_frame_length-1)
            self.playing=False
            print('Video End')


    def playStepBackwards(self):
        """
        Updates self.current_frame one step backwards N frames, where N = self.frame_step
        """
        print('Backward %i frames' % self.frame_step)
        delta = self.cur_frame_no - self.frame_step
        if delta >= 0:
            self.cur_frame_no -= self.frame_step
            self.video.set(1, self.cur_frame_no)
            self.getNextFrame()


    def waitKey(self, millisec):
        """
        Sets key wating for specified time interval.
        If Esc pressed, self.playing flag is set to False
        """
        c_key = cv2.waitKey(millisec)
        if c_key & 0xFF == 27:
            self.playing = False
        return c_key


    def frameStepChange(self, sec):
        """
        Function for changing framestep based on sec value
        """
        if sec == 0:
            self.setFrameStep(1)
            print('Step = 1 frame')
        else:
            self.setFrameStep(self.fps * sec)
            print(f'Step = {sec:.1f} sec')


    def scaleUp(self):
        self.scale_factor += 0.05
        print(f'Scale {100 * self.scale_factor:3.2f}%')


    def scaleDown(self):
        self.scale_factor = self.scale_factor - 0.05 if self.scale_factor > 0.1 else  self.scale_factor
        print(f'Scale {100.0 * self.scale_factor:3.2f} %')


    def waitKeyHandle(self):
        """
        Handles waitKey event according to func_set dictionary
        """
        c_key = cv2.waitKey()
        if c_key & 0xFF == 27:
            self.playing = False
            # return None
        idx = chr(c_key)
        if idx in self.func_set.keys():
            ret = self.func_set[idx]()
            # return ret
        return c_key


    def releaseVideo(self):
        if self.video is None:
            print("ERROR: video is not opened")
        else:
            self.video.release()
            print('INFO: Video file closed')
            self.cur_frame_no = 0
            self.vid_frame_length = 0
            self.video = None



if __name__ == "__main__":

    video_path =  'D:\R_20221002_003800_20221002_004054.avi'
    # video_path = 'D:/test1.mp4'

    import glob
    # video_path = glob.glob('*.avi')

    
    player = VideoPlayer()
    player.openVideoFile(video_path)
    player.getNextFrame()
    player.setScaleFactor(0.5)
    while (player.playing):
        player.show()
        player.waitKeyHandle()
    player.releaseVideo()


    # player.play()