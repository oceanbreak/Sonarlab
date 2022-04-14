import cv2
import numpy as np
from matplotlib import pyplot as plt
from functools import partial


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
        self.frame_step = 1
        self.vid_frame_length = 0
        self.cur_frame_no = 0
        self.fps = 0
        self.scale_factor = 1
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
            self.playing = False # Flag that says that video is not playing yet
            print(f'INFO: {video_file} loaded successfully')

        except Exception:
            print('ERROR: %s is not a video' % video_file)
            self.video = None
        if self.vid_frame_length == 0.0:
            self.video = None
            print('ERROR: Zero length video loaded')


    def setScaleFactor(self, value):
        self.scale_factor = value


    def setFrameStep(self, value):
        if value > 0.5:
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
        new_size = (cur_frame.shape[1] // self.scale_factor, cur_frame.shape[0] // self.scale_factor)
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
        self.scale_factor += 1
        print(f'Scale {100 / self.scale_factor:3.2f}%')


    def scaleDown(self):
        self.scale_factor = self.scale_factor - 1 if self.scale_factor > 1 else  self.scale_factor
        print(f'Scale {100.0 / self.scale_factor:3.2f} %')


    def waitKeyHandle(self):
        """
        Handles waitKey event according to func_set dictionary
        """
        c_key = cv2.waitKey()
        if c_key & 0xFF == 27:
            self.playing = False
        idx = chr(c_key)
        if idx in self.func_set.keys():
            self.func_set[idx]()


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

    # video_path =  'D:\DATA\Videomodule video samples/R_20200915_142747_20200915_143147.avi'
    video_path = 'D:/VIDEOPLATFORM_REC/2021.06.28 Ст 7024/R_20210628_005251_20210628_005649.avi'

    def printH(x, y):
        print('Hello %i' % (x+y))
    
    player = VideoPlayer()
    player.addFunction('s', printH, 8,3)
    player.openVideoFile(video_path)
    player.getNextFrame()
    player.setScaleFactor(3)
    while (player.playing):
        # player.show()
        fr = player.getScaledCurFrame()
        cv2.imshow('Frame', fr)
        # player.playStepForwards()
        # player.waitKey(10)
        player.waitKeyHandle()
    player.releaseVideo()


    # player.play()