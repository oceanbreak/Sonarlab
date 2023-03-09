from SonarImaging import VideoPlayer, eqHist
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os

def estimateHistogram(img, show=(1,1,1,1), bins=64):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if show[0]:
        plt.hist(b.reshape(-1), bins=bins, color='blue', histtype='step')
    if show[1]:
        plt.hist(g.reshape(-1), bins=bins, color='green', histtype='step')
    if show[2]:
        plt.hist(r.reshape(-1), bins=bins, color='red', histtype='step')
    if show[3]:
        plt.hist(gray.reshape(-1), bins=bins, color='black', histtype='step')
    plt.show()

########################## HAZE ESTIMATION ##############################
def getMinChannel(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("bad image shape, input must be color image")
        return None
    return np.amin(img, 2).astype('uint8')

def getDarkChannel(img, blockSize):
    if len(img.shape) == 2:
        pass
    else:
        print("bad image shape, input image must be two demensions")
        return None
    # blockSizeÃ¦Â£â‚¬Ã¦Å¸Â¥
    if blockSize % 2 == 0 or blockSize < 3:
        print('blockSize is not odd or too small')
        return None   
    # Try with erode
    kernel = np.ones((35, 35), 'uint8')
    return cv2.erode(img, kernel, iterations=1)

def hazeCoefficient(dcp_img, alpha=1, beta=1, mu=0):
    if len(dcp_img.shape) > 2:
        print('1-channel image accepted only')
        return None
    fave = np.average(dcp_img.reshape(-1))
    return fave
    return 1/(1 + alpha * np.exp(-beta * (np.sqrt(fave) - mu)))


########################## BLUR ESTIMATION ##############################
def variance_of_laplacian(image):
    img = cv2.resize(image, (720,480))
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    lap = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap




if __name__ == "__main__":
    PATH = "D:\\DATA\\videomodule_different_conditions.mp4"
    VP = VideoPlayer()
    VP.openVideoFile(PATH)
    VP.setScaleFactor(4)
    VP.getNextFrame()

    haze_index_arr = []
    blur_index_arr = []

    while VP.playing:
        VP.playStepForwards()
        # VP.show()
        print(f'FRAME {VP.cur_frame_no} OF {int(VP.vid_frame_length)}')
        frame = VP.getCurrentFrame()[:-100, :, :]
        frame = eqHist(frame)

        # BLUR
        blur_index = variance_of_laplacian(frame)
        blur_index_arr.append(blur_index)

        # DCP
        frame = cv2.resize(frame, (500,500))
        cv2.imshow('Analyze_frame', frame)
        dcp = getMinChannel(frame)
        dcp = getDarkChannel(dcp, 55)
        # cv2.imshow('DCP', dcp)
        haze_index = hazeCoefficient(dcp, 13.7474657, 1.02377299, 3.63024834)
        haze_index_arr.append(haze_index)
        VP.waitKeyHandle(delay=10)

    fig,ax = plt.subplots()
    ax2 = ax.twinx() 
    ax.plot(haze_index_arr, color='royalblue', label='Haze')
    ax2.plot(blur_index_arr, color='lightseagreen', label='Focus')

    ax.set_title('Image parameters')
    ax.set_ylabel('Haze coefficient')
    ax2.set_ylabel('Focus coefficient')

    ax.tick_params(axis='y', labelcolor='royalblue')
    ax2.tick_params(axis='y', labelcolor='lightseagreen')

    ax.grid()
    plt.show()



