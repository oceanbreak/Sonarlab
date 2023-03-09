from SonarImaging import VideoPlayer, eqHist
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob
import os

def estimateHistogram(fig, ax, img, show=(1,1,1,1), bins=64):
    ax.clear()
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if show[0]:
        ax.hist(b.reshape(-1), bins=bins, color='blue', histtype='step')
    if show[1]:
        ax.hist(g.reshape(-1), bins=bins, color='green', histtype='step')
    if show[2]:
        ax.hist(r.reshape(-1), bins=bins, color='red', histtype='step')
    if show[3]:
        ax.hist(gray.reshape(-1), bins=bins, color='black', histtype='step')
    fig.canvas.draw()
    renderer = fig.canvas.renderer
    ax.draw(renderer)




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
    index = 1
    PATH = "D:\\DATA"
    video_files = glob.glob(os.path.join(PATH, "*.mp4"))
    for video_file in video_files:

        VP = VideoPlayer()
        VP.openVideoFile(video_file)
        VP.setFrameStep(5)
        VP.setScaleFactor(4)
        VP.getNextFrame()

        haze_index_arr = []
        blur_index_arr = []

        # Histogram figure
        fig1,ax1 = plt.subplots(figsize=(6,3))
        plt.show(block=False)  

        while VP.playing:
            VP.playStepForwards()
            # VP.show()
            print(f'FRAME {VP.cur_frame_no} OF {int(VP.vid_frame_length)}')
            frame = VP.getCurrentFrame()[:-100, :, :]
            cv2.imshow('Raw frame', cv2.resize(frame, (640,480)))
            frame = eqHist(frame)

            # BLUR
            blur_index = variance_of_laplacian(frame)
            blur_index_arr.append(blur_index)

            # DCP
            frame = cv2.resize(frame, (640,480))
            estimateHistogram(fig1, ax1, frame, (1,1,1,0), 256)
            
            cv2.imshow('Equalized frame', frame)
            dcp = getMinChannel(frame)
            dcp = getDarkChannel(dcp, 55)
            # cv2.imshow('DCP', dcp)
            haze_index = hazeCoefficient(dcp, 13.7474657, 1.02377299, 3.63024834)
            haze_index_arr.append(haze_index)
            VP.waitKeyHandle(delay=10)

        fig,ax = plt.subplots(figsize=(12,3))
        temp = os.path.split(video_file)[-1]
        temp = '.'.join(temp.split('.')[:-1])
        output_name = f'{temp}_parameters.png'
        output_name = os.path.join(PATH, output_name)
        ax2 = ax.twinx() 
        ax.plot(haze_index_arr, color='royalblue', label='Haze')
        ax2.plot(blur_index_arr, color='lightseagreen', label='Focus')

        temp = '.'.join(temp.split('.')[:-1])
        ax.set_title(f'Parameters of {temp} video')
        ax.set_ylabel('Haze coefficient')
        ax2.set_ylabel('Focus coefficient')

        ax.tick_params(axis='y', labelcolor='royalblue')
        ax2.tick_params(axis='y', labelcolor='lightseagreen')

        ax.grid()

        plt.savefig(output_name, dpi=300)
        plt.show(block=False)

        plt.pause(5)
        plt.close('all')
        index += 1



