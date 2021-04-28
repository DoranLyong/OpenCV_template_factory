"""
Convert images to a video file. 

(ref) https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
"""

#%% 
from glob import glob
import os.path as osp 

import cv2
from tqdm import tqdm 
import numpy as np 
from natsort import natsorted  # (ref) https://pypi.org/project/natsort/


#%% video setting 
fourcc = cv2.VideoWriter_fourcc(*'MP4V')



# %% get image name list 
img_array = [] 
file_list = natsorted(glob(osp.join("inference_outputs", "*.png")))


#%% load images 
for filename in tqdm(file_list):
    img = cv2.imread(filename)

    height, width, channel = img.shape
    size = (width,height)
    img_array.append(img)


# %% initialize the video object 
out = cv2.VideoWriter('Zebra_counting.mp4',fourcc, 30, size)  # 30 fps 


##%% 
for idx, item in enumerate(tqdm(img_array)):
    out.write(img_array[idx])

out.release()