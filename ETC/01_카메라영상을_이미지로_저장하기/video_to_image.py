"""
Code author: DoranLyong 
Reference : 
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
* https://076923.github.io/posts/Python-opencv-4/
* https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html
* https://m.blog.naver.com/samsjang/220500854338
"""
from pathlib import Path

import numpy as np 
import cv2 


# make a directory to save images
DATA_DIR = Path('data')
(DATA_DIR).mkdir(parents=True, exist_ok=True)


def main(): 
    capture = cv2.VideoCapture("ZebraTrim.mp4")


    while(True):
        # capture frame-by-frame 
        ret, frame = capture.read() 

        if not ret:  # if read the image in False 
            print("No frame") 
            break 

        print(f"Frame_num: {capture.get(cv2.CAP_PROP_POS_FRAMES)}/{capture.get(cv2.CAP_PROP_FRAME_COUNT)}")
        
        # Display the resulting frame 
        cv2.imshow('frame', frame)
        title = "frame_" + str(int(capture.get(cv2.CAP_PROP_POS_FRAMES))-1) + '.png'
        cv2.imwrite(f"{DATA_DIR}/{title}", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    # When everything done, release the capture
    capture.release()





if __name__ == "__main__":
    main() 

    cv2.destroyAllWindows()