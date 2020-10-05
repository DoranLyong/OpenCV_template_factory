"""
Code author: DoranLyong 

Reference : 
* Camera subthread / https://webnautes.tistory.com/1382
"""

import argparse
from queue import Queue
from _thread import *

import numpy as np 
import cv2 


# _Set queue 
enclosure_queue = Queue() 

# _Start: video setting 
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
record = False
# _End: video setting 



# _WebCam process 
def WebCam(queue):

    global fourcc, record
    
    print("******  Camera Loading...  ******", end="\n ")
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
        

    try: 
        while True:
            ret, frame = cap.read() 

            if not ret:
                break

            cv2.namedWindow("WebCam_view", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("WebCam_view", frame)
            key = cv2.waitKey(1)

            
            

            """
            Start: video capture signal
            """
            if key == 27 : # 'ESC'
                break

            elif key == ord('s'): # press 's' 
                print("Capturing for image...")
                cv2.imwrite( "%s.png" %("rgb_image"), frame)

            elif key == ord('v'): # press 'v'
                print("Recording start...")
                record = True 
                
                video_rgb = cv2.VideoWriter("%s.mp4" %("rgb_video"), fourcc, 30.0, (frame.shape[1], frame.shape[0]), 1)

            elif key == 32: # press 'SPACE' 
                print("Recording stop...")
                record = False 
                video_rgb.release()

            if record == True: 
                print("Video recording...")
                video_rgb.write(frame)            



    finally: 
        cv2.destroyAllWindows()

        # _Stop streaming
        cap.release()





if __name__ == "__main__":
    
    # _Webcam process is loaded onto subthread
    start_new_thread(WebCam, (enclosure_queue,)) 


    try:
        while(1):
            pass 

    except KeyboardInterrupt as e:
        print("******  Server closed ****** ", end="\n \n" ) 


    

