import sys 
import os.path as osp 

from tqdm import tqdm 
import numpy as np 
import cv2

# model & configure 
models = {'caffe':'res10_300x300_ssd_iter_140000_fp16.caffemodel', 'tf1':'opencv_face_detector_uint8.pb' }
cfgs   = {'caffe':'deploy.prototxt', 'tf1':'opencv_face_detector.pbtxt'}


model = osp.join('.','opencv_face_detector', models['caffe'] )
config = osp.join('.','opencv_face_detector', cfgs['caffe'])


net = cv2.dnn.readNet(model, config)

if net.empty():
    print("Net open failed!")
    sys.exit()


dataPath = osp.join('.','data','faces.jpg')
img = cv2.imread(dataPath, cv2.IMREAD_COLOR)
blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123))


net.setInput(blob)  # network input 
out = net.forward() # network output 
print("output shape: ", out.shape)


detect = out[0, 0, :, :]
print("detection_matrix shape:", detect.shape)
print("detection_matrix value: ",detect[0,:])
(h, w) = img.shape[:2]


# Rescaling to bbox 
for i in tqdm(range(detect.shape[0])):
    confidence = detect[i, 2]

    if confidence < 0.5: # threshold 
        break 

    x1 = int(detect[i, 3] * w)
    y1 = int(detect[i, 4] * h)
    x2 = int(detect[i, 5] * w)
    y2 = int(detect[i, 6] * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
    label = f'Face: {confidence: 4.2f}'
    cv2.putText(img, label, (x1, y1-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 1, cv2.LINE_AA
                )


cv2.imshow('faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()