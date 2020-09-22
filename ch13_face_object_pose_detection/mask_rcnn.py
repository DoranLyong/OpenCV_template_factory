import sys 
import os.path as osp 

from tqdm import tqdm
import numpy as np 
import cv2 


def drawBox(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), colors[classId], 2)

    label = f'{classes[classId]}: {conf:.2f}'

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left - 1, top - labelSize[1] - baseLine),
                    (left + labelSize[0], top), colors[classId], -1)
    cv2.putText(img, label, (left, top - baseLine), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)



# model & configure 
model = osp.join('.','mask_rcnn','frozen_inference_graph.pb')
cfg = osp.join('.','mask_rcnn','mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
class_lbls = osp.join('.', 'mask_rcnn','coco_90.names') # class_labels
confThreshold = 0.6
maskThreshold = 0.3


# test data 
img_files = ['dog.jpg', 'traffic.jpg', 'sheep.jpg', 'kite.jpg','person.jpg']



# build network 
net = cv2.dnn.readNet(model, cfg)

if net.empty():
    print('Net open failed!')
    sys.exit()


# load the class file 

classes = []
with open(class_lbls, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

colors = np.random.uniform(0, 255, size=(len(classes), 3)) # random color 



# Run

for f in img_files:
    dataPath = osp.join('.', 'data', f)
    img = cv2.imread(dataPath)

    if img is None:
        continue

    # blob & inference 
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])     # boxes.shape=(1, 1, 100, 7)
                                                                                # masks.shape=(100, 90, 15, 15)
    
    h, w = img.shape[:2]
    numClasses = masks.shape[1]  # 90
    numDetections = boxes.shape[2]  # 100

    boxesToDraw = []
    for i in range(numDetections):
        box = boxes[0, 0, i]  # box.shape=(7,)
        mask = masks[i]  # mask.shape=(90, 15, 15)
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])
            #print(classId, classes[classId], score)

            x1 = int(w * box[3])
            y1 = int(h * box[4])
            x2 = int(w * box[5])
            y2 = int(h * box[6])

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            boxesToDraw.append([img, classId, score, x1, y1, x2, y2])
            classMask = mask[classId]

            # resize each 15x15 mask to the size of bbox in each object 
                    
            classMask = cv2.resize(classMask, (x2 - x1 + 1, y2 - y1 + 1))
            mask = (classMask > maskThreshold)

            # then, assign transparent colors.     
            roi = img[y1:y2+1, x1:x2+1][mask]
            img[y1:y2+1, x1:x2+1][mask] = (0.7 * colors[classId] + 0.3 * roi).astype(np.uint8)


    # draw bbox in each object & put class name 
    for box in tqdm(boxesToDraw):
        drawBox(*box)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()



