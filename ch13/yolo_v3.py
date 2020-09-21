import sys 
import os.path as osp 


from tqdm import tqdm 
import numpy as np 
import cv2 

# model & configure 
model = osp.join('.', 'yolo_v3', 'yolov3.weights')
cfg   = osp.join('.', 'yolo_v3', 'yolov3.cfg')
class_labels = osp.join('.', 'yolo_v3', 'coco.names')

confThreshold = 0.5   # confidence 
nmsThreshold = 0.4 


# test images 
img_files = ['dog.jpg', 'person.jpg', 'sheep.jpg', 'kite.jpg', 'traffic.jpg']


# Load network 
net = cv2.dnn.readNet(model, cfg)

if net.empty():
    print('Net open failed!')
    sys.exit()


# Load classes
classes = [] 
with open(class_labels, 'rt') as f: 
    classes = f.read().rstrip('\n').split('\n')
    
colors = np.random.uniform(0, 255, size=(len(classes), 3)) # random bbox color 


# Get output_layer names 
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] -1] for i in net.getUnconnectedOutLayers()]   # ['yolo_82, 'yolo_94', 'yolo_106']


# Run 
for f in img_files:
    dataPath = osp.join('.', 'data', f)
    img = cv2.imread(dataPath)

    if img is None: 
        continue 

    # Blob & Inference 
    blob = cv2.dnn.blobFromImage(img, 1/255. , (416, 416), swapRB=True)
    net.setInput(blob) # network input 
    outs = net.forward(output_layers)  # network output 

    h, w = img.shape[:2]

    class_ids = []
    confidences = [] 
    bboxes = [] 

    # get bboxes
    for out in outs:
        for detection in out: 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confThreshold:
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)


                # most-left 
                sx = int(cx - bw / 2)
                sy = int(cy - bh / 2)

                bboxes.append([sx, sy, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # NMS 
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, confThreshold, nmsThreshold)

    for i in tqdm(indices) :
        i = i[0]
        sx, sy, bw, bh = bboxes[i]
        
        label = f'{classes[class_ids[i]] }: {confidences[i]: .2}'
        color = colors[class_ids[i]]
        cv2.rectangle(img, (sx, sy, bw, bh), color, 2)
        cv2.putText(img, label, (sx, sy -10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA,
                    )
    
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 1, cv2.LINE_AA,
                )

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

        

