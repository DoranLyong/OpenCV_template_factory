import sys
import os.path as osp 

from tqdm import tqdm 
import numpy as np
import cv2


# model & configure 
model = osp.join('.','openpose', 'pose_iter_440000.caffemodel')
cfg = osp.join('.', 'openpose', 'pose_deploy_linevec.prototxt')

# num_pose_point, num_point_connection, num_point_pair
nparts = 18
npairs = 17
pose_pairs = [(1, 2), (2, 3), (3, 4),  # left arm 
              (1, 5), (5, 6), (6, 7),  # right arm 
              (1, 8), (8, 9), (9, 10),  # left leg 
              (1, 11), (11, 12), (12, 13),  # right leg 
              (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]  # face 

# test data 
img_files = ['pose1.jpg', 'pose2.jpg', 'pose3.jpg']

# build network 
net = cv2.dnn.readNet(model, cfg)

if net.empty():
    print('Net open failed!')
    sys.exit()

for f in img_files:
    dataPath = osp.join('.','data',f)
    img = cv2.imread(dataPath)

    if img is None:
        continue

    # blob & inference 
    blob = cv2.dnn.blobFromImage(img, 1/255., (368, 368))
    net.setInput(blob)
    out = net.forward()  # out.shape=(1, 57, 46, 46)

    h, w = img.shape[:2]

    # extract detected points
    points = []
    for i in range(nparts):
        heatMap = out[0, i, :, :]

        '''
        heatImg = cv2.normalize(heatMap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        heatImg = cv2.resize(heatImg, (w, h))
        heatImg = cv2.cvtColor(heatImg, cv2.COLOR_GRAY2BGR)
        heatImg = cv2.addWeighted(img, 0.5, heatImg, 0.5, 0)
        cv2.imshow('heatImg', heatImg)
        cv2.waitKey()
        '''

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int(w * point[0] / out.shape[3])
        y = int(h * point[1] / out.shape[2])

        points.append((x, y) if conf > 0.1 else None)  # heat map threshold=0.1

    # drawing results 
    for pair in pose_pairs:
        p1 = points[pair[0]]
        p2 = points[pair[1]]

        if p1 is None or p2 is None:
            continue

        cv2.line(img, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(img, p1, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p2, 4, (0, 0, 255), -1, cv2.LINE_AA)

    # inference time 
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey()

cv2.destroyAllWindows()
