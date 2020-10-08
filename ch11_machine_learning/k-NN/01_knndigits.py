# code=<utf-8>
"""
k-NN 알고리즘을 확용한 필기체 인식 (성능이 좋지는 않다)

* data/digits.png  파일을 훈련 데이터셋으로 활용 
* 20x20 을 한 셀로 본다 → 각 레이블 별로 쪼개서 행렬 구조로 만든다 (5000, 400)
* k-nn 알고리즘으로 학습한다. 
* 마우스로 그린 숫자를 잘 분류하는지 확인한다. 
"""
import os 
import os.path as osp 
import sys 
sys.path.insert(0, osp.pardir)  # 현재 파일의 부모 디렉토리의 경로를 추가 

import numpy as np 
import cv2 



""" 1. 마우스로 숫자를 그릴 때 사용 """ 
oldx, oldy = -1, -1

def on_mouse(event, x, y, flags, _):
    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        oldx, oldy = -1, -1

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (255, 255, 255), 40, cv2.LINE_AA)
            oldx, oldy = x, y
            cv2.imshow('img', img)



""" 2. 학습 & 레이블 행렬 생성 """ 
imgPath = osp.join(osp.abspath(os.getcwd()), 'data','digits.png')

digits = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

if digits is None:
    print('Image load failed!')
    sys.exit()

h, w = digits.shape[:2]

cells = [np.hsplit(row, w//20) for row in np.vsplit(digits, h//20)]
cells = np.array(cells)
train_images = cells.reshape(-1, 400).astype(np.float32)
train_labels = np.repeat(np.arange(10), len(train_images)/10)

""" 3. k-NN 학습 """ 
knn = cv2.ml.KNearest_create()
knn.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)



""" 4. 사용자 입력 영상에 대한 예측 """ 
img = np.zeros((400, 400), np.uint8)

cv2.imshow('img', img)
cv2.setMouseCallback('img', on_mouse)

while True:
    key = cv2.waitKey()

    if key == 27:
        break
    elif key == ord(' '):
        test_image = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        test_image = test_image.reshape(-1, 400).astype(np.float32)

        ret, _, _, _ = knn.findNearest(test_image, 5)
        print(int(ret))

        img.fill(0)
        cv2.imshow('img', img)

cv2.destroyAllWindows()
