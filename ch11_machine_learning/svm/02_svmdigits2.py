# code=<utf-8>
"""
- 01_svmdigits.py 에서 학습된 내용을 저장한 'svmdigits.yml' 파일을 불러옴
- 불러온 파라미터로 SVM을 학습 없이 바로 사용하기 
"""
import sys
import os 
import os.path as osp 
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


""" 3. HOG 특징 추출 """ 
h, w = digits.shape[:2]
hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)
print('Descriptor Size:', hog.getDescriptorSize())

cells = [np.hsplit(row, w//20) for row in np.vsplit(digits, h//20)]
cells = np.array(cells)
cells = cells.reshape(-1, 20, 20)  # shape=(5000, 20, 20)

desc = []
for img in cells:
    desc.append(hog.compute(img))

train_desc = np.array(desc)
train_desc = train_desc.squeeze().astype(np.float32)
train_labels = np.repeat(np.arange(10), len(train_desc)/10)


""" 4. 학습된 SVM 모델 불러오기 """

svm = cv2.ml.SVM_load('svmdigits.yml')

if svm.empty():
    print('SVM load failed!')
    sys.exit()


""" 5. 사용자 입력 영상에 대한 예측 """ 

img = np.zeros((400, 400), np.uint8)

cv2.imshow('img', img)
cv2.setMouseCallback('img', on_mouse)

while True:
    key = cv2.waitKey()

    if key == 27:
        break
    elif key == ord(' '):
        test_image = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        test_desc = hog.compute(test_image).T

        _, res = svm.predict(test_desc)
        print(int(res[0, 0]))

        img.fill(0)
        cv2.imshow('img', img)

cv2.destroyAllWindows()
