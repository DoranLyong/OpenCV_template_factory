# code=<utf-8>
"""
* svm 알고리즘 구현하기 
"""
import sys 

import numpy as np 
import cv2 




""" 1. 학습 데이터 & 레이블  """
trains = np.array([[150, 200], [200, 250],
                   [100, 250], [150, 300],
                   [350, 100], [400, 200],
                   [400, 300], [350, 400]], dtype=np.float32)
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])



""" 2. 모델 초기화 및 학습 """ 
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
#svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setKernel(cv2.ml.SVM_RBF)  # 뉴럴넷의 활성화 함수와 비슷한 역할 


# svm.trainAuto(trains, cv2.ml.ROW_SAMPLE, labels)  # 자동 훈련 기능을 빼고 생으로 학습시켜보자 

svm.setC(2.5)
svm.setGamma(0.00001)  # 감마 값을 바꿔가면서 hyperplane이 어떻게 변하는지 확인하자
svm.train(trains, cv2.ml.ROW_SAMPLE, labels)
print('C:', svm.getC())
print('Gamma:', svm.getGamma())



""" 3. 결과 시각화 """ 
w, h = 500, 500
img = np.zeros((h, w, 3), dtype=np.uint8)

for y in range(h):
    for x in range(w):
        test = np.array([[x, y]], dtype=np.float32)
        _, res = svm.predict(test)
        ret = int(res[0, 0])

        if ret == 0:
            img[y, x] = (128, 128, 255)  # Red (클래스 0)
        else:
            img[y, x] = (128, 255, 128)  # Green (클래스 1)

color = [(0, 0, 128), (0, 128, 0)]

for i in range(trains.shape[0]):
    x = int(trains[i, 0])
    y = int(trains[i, 1])
    l = labels[i]

    cv2.circle(img, (x, y), 5, color[l], -1, cv2.LINE_AA)

cv2.imshow('svm', img)
cv2.waitKey()
cv2.destroyAllWindows()