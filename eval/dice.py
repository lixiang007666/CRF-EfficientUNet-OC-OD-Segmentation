import cv2
import os
import numpy as np

sum=0

for i in range(1, 5):
    s2 = cv2.imread("Label/" + str(i) + ".png", 0)  # 模板
    row, col = s2.shape[0], s2.shape[1]
    s1 = cv2.imread("Prediction/" + str(i) + ".png", 0)  # 读取配准后图像
    d = []
    s = []
    for r in range(row):
        for c in range(col):
            if s1[r][c] == s2[r][c]:  # 计算图像像素交集
                s.append(s1[r][c])
    m1 = np.linalg.norm(s)
    m2 = np.linalg.norm(s1.flatten()) + np.linalg.norm(s2.flatten())
    d.append(2*m1/m2)
    msg = "这是第{}张图的dice系数".format(i) + str(2 * m1 / m2)
    # print(2*m1/m2)
    print(msg)
# print(d)