
import cv2
import numpy as np

#读取图像
image = cv2.imread('V0001.jpg')
#对图像进行灰度化处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("img0",gray)
#对灰度化图像进行标准化，该函数的参数依次是：输入数组，输出数组，最小值，最大值，标准化模式。
cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
cv2.imwrite("V0001_1.jpg",gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


