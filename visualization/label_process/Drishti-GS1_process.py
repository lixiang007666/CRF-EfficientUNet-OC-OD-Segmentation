import cv2 as cv
import numpy as np

img = np.zeros([512, 512, 3], np.uint8)  ##创建一副黑色的图片


"""画圆"""
cv.circle(img, (447, 63), 63, (0, 0, 255), -1)


cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()