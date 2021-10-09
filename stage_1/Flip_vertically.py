# 使用OpenCV翻转图像（水平、垂直、水平垂直）
# USAGE
# python opencv_flip.py

# 导入必要的包
import argparse
import cv2


# 加载原始图像并展示
image = cv2.imread('/Users/lixiang/Desktop/work/RetinaImageGDOP-master/prediction/G0358.bmp')
cv2.imshow("Original", image)
# print("[INFO] flipping image horizontally...")

# # --image 输入图像
# # --flipCode 1沿y轴水平翻转，0沿x轴垂直翻转，-1水平垂直翻转
# # 水平翻转图像
# flipped = cv2.flip(image, 1)
# cv2.imshow("Flipped Horizontally", flipped)

# 垂直翻转图像
flipped = cv2.flip(image, 0)
print("[INFO] flipping image vertically...")
cv2.imwrite('/Users/lixiang/Desktop/work/RetinaImageGDOP-master/prediction/G0358.bmp',flipped)

# # 水平垂直翻转
# flipped = cv2.flip(image, -1)
# print("[INFO] flipping image horizontally and vertically...")
# cv2.imshow("Flipped Horizontally & Vertically", flipped)
# cv2.waitKey(0)

cv2.destroyAllWindows()
