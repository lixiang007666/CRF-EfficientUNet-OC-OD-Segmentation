#导入opencv
import cv2
#读入原始图像，使用cv2.IMREAD_UNCHANGED
img = cv2.imread("V0001.jpg",cv2.IMREAD_UNCHANGED)# 读入要处理的图片，参数1为图片路径
#查看打印图像的shape
shape = img.shape
print(shape)
#判断通道数是否为3通道或4通道
if shape[2] == 3 or shape[2] == 4 :
    #将彩色图转化为单通道图
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray_image",img_gray)
cv2.imwrite("V0001.jpg",img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
