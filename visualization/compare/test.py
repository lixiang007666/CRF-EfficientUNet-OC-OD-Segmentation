# 将分割图和原图合在一起
from PIL import Image
import matplotlib.pyplot as plt

# image1 原图
# image2 分割图
image1 = Image.open("V0001.jpg")
image2 = Image.open("V0001.bmp")

plt.figure()


plt.imshow(image1)


plt.imshow(image2)


plt.imshow(image1)
plt.imshow(image2, alpha=0.5)

plt.show()
