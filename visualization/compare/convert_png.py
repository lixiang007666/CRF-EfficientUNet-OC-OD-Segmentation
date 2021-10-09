import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
import numpy as np
import cv2
from PIL import Image

file = '/Users/lixiang/PycharmProjects/visualization/trainannot/V0358.nii.gz'  # 你的nii或者nii.gz文件路径
img = nib.load(file)

width, height, queue = img.dataobj.shape

OrthoSlicer3D(img.dataobj).show()


img_arr = img.dataobj


print(img_arr)
print(np.unique(img_arr))

img_arr=np.array(img_arr)
img_arr[img_arr==1]=128
img_arr[img_arr==2]=255


data = img_arr
print(data)
data[:80,]=255#[行开始:行结束,列开始:列结束]
data[1600:,]=255
data[:,1600:]=255
data[:,:100]=255
cv2.imwrite('G0358.bmp',data)
img = Image.open('G0358.bmp')
img = img.transpose(Image.ROTATE_90)
img.save('G0358.bmp')

