import numpy as np
import os
from skimage import io
'''
该脚本主要实现语义分割中多类结果的评估功能
要求：预测结果文件夹和真值文件夹中各个图像的文件名应该一样，对同一种类像素的灰度表示也应该一样
'''

pre_path = 'Prediction'                         #预测结果的文件夹
gt_path = 'Label'                               #ground truth文件夹
img_size = (512, 512)                           #图像的尺寸（只需要长宽）
classes = np.array([0, 1, 2, 3]).astype('uint8')#每一类的灰度值表示
files = os.listdir(pre_path)

res = []
for clas in classes:

    D = np.zeros([len(files), img_size[0], img_size[1], 2]).astype(bool)#存储每一类的二值数据
    # print(D.shape)
    for i, file in enumerate(files):
        img1 = io.imread(os.path.join(pre_path, file), as_gray=True)#以灰度值的形式读取
        img2 = io.imread(os.path.join(gt_path, file), as_gray=True)#以灰度值的形式读取
        D[i, :, :, 0] = img1 == clas
        D[i, :, :, 1] = img2 == clas
    res.append(np.sum(D[..., 0] & D[..., 1])/np.sum(D[..., 0] | D[..., 1])) #计算IOU
    # print(res)
#结果输出
for i, clas in enumerate(classes):
    print("Class "+str(clas)+' :'+str(res[i]))


