from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import glob
import skimage.io as io
import skimage.transform as trans
from PIL import Image
import cv2
import os,array
import pandas as pd
def adjustData(img,mask,flag_multi_class,num_class):   
    if(flag_multi_class):        
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)
def adjustDataforAuencoder(img,mask,flag_multi_class,num_class):   
    if(flag_multi_class):        
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255        
    return (img,mask)
def trainGenerator ( batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,image_size=160,seed = 1 ):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    target_size = (image_size,image_size)
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def ValidationGenerator(test_path,num_image = 109,image_size = 160,flag_multi_class = False,as_gray = True):
    target_size = (image_size,image_size)
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
def AutoEncoder_trainGenerator ( batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 1,save_to_dir = None,image_size=160,seed = 1 ):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    target_size = (image_size,image_size)
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
def Pretext_trainGenerator ( batch_size,train_path,image_folder,mask_folder,aug_dict_img,aug_dict_mask,image_color_mode = "rgb",
                    mask_color_mode = "rgb",image_save_prefix  = "image",mask_save_prefix  = "label",
                    flag_multi_class = False,num_class = 1,save_to_dir = None,image_size=160,seed = 1 ):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    target_size = (image_size,image_size)
    image_datagen = ImageDataGenerator(**aug_dict_img)
    mask_datagen = ImageDataGenerator(**aug_dict_mask)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustDataforAuencoder(img,mask,flag_multi_class,num_class)
        yield (img,mask)
def AutoEncoder_ValidationGenerator(test_path,num_image = 109,image_size = 160,flag_multi_class = False,as_gray = False):
    target_size = (image_size,image_size)
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

def testGenerator(test_path,num_image = 300,target_size = (384,384),flag_multi_class = False,as_gray = False):
    sourceFiles1 = os.listdir(test_path)
    for i in range(len(sourceFiles1)):
        img = io.imread(os.path.join(test_path,sourceFiles1[i]),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
def savePredictLabelToCVS(save_path,fileName,imagesize,testlen,results):
    columnNames = list()
    width = imagesize
    height = imagesize
    wxh=width * height
    columnNames = list()
    columnNames.append('id')
    for i in range(wxh):
        pixel = 'p'
        pixel += str(i)
        columnNames.append(pixel)
    result_data = pd.DataFrame(columns = columnNames)
    for i in range(testlen):
        data = []    
        data.append(i)
        rawData=np.reshape(results[i]*255, (width, height))
        for y in range(width):
            for x in range(height):
                data.append(rawData[x,y])                               
                k = 0
        result_data.loc[i] = [data[k] for k in range(wxh+1)]  
    
    result_data.to_csv(os.path.join(resultFolder,fileName),index = False)
def savePredictLabelToImg(save_path,imagesize,testlen,results):   
    width = imagesize
    height = imagesize
    wxh=width * height       
    for i in range(testlen):        
        rawData=np.reshape(results[i]*255, (width, height))
        im = Image.fromarray(rawData)
        im.save(os.path.join(save_path,str(i)+'.bmp'))
def saveResultToCSV(save_path,fileName,imagesize,testlen,results):
    import os,array
    import numpy 
    import pandas as pd
    columnNames = list()
    width = 384
    height = 384
    wxh=width * height
    columnNames = list()
    columnNames.append('val_acc')
    columnNames.append('val_dice')
    columnNames.append('val_iou')
    columnNames.append('val_recall')
    columnNames.append('val_precision')
    columnNames.append('val_f1')
    columnNames.append('val_specificity')
    evaluate_data = pd.DataFrame(columns = columnNames)
    data = []
    data.append(Colon_scores_val[1])   
    data.append(Colon_scores_val[2]) 
    data.append(Colon_scores_val[3]) 
    data.append(Colon_scores_val[4]) 
    data.append(Colon_scores_val[5]) 
    data.append(Colon_scores_val[6]) 
    data.append(Colon_scores_val[7]) 
    evaluate_data.loc[0] = [data[k] for k in range(7)] 
    valuate_data.to_csv(os.path.join(resultFolder,fileName),index = False)
def ConvertToArray(test_path,num_image = 300,target_size = (384,384),as_gray = True):
    sourceFiles1 = os.listdir(test_path)
    img_arr=np.zeros(num_image)
    for i in range(len(sourceFiles1)):
        img = io.imread(os.path.join(test_path,sourceFiles1[i]),as_gray = True)
        img = img / 255
        img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        img_arr.append(img)
    return img_arr