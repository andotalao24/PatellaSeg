import os
import math
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


ImgHeight = 256
ImgWidth =256

def removeBlackImg(imglist):
    #delete the img with no edge from the list and return the new list 
    #imglist is a numpy array
    #return the idx of start and end where imgs are not purely black, both included
    start=0
    end=-1
    ret=[]
    for i in range(imglist.shape[0]):
        img=np.squeeze(imglist[i])
        if not noEdge(img):
            if start==0:
                start=i
            end+=1
            ret.append(img)
    end+=start
    ret=np.array(ret)
    return ret,start,end
    
def removeBlackImg_list(imglist):
    new_list=[]
    info_list=[]#store start and end idx
    for i in range(len(imglist)):
        tmp,start,end=removeBlackImg(imglist[i])
        new_list.append(tmp)
        info_list.append([start,end])
    return new_list,info_list

def toBinary(img,threshold):
    img[img<threshold]=0
    img[img>=threshold]=1
    return img
                
def preprocess_img(image,mask=False):
    
    if mask:
        image = tf.image.decode_jpeg(image,channels=1) 
        
    else:
        image = tf.image.decode_png(image,channels=3) 
    image = tf.image.resize(image,[ImgHeight,ImgWidth]) 
    image /= 255.0
    if mask:
        image=tf.cast(image>0.5,tf.float32)
    return image
 
def load_and_preprocess_image(path):
    image = tf.io.read_file(path) 
    return preprocess_img(image)

def load_and_preprocess_mask(path):
    image = tf.io.read_file(path) 
    return preprocess_img(image,True)

def load_into_df_3d(DataPath):
    def cmpwhich_mask(elem):
        idx=elem.index('_')
        num=int(elem[:idx])
        return num
    def cmpwhich(elem):
        num=int(elem[-8:-4 ])
        return num

    images = []
    masks = []

    for dirname, _, filenames in os.walk(DataPath):
        tmpimg=[]
        tmpmask=[]
        for filename in filenames:# in one folder
            if 'mask'in filename :
                tmpmask.append(filename)
                img_name=name_gen(filename[:-9])
                tmpimg.append(img_name)
        tmpmask.sort(key=cmpwhich_mask)
        tmpimg.sort(key=cmpwhich)
        for i in range(len(tmpmask)):
            tmpimg[i]=(os.path.join(DataPath,dirname)+'\\'+tmpimg[i])
            tmpmask[i]=(os.path.join(DataPath,dirname)+'\\'+tmpmask[i])
        if tmpmask==[]:
            continue
        images.append(tmpimg)
        masks.append(tmpmask)
    return pd.DataFrame({'images': images, 'masks': masks})

def load_into_df_3d_pred(DataPath):#add pred 
    def cmpwhich_mask(elem):
        idx=elem.index('_')
        num=int(elem[:idx])
        return num
    def cmpwhich(elem):
        num=int(elem[-8:-4 ])
        return num

    images = []
    masks = []
    preds=[]
    for dirname, _, filenames in os.walk(DataPath):
        tmpimg=[]
        tmpmask=[]
        tmpred=[]
        for filename in filenames:# in one folder
            if 'mask'in filename :
                tmpmask.append(filename)
                tmpred.append(filename.replace('mask', 'pred'))
                img_name=name_gen(filename[:-9])
                tmpimg.append(img_name)
        tmpmask.sort(key=cmpwhich_mask)
        tmpred.sort(key=cmpwhich_mask)
        tmpimg.sort(key=cmpwhich)
        for i in range(len(tmpmask)):
            tmpimg[i]=(os.path.join(DataPath,dirname)+'\\'+tmpimg[i])
            tmpmask[i]=(os.path.join(DataPath,dirname)+'\\'+tmpmask[i])
            tmpred[i]=(os.path.join(DataPath,dirname)+'\\'+tmpred[i])
        if tmpmask==[]:
            continue
        images.append(tmpimg)
        masks.append(tmpmask)
        preds.append(tmpred)
    return pd.DataFrame({'images': images, 'masks': masks,'preds':preds})


def create_ds(dataframe,dataframeTest,BATCH_SIZE=32,BUFFER_SIZE = 1000):

    pathds=tf.data.Dataset.from_tensor_slices(dataframe['images'])
    imgds=pathds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    pathds=tf.data.Dataset.from_tensor_slices(dataframe['masks'])
    mskds=pathds.map(load_and_preprocess_mask, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    trainds = tf.data.Dataset.zip((imgds,mskds))
    trainds = trainds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    trainds = trainds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    pathds=tf.data.Dataset.from_tensor_slices(dataframeTest['images'])
    imgds=pathds.map(load_and_preprocess_image)
    pathds=tf.data.Dataset.from_tensor_slices(dataframeTest['masks'])
    mskds=pathds.map(load_and_preprocess_mask)
    testds = tf.data.Dataset.zip((imgds,mskds))
    testds = testds.batch(BATCH_SIZE)
    
    return trainds,testds
    

def create_ds_2d(datapath):
    images = []
    masks = []
    for dirname, _, filenames in os.walk(datapath):
        for filename in filenames:
            if 'mask' in filename:
                file_name = os.path.join(datapath, dirname, filename)
                masks.append(file_name)
                images.append(file_name.replace('_mask', ''))
    df=pd.DataFrame({'images':images,'masks':masks})
    return df

#[path1,path2,...path3,...]
def readImg(pathlist,h,w,mask=False):
    imglist=[]
    for path in pathlist:
        img=cv2.imread(path)
        img=cv2.resize(img,(h,w))
        if mask:
            img=img[:,:,0] 
            img=img.reshape(h,w,1)
        img=img.astype(np.float32)
        img/=255.0
        imglist.append(img)
    return imglist

def store_pred_2d(model,imglist,dataframe):
    #print(np.array(imglist).shape)
    pred_mask=model.predict(np.array(imglist))
    count=0
    name='_pred'
    for img in pred_mask:
        filename=dataframe['masks'][count].replace('mask','pred') # eg 1_pred.jpg
        img=cv2.resize(img,(256,256))
        count+=1
        img*=255.0
        img=img.astype('uint8')
        plt.imsave(filename,np.squeeze(img),cmap='gray')

def create_ds_3d(datapath,depth):
    # return a dataframe
    # absolute path
    # pred and mask
    def cmpwhich(elem):
        idx=elem.index('_',-15)
        num=int(elem[idx+1:-4])
        return num
    def cmpwhich_msk(elem): 
        idx=elem.index('_',-15)
        #print(elem[idx+1:-9]+' '+elem)
        num=int(elem[idx+1:-9])# hard coding 
        return num

    images = []
    masks = []
    for dirname, _, filenames in os.walk(datapath):
        tmpimg=[]
        tmpmask=[]
        for filename in filenames:# in one folder
            if 'mask'in filename :
                file_name=os.path.join(datapath,dirname,filename)
                
                # if allBlack(file_name)and tf.random.uniform(()) > 0.7:
                    # continue
                tmpmask.append(file_name)
                tmpimg.append(file_name.replace('mask', ''))

        tmpmask.sort(key=cmpwhich_msk)
        tmpimg.sort(key=cmpwhich)
            
        for i in range(math.ceil(len(tmpimg)/depth)):
            s=i*depth
            e=s+depth
            if e>len(tmpimg):
                images.append(tmpimg[-depth:])
                masks.append(tmpmask[-depth:])
            else:
                images.append(tmpimg[s:e])
                masks.append(tmpmask[s:e])  
    df=pd.DataFrame({'images':images,'masks':masks})
    return df


def create_ds_3d_pred(datapath,depth,pred):
    # return a dataframe
    # absolute path
    # pred and mask

    def cmpwhich(elem): 
        print(elem)
        idx=elem.index('_',-15)
        #print(elem[idx+1:-9]+' '+elem)
        num=int(elem[idx+1:-9])# hard coding 
        return num

    images = []
    masks = []
    for dirname, _, filenames in os.walk(datapath):
        tmpimg=[]
        tmpmask=[]
        for filename in filenames:# in one folder
            if 'mask'in filename :
                file_name=os.path.join(datapath,dirname,filename)
                tmpmask.append(file_name)
                tmpimg.append(file_name.replace('mask', pred))

        tmpmask.sort(key=cmpwhich)
        tmpimg.sort(key=cmpwhich)
            
        for i in range(math.ceil(len(tmpimg)/depth)):
            s=i*depth
            e=s+depth
            if e>len(tmpimg):
                images.append(tmpimg[-depth:])
                masks.append(tmpmask[-depth:])
            else:
                images.append(tmpimg[s:e])
                masks.append(tmpmask[s:e])  
    df=pd.DataFrame({'images':images,'masks':masks})
    return df

def create_dataset(imglist,masklist,BUFFER_SIZE,BATCH_SIZE,test=False):
    imgds=tf.data.Dataset.from_tensor_slices(imglist)
    mskds=tf.data.Dataset.from_tensor_slices(masklist)
    ds=tf.data.Dataset.zip((imgds,mskds))

    if test:
        ds=ds.batch(BATCH_SIZE)
    else:
        ds = ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds



def allBlack(filename):
    img=cv2.imread(filename)
    if np.all((img == 0)):
        return True
    return False

def noEdge(img):
    edges=cv2.Canny(np.uint8((img)),0,0)
    indices = np.where(edges != [0])
    yarr=np.array(indices[0])
    if yarr.shape[0]==0:
        return True
    return False

def showImg(img):
    plt.imshow(img)
    plt.show()

def toBinary(img,threshold):
    img[img<threshold]=0
    img[img>=threshold]=1
    return img

def name_gen(idx):
    l1=len(idx)
    lt=4-l1
    zero=''
    for i in range(lt):
        zero+='0'
    return 'axial'+zero+idx+'.png'