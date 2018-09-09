import cv2
import torch
from math import sqrt, sin, cos
import numpy as np
import random
# import torch.nn as nn

def loadPretrain(model, preTrainModel):
    preTrainDict = torch.load(preTrainModel)
    model_dict = model.state_dict()
    print 'preTrainDict:',preTrainDict.keys()
    print 'modelDict:',model_dict.keys()
    preTrainDict = {k:v for k,v in preTrainDict.items() if k in model_dict}
    for item in preTrainDict:
        print '  Load pretrained layer: ',item
    model_dict.update(preTrainDict)
    # for item in model_dict:
    #   print '  Model layer: ',item
    model.load_state_dict(model_dict)
    return model

def loadPretrain2(model, preTrainModel):
    preTrainDict = torch.load(preTrainModel)
    model_dict = model.state_dict()
    # print 'preTrainDict:',preTrainDict.keys()
    # print 'modelDict:',model_dict.keys()
    # update the keyname according to the last two words
    loadDict = {}
    for k,v in preTrainDict.items():
        keys = k.split('.')
        for k2,v2 in model_dict.items():
            keys2 = k2.split('.')
            if keys[-1]==keys2[-1] and (keys[-2]==keys2[-2] or 
                (keys[-2][1:]==keys2[-2][2:] and keys[-2][0]=='d' and keys2[-2][0:2]=='de')): # compansate for naming bug
                loadDict[k2]=v
                print '  Load pretrained layer: ',k2
                break

    model_dict.update(loadDict)
    # for item in model_dict:
    #   print '  Model layer: ',item
    model.load_state_dict(model_dict)
    return model


def getColor(x,y,maxx,maxy):
    y = y*maxx/maxy
    maxy = maxx # normalize two axis
    x1, y1, t = x, y, maxx
    r = np.clip(1-sqrt(float(x1*x1+y1*y1))/t,0,1)
    x1, y1 = maxx-x, y
    g = np.clip(1-sqrt(float(x1*x1+y1*y1))/t,0,1)
    x1, y1 = x, maxy-y
    b = np.clip(1-sqrt(float(x1*x1+y1*y1))/t,0,1)
    # x1, y1 = maxx-x, maxy-y
    # a = sqrt(float(x1*x1+y1*y1))/t
    a = 1
    return (r,g,b,a)

def img_normalize(img, mean=[0,0,0], std=[1,1,1]): # resnet: mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
    img = img[:,:,[2,1,0]] # bgr to rgb
    img = img.astype(np.float32)/255.0
    img = (img-np.array(mean))/np.array(std)
    img = img.transpose(2,0,1)
    return img

def img_denormalize(img, mean=[0,0,0], std=[1,1,1]): # used for visualization only
    # print img.shape
    img = img.transpose(1,2,0)
    img = img*np.array(std)+np.array(mean)
    img = img.clip(0,1) # network can output values out of range
    img = (img*255).astype(np.uint8)
    img = img[:,:,[2,1,0]]
    return img

def seq_show(imgseq, scale = 0.3):
    # input a numpy array: n x 3 x h x w
    imgnum = imgseq.shape[0]
    imgshow = []
    for k in range(imgnum):
        imgshow.append(img_denormalize(imgseq[k,:,:,:])) # n x h x w x 3
    imgshow = np.array(imgshow)
    imgshow = imgshow.transpose(1,0,2,3).reshape(imgseq.shape[2],-1,3) # h x (n x w) x 3
    imgshow = cv2.resize(imgshow,(0,0),fx=scale,fy=scale)
    cv2.imshow('img',imgshow)
    cv2.waitKey(0)

def put_arrow(img, dir, centerx=150, centery=96):
    # print type(img), img.dtype, img.shape
    img = img.copy()
    cv2.line(img, (centery-30,centerx), (centery+30,centerx), (0, 255, 0), 2)
    cv2.line(img, (centery,centerx-30), (centery,centerx+30), (0, 255, 0), 2)

    cv2.arrowedLine(img, (centery,centerx), (int(centery+40*dir[1]),int(centerx-40*dir[0])), (0, 0, 255), 4)

    return img

def seq_show_with_arrow(imgseq, dirseq, scale = 0.8, mean=[0,0,0], std=[1,1,1]):
    # imgseq: a numpy array: n x 3 x h x w
    # dirseq: a numpy array: n x 2
    imgnum = imgseq.shape[0]
    imgshow = []
    for k in range(imgnum):
        img = img_denormalize(imgseq[k,:,:,:], mean, std)
        img = put_arrow(img, dirseq[k,:])
        imgshow.append(img) # n x h x w x 3
    imgshow = np.array(imgshow)
    imgshow = imgshow.transpose(1,0,2,3).reshape(imgseq.shape[2],-1,3) # h x (n x w) x 3
    imgshow = cv2.resize(imgshow,(0,0),fx=scale,fy=scale)
    cv2.imshow('img',imgshow)
    cv2.waitKey(0)

def groupPlot(datax, datay, group=10):
    datax, datay = np.array(datax), np.array(datay)
    if len(datax)%group>0:
        datax = datax[0:len(datax)/group*group]
        datay = datay[0:len(datay)/group*group]
    datax, datay = datax.reshape((-1,group)), datay.reshape((-1,group))
    datax, datay = datax.mean(axis=1), datay.mean(axis=1)
    return (datax, datay)

# amigo add for data augmentation before normalization
def im_hsv_augmentation(image, Hscale = 10,Sscale = 60, Vscale = 60):
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # change HSV
    h = random.random()*2-1
    s = random.random()*2-1
    v = random.random()*2-1
    imageHSV[:,:,0] = np.clip(imageHSV[:,:,0]+Hscale*h,0,255)
    imageHSV[:,:,1] = np.clip(imageHSV[:,:,1]+Sscale*s,0,255)
    imageHSV[:,:,2] = np.clip(imageHSV[:,:,2]+Vscale*v,0,255)
    image = cv2.cvtColor(imageHSV,cv2.COLOR_HSV2BGR)
    return image

def im_crop(image, maxscale=0.2):
    imgshape = image.shape
    startx = int(random.random()*maxscale*imgshape[1])
    starty = int(random.random()*maxscale*imgshape[0])
    endx = int(imgshape[1]-random.random()*maxscale*imgshape[1])
    endy = int(imgshape[0]-random.random()*maxscale*imgshape[0])
    return image[starty:endy,startx:endx,:]

def im_scale_norm_pad(img, outsize=192, mean=[0,0,0], std=[1,1,1], down_reso=False, down_len=30):
    # downsample the image for data augmentation
    minlen = np.min(img.shape[0:2])
    down_len = random.randint(down_len,down_len*5)
    if down_reso and minlen>down_len:
        resize_scale = float(down_len)/minlen
        img = cv2.resize(img, (0,0), fx = resize_scale, fy = resize_scale)

    resize_scale = float(outsize)/np.max(img.shape)
    # if the image is too 
    miniscale = 1.8
    x_scale, y_scale = resize_scale, resize_scale
    if img.shape[0] * resize_scale < outsize/miniscale:
        y_scale = outsize/miniscale/img.shape[0]
    if img.shape[1] * resize_scale < outsize/miniscale:
        x_scale = outsize/miniscale/img.shape[1]
   
    img = cv2.resize(img, (0,0), fx = x_scale, fy = y_scale)
    img = img_normalize(img, mean=mean, std=std)
    # print img.shape
    imgw = img.shape[2]
    imgh = img.shape[1]
    startx = (outsize-imgw)/2
    starty = (outsize-imgh)/2
    # print startx, starty
    outimg = np.zeros((3,outsize,outsize), dtype=np.float32)
    outimg[:, starty:starty+imgh, startx:startx+imgw] = img

    return outimg
