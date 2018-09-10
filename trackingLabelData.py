import cv2
import numpy as np

from os.path import isfile, join, isdir, split
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, seq_show, im_crop, im_hsv_augmentation, put_arrow, seq_show_with_arrow
import pandas as pd

import matplotlib.pyplot as plt

class TrackingLabelDataset(Dataset):

    def __init__(self, filename='/datadrive/data/aayush/combined_data2/train/annotations/person_annotations.csv',
                        imgsize = 192, data_aug = False, maxscale=0.1,
                        mean=[0,0,0],std=[1,1,1]):

        self.imgsize = imgsize
        self.aug = data_aug
        self.mean = mean
        self.std = std
        self.filename = filename
        self.maxscale = maxscale

        self.items = []
        if filename[-3:]=='csv':
            self.items = pd.read_csv(filename)
        else: # text file used by DukeMTMC dataset
            imgdir = split(filename)[0]
            # imgdir = join(imgdir,'heading') # a subdirectory containing the images
            with open(filename,'r') as f:
                lines = f.readlines()
            for line in lines:
                [img_name, angle] = line.strip().split(' ')
                self.items.append({'path':join(imgdir,img_name), 'direction_angle':angle})

        print 'Read images',len(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.filename[-3:]=='csv':
            point_info = self.items.iloc[idx]
        else:
            point_info = self.items[idx]
        #print(point_info)
        img_name = point_info['path']
        direction_angle = point_info['direction_angle']

        direction_angle_cos = np.cos(float(direction_angle))
        direction_angle_sin = np.sin(float(direction_angle))
        label = np.array([direction_angle_sin, direction_angle_cos], dtype=np.float32)
        img = cv2.imread(img_name)

        if img is None:
            print 'error image:', img_name
            return
        if self.aug:
            img = im_hsv_augmentation(img, Hscale = 10,Sscale = 60, Vscale = 60)
            img = im_crop(img, maxscale=self.maxscale)

        outimg = im_scale_norm_pad(img, outsize=self.imgsize, mean=self.mean, std=self.std, down_reso=True)

        return {'img':outimg, 'label':label}

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    # trackingLabelDataset = TrackingLabelDataset(data_aug = True, maxscale=0.1)
    # trackingLabelDataset = TrackingLabelDataset(filename='/datadrive/data/aayush/combined_data2/train/annotations/car_annotations.csv')
    trackingLabelDataset = TrackingLabelDataset(filename='/datadrive/person/DukeMTMC/trainval_duke.txt', data_aug=True)
    # print len(trackingLabelDataset)
    # for k in range(1000):
    #     img = trackingLabelDataset[k*10]['img']
    #     label = trackingLabelDataset[k*10]['label']
    #     print img.dtype, label
    #     print np.max(img), np.min(img), np.mean(img)
    #     print img.shape
    #     img = img_denormalize(img)
    #     img = put_arrow(img, label)
    #     cv2.imshow('img',img)
    #     cv2.waitKey(0)

    dataloader = DataLoader(trackingLabelDataset, batch_size=16, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)

    # import ipdb;ipdb.set_trace()

    for sample in dataloader:
      print sample['label'], sample['img'].size()
      seq_show_with_arrow(sample['img'].numpy(), sample['label'].numpy(), scale = 0.5)

