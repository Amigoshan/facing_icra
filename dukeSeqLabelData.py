# for dukeMCMT dataset
# return a sequence of data with label

import cv2
import numpy as np

from os.path import isfile, join, isdir, split
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show_with_arrow, im_hsv_augmentation, im_crop
import random
import matplotlib.pyplot as plt

import pickle

class DukeSeqLabelDataset(Dataset):

    def __init__(self, labelfile='/datadrive/person/DukeMTMC/heading_gt.txt',
                        imgsize = 192, batch = 32, 
                        data_aug=False, 
                        mean=[0,0,0],std=[1,1,1]):

        self.imgsize = imgsize
        self.imgnamelist = []
        self.batch = batch
        self.aug = data_aug
        self.mean = mean
        self.std = std
        self.episodeNum = []

        frame_iter = 6

        sequencelist = []
        imgdir = split(labelfile)[0]
        imgdir = join(imgdir,'heading') # a subdirectory containing the images
        with open(labelfile,'r') as f:
            lines = f.readlines()

        lastind = -1
        for line in lines:
            [img_name, angle] = line.strip().split(' ')
            frameid = img_name.strip().split('_')[2][5:]
            try:
                frameid = int(frameid)
            except:
                print 'filename parse error:', img_name, frameid
                continue

            filepathname = join(imgdir, img_name)

            if lastind<0 or frameid==lastind+frame_iter:
                sequencelist.append((filepathname,angle))
                lastind = frameid
            else: # the index is not continuous
                if len(sequencelist)>=batch:
                    self.imgnamelist.append(sequencelist)
                    print '** sequence: ', len(sequencelist)
                    sequencelist = []
                else:
                    print 'sequence too short'
                lastind = -1


        sequencenum = len(self.imgnamelist)
        print 'Read', sequencenum, 'sequecnes...'
        print 'Including images ', np.sum(np.array([len(imglist) for imglist in self.imgnamelist]))

        total_seq_num = 0
        for sequ in self.imgnamelist:
            total_seq_num += len(sequ) - batch + 1
            self.episodeNum.append(total_seq_num)
        self.N = total_seq_num


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        epiInd=0 # calculate the epiInd
        while idx>=self.episodeNum[epiInd]:
            # print self.episodeNum[epiInd],
            epiInd += 1 
        if epiInd>0:
            idx -= self.episodeNum[epiInd-1]

        # print epiInd, idx
        imgseq = []
        labelseq = []
        for k in range(self.batch):
            img = cv2.imread(self.imgnamelist[epiInd][idx+k][0])
            angle = self.imgnamelist[epiInd][idx+k][1]
            direction_angle_cos = np.cos(float(angle))
            direction_angle_sin = np.sin(float(angle))
            label = np.array([direction_angle_sin, direction_angle_cos], dtype=np.float32)

            if self.aug:
                img = im_hsv_augmentation(img)
                img = im_crop(img)

            outimg = im_scale_norm_pad(img, outsize=self.imgsize, mean=self.mean, std=self.std, down_reso=True)

            imgseq.append(outimg)
            labelseq.append(label)

        return {'imgseq': np.array(imgseq), 'labelseq':np.array(labelseq)}

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)

    # unlabelset = FolderUnlabelDataset(imgdir='/datadrive/person/DukeMTMC/heading',batch = 32, data_aug=True, include_all=True,datafile='duke_unlabeldata.pkl')
    unlabelset = DukeSeqLabelDataset(batch=24, data_aug=True)
    # unlabelset = FolderUnlabelDataset(imgdir='/datadrive/person/DukeMTMC/heading',batch = 24, data_aug=True, include_all=True)
    print len(unlabelset)
    import ipdb; ipdb.set_trace()

    for k in range(10):
        sample = unlabelset[k*1000]
        imgseq, labelseq = sample['imgseq'], sample['labelseq']
        print imgseq.dtype, imgseq.shape
        seq_show_with_arrow(imgseq,labelseq, scale=0.8)

    dataloader = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
   
    while True:

        try:
            sample = dataiter.next()
        except:
            dataiter = iter(dataloader)
            sample = dataiter.next()

        imgseq, labelseq = sample['imgseq'], sample['labelseq']
        seq_show_with_arrow(imgseq.squeeze().numpy(), labelseq.squeeze().numpy(), scale=0.8)
          
