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
        lastcam = -1
        for line in lines:
            [img_name, angle] = line.strip().split(' ')
            frameid = img_name.strip().split('_')[2][5:]
            try:
                frameid = int(frameid)
            except:
                print 'filename parse error:', img_name, frameid
                continue

            filepathname = join(imgdir, img_name)
            camnum = img_name.strip().split('_')[0]

            # import ipdb; ipdb.set_trace()
            if (lastind<0 or frameid==lastind+frame_iter) and (camnum==lastcam or lastcam==-1):
                sequencelist.append((filepathname,angle))
                lastind = frameid
                lastcam = camnum
            else: # the index is not continuous
                if len(sequencelist)>=batch:
                    self.imgnamelist.append(sequencelist)
                    print '** sequence: ', len(sequencelist)
                    sequencelist = []
                else:
                    print 'sequence too short'
                lastind = -1
                lastcam = -1


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

def unlabelloss(labelseq):
    thresh = 0.005
    unlabel_batch = labelseq.shape[0]
    loss_unlabel = 0
    for ind1 in range(unlabel_batch-5): # try to make every sample contribute
        # randomly pick two other samples
        ind2 = random.randint(ind1+2, unlabel_batch-1) # big distance
        ind3 = random.randint(ind1+1, ind2-1) # small distance

        # target1 = Variable(x_encode[ind2,:].data, requires_grad=False).cuda()
        # target2 = Variable(x_encode[ind3,:].data, requires_grad=False).cuda()
        # diff_big = criterion(x_encode[ind1,:], target1) #(labelseq[ind1]-labelseq[ind2])*(labelseq[ind1]-labelseq[ind2])
        diff_big = (labelseq[ind1]-labelseq[ind2])*(labelseq[ind1]-labelseq[ind2])
        diff_big = diff_big.sum()/2.0
        # diff_small = criterion(x_encode[ind1,:], target2) #(labelseq[ind1]-labelseq[ind3])*(labelseq[ind1]-labelseq[ind3])
        diff_small = (labelseq[ind1]-labelseq[ind3])*(labelseq[ind1]-labelseq[ind3])
        diff_small = diff_small.sum()/2.0
        # import ipdb; ipdb.set_trace()
        cost = max(diff_small-thresh-diff_big, 0)
        # print diff_big, diff_small, cost
        loss_unlabel = loss_unlabel + cost
    print loss_unlabel


if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)

    # unlabelset = FolderUnlabelDataset(imgdir='/datadrive/person/DukeMTMC/heading',batch = 32, data_aug=True, include_all=True,datafile='duke_unlabeldata.pkl')
    unlabelset = DukeSeqLabelDataset(batch=24, data_aug=True)
    # unlabelset = FolderUnlabelDataset(imgdir='/datadrive/person/DukeMTMC/heading',batch = 24, data_aug=True, include_all=True)
    print len(unlabelset)
    # import ipdb; ipdb.set_trace()

    # for k in range(10):
    #     sample = unlabelset[k*1000]
    #     imgseq, labelseq = sample['imgseq'], sample['labelseq']
    #     print imgseq.dtype, imgseq.shape
    #     seq_show_with_arrow(imgseq,labelseq, scale=0.8)

    dataloader = DataLoader(unlabelset, batch_size=1, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
   
    while True:

        try:
            sample = dataiter.next()
        except:
            dataiter = iter(dataloader)
            sample = dataiter.next()

        imgseq, labelseq = sample['imgseq'].squeeze().numpy(), sample['labelseq'].squeeze().numpy()
        unlabelloss(labelseq)
        fakelabel = np.random.rand(24,2)
        unlabelloss(fakelabel)
        seq_show_with_arrow(imgseq, labelseq, scale=0.8)
