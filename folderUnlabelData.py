# extend with UCF data

import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
import xml.etree.ElementTree
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show, im_hsv_augmentation, im_crop
import random
import matplotlib.pyplot as plt


class FolderUnlabelDataset(Dataset):

    def __init__(self, imgdir='/datasets/dirimg/',
                        imgsize = 192, batch = 32, 
                        data_aug=False,
                        mean=[0,0,0],std=[1,1,1]):

        self.imgsize = imgsize
        self.imgnamelist = []
        # self.fileprefix = 'drone_'
        self.batch = batch
        self.aug = data_aug
        self.mean = mean
        self.std = std
        self.episodeNum = []

        self.folderlist = listdir(imgdir)

        for f_ind, foldername in enumerate(self.folderlist):

            folderpath = join(imgdir, foldername)
            imglist = listdir(folderpath)
            imglist = sorted(imglist)

            sequencelist = []
            # missimg = 0
            lastind = -1
            for filename in imglist:
                if filename.split('.')[-1]!='jpg': # only process jpg file
                    continue
                fileind = filename.split('.')[0].split('_')[-1]
                try:
                    fileind = int(fileind)
                except:
                    print 'filename parse error:', filename, fileind
                    continue
                # filename = self.fileprefix+foldername+'_'+str(imgind)+'.jpg'
                filepathname = join(folderpath, filename)

                if lastind<0 or fileind==lastind+1:
                # if isfile(filepathname):
                    sequencelist.append(filepathname)
                    lastind = fileind
                    # if missimg>0:
                        # print '  -- last missimg', missimg
                    # missimg = 0
                else: # the index is not continuous
                    if len(sequencelist)>=batch:
                        # missimg = 1
                        self.imgnamelist.append(sequencelist)
                        # print 'image lost:', filename
                        print '** sequence: ', len(sequencelist)
                        # print sequencelist
                        sequencelist = []
                    lastind = -1
                    # else:
                        # missimg += 1
            if len(sequencelist)>=batch:          
                self.imgnamelist.append(sequencelist)
                print '** sequence: ', len(sequencelist)
                sequencelist = []


        sequencenum = len(self.imgnamelist)
        print 'Read', sequencenum, 'sequecnes...'
        print np.sum(np.array([len(imglist) for imglist in self.imgnamelist]))

        total_seq_num = 0
        for sequ in self.imgnamelist:
            total_seq_num += len(sequ) - batch + 1
            self.episodeNum.append(total_seq_num)
        self.N = total_seq_num
        # print total_seq_num
        # print self.episodeNum

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
        for k in range(self.batch):
            img = cv2.imread(self.imgnamelist[epiInd][idx+k])

            if self.aug:
                img = im_hsv_augmentation(img)
                img = im_crop(img)

            outimg = im_scale_norm_pad(img, outsize=self.imgsize, mean=self.mean, std=self.std, down_reso=True)

            imgseq.append(outimg)

        return np.array(imgseq)

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    folderUnlabelDataset = FolderUnlabelDataset(batch=24, data_aug=True, extend=True)
    for k in range(1):
        imgseq = folderUnlabelDataset[k*1000]
        print imgseq.dtype, imgseq.shape
        seq_show(imgseq, scale=0.8)
        # cv2.imshow('img',folderUnlabelDataset.img_denormalize(imgseq[5,:,:,:]))
        # cv2.waitKey(0)

    dataloader = DataLoader(folderUnlabelDataset, batch_size=1, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
   
    while True:


        try:
            sample = dataiter.next()
        except:
            dataiter = iter(dataloader)
            sample = dataiter.next()

        seq_show(sample.squeeze().numpy(), scale=0.8)
          
