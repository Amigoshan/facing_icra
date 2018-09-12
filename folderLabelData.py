import cv2
import numpy as np

from os.path import isfile, join, isdir
from os import listdir
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show, im_crop, im_hsv_augmentation, put_arrow

import random

class FolderLabelDataset(Dataset):

    def __init__(self, imgdir='/home/wenshan/headingdata/label',
                        imgsize = 192, data_aug = False, maxscale=0.1,
                        mean=[0,0,0],std=[1,1,1]):

        self.imgsize = imgsize
        self.imgnamelist = []
        self.labellist = []
        # self.dir2ind = {'n': 0,'ne': 1,'e': 2, 'se': 3,'s': 4,'sw': 5,'w': 6,'nw': 7}
        self.dir2val = {'n':  [1., 0.],
                        'ne': [0.707, 0.707],
                        'e':  [0., 1.],
                        'se': [-0.707, 0.707],
                        's':  [-1., 0.],
                        'sw': [-0.707, -0.707],
                        'w':  [0., -1.],
                        'nw': [0.707, -0.707]}
        self.aug = data_aug
        self.mean = mean
        self.std = std
        self.maxscale = maxscale

        imgind = 0
        for clsfolder in listdir(imgdir):
            
            clsval = self.dir2val[clsfolder]

            clsfolderpath = join(imgdir, clsfolder)

            for imgname in listdir(clsfolderpath):
                if imgname[-3:] == 'jpg' or imgname[-3:] == 'png':
                    self.imgnamelist.append(join(clsfolderpath, imgname))
                    self.labellist.append(clsval)

        self.N = len(self.imgnamelist)
        print 'Read', self.N, 'images...'

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = cv2.imread(self.imgnamelist[idx]) # in bgr
        label = np.array(self.labellist[idx], dtype=np.float32)

        # random fliping
        flipping = False
        if self.aug and random.random()>0.5:
            flipping = True
            label[1] = -label[1]
        if self.aug:
            img = im_hsv_augmentation(img)
            img = im_crop(img, maxscale=self.maxscale)

        outimg = im_scale_norm_pad(img, outsize=self.imgsize, mean=self.mean, std=self.std, down_reso=True, flip=flipping)

        return {'img':outimg, 'label':label}

if __name__=='__main__':
    # test 
    np.set_printoptions(precision=4)
    import ipdb;ipdb.set_trace()
    facingDroneLabelDataset = FolderLabelDataset(imgdir='/datadrive/3DPES/facing_labeled', data_aug=True)
    for k in range(100):
        sample = facingDroneLabelDataset[k*100]
        img = sample['img']
        label = sample['label']
        # print img.dtype, label
        # print np.max(img), np.min(img), np.mean(img)
        # print img.shape
        img = img_denormalize(img)
        img = put_arrow(img, label)
        cv2.imshow('img',img)
        cv2.waitKey(0)

    dataloader = DataLoader(facingDroneLabelDataset, batch_size=4, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)

    # import ipdb;ipdb.set_trace()

    for sample in dataloader:
      print sample['label'], sample['img'].size()
      print seq_show(sample['img'].numpy())

