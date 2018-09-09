from trackingLabelData import TrackingLabelDataset
from torch.utils.data import Dataset, DataLoader
from utils import im_scale_norm_pad, img_denormalize, seq_show, im_crop, im_hsv_augmentation, put_arrow, seq_show_with_arrow

class LabelDataset(Dataset):

    def __init__(self, balence=False):
        self.balencelist = [2,1]
        self.balence = balence

        self.datasetlist = []
        virat = TrackingLabelDataset(data_aug = True, maxscale=0.1) # 69680
        duke = TrackingLabelDataset(filename='/datadrive/person/DukeMTMC/trainval_duke.txt', data_aug=True) # 225426

        self.datasetlist.append(virat)
        self.datasetlist.append(duke)

        self.datanumlist = []
        for k, dataset in enumerate(self.datasetlist):
            datanum = len(dataset)
            if self.balence:
                datanum *= self.balencelist[k]
            self.datanumlist.append(datanum)

        self.totalnum = sum(self.datanumlist)


    def __len__(self):
        return self.totalnum

    def __getitem__(self, idx):
        ind = idx
        for k,datanum in enumerate(self.datanumlist):
            if ind >= datanum:
                ind -= datanum
            else: # find the value
                if self.balence:
                    ind = ind%(int(self.datanumlist[k]/self.balencelist[k]))
                return self.datasetlist[k][ind]
        print 'Error Index:', ind
        return 

        

if __name__=='__main__':
    # test 
    import numpy as np
    import cv2
    np.set_printoptions(precision=4)

    labeldataset = LabelDataset(balence=True)

    dataloader = DataLoader(labeldataset, batch_size=16, shuffle=True, num_workers=1)

    datalist=[0,69679,69680,69680*2-1,69680*2,364785,364786]
    for k in datalist:
        sample = labeldataset[k]
        img = sample['img']
        label = sample['label']
        print img.dtype, label
        print np.max(img), np.min(img), np.mean(img)
        print img.shape
        img = img_denormalize(img)
        img = put_arrow(img, label)
        cv2.imshow('img',img)
        cv2.waitKey(0)


    dataiter = iter(dataloader)

    # import ipdb;ipdb.set_trace()

    print len(labeldataset)
    for sample in dataloader:
      print sample['label'], sample['img'].size()
      seq_show_with_arrow(sample['img'].numpy(), sample['label'].numpy(), scale = 0.5)
