# Combine two labeled dataset together

from folderUnlabelData import FolderUnlabelDataset
from torch.utils.data import Dataset, DataLoader
from utils import seq_show, put_arrow, seq_show_with_arrow

class UnlabelDataset(Dataset):

    def __init__(self, batch, balence=False, mean=[0,0,0], std=[1,1,1]):
        self.balencelist = [4,1]
        self.balence = balence

        self.datasetlist = []
        ucf = FolderUnlabelDataset(batch = batch, data_aug=True, datafile='ucf_unlabeldata.pkl',mean=mean,std=std) # 940
        duke = FolderUnlabelDataset(batch = batch, data_aug=True, datafile='duke_unlabeldata.pkl',mean=mean,std=std) # 3997

        self.datasetlist.append(ucf)
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

    unlabeldataset = UnlabelDataset(batch=24, balence=True)

    dataloader = DataLoader(unlabeldataset, batch_size=1, shuffle=True, num_workers=1)

    # # datalist=[0,69679,69680,69680*2-1,69680*2,364785,364786]
    # for k in dataloader:
    #     sample = labeldataset[k]
    #     img = sample['img']
    #     label = sample['label']
    #     print img.dtype, label
    #     print np.max(img), np.min(img), np.mean(img)
    #     print img.shape
    #     img = img_denormalize(img)
    #     img = put_arrow(img, label)
    #     cv2.imshow('img',img)
    #     cv2.waitKey(0)


    dataiter = iter(dataloader)

    # import ipdb;ipdb.set_trace()

    print len(unlabeldataset)
    for sample in dataloader:
      seq_show(sample.squeeze().numpy(), scale=0.8)
