import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from mobilenet import mobilenet_v1_050

class MobileReg(nn.Module):

    def __init__(self, hidnum=256, regnum=2): # input size should be 112
        super(MobileReg,self).__init__()
        self.feature = mobilenet_v1_050()
        self.conv7 = nn.Conv2d(hidnum, hidnum, 3) # conv to 1 by 1
        self.reg = nn.Linear(hidnum, regnum)
        self._initialize_weights()

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        x = self.feature(x)
        # print x.size()
        x = F.relu(self.conv7(x), inplace=True)
        # print x.size()
        x = self.reg(x.view(x.size()[0], -1))

        return x


    def _initialize_weights(self):
        for m in self.modules():
            # print type(m)
            if isinstance(m, nn.Conv2d):
                # print 'conv2d'
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # print 'batchnorm'
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # print 'linear'
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_pretrained_pth(self, fname): # load mobilenet-from-tf - amigo
        params = torch.load(fname)
        self.feature.load_from_npz(params)

if __name__ == '__main__':
    
    inputVar = Variable(torch.rand((10,3,192,192)))
    net = MobileReg()
    net.load_pretrained_pth('pretrained_models/mobilenet_v1_0.50_224.pth')
    outputVar = net(inputVar)
    print outputVar
    # hiddens = [3,16,32,32,64,64,128,256] 
    # kernels = [4,4,4,4,4,4,3]
    # paddings = [1,1,1,1,1,1,0]
    # strides = [2,2,2,2,2,2,1]

    # datasetdir='/home/wenshan/datasets'
    # unlabel_batch = 4
    # lr = 0.005

    # from facingDroneUnlabelData import FacingDroneUnlabelDataset
    # from torch.utils.data import DataLoader
    # from os.path import join
    # import torch.nn as nn
    # import torch.optim as optim

    # stateEncoder = EncoderReg_Pred(hiddens, kernels, strides, paddings, actfunc='leaky',rnnHidNum=128)
    # print stateEncoder
    # paramlist = list(stateEncoder.parameters())
    # # for par in paramlist:
    # #     print par.size()
    # print len(paramlist)
    # stateEncoder.cuda()
    # imgdataset = FacingDroneUnlabelDataset(imgdir=join(datasetdir,'dirimg'), 
    #                                    batch = unlabel_batch, data_aug=True, extend=False)    
    # dataloader = DataLoader(imgdataset, batch_size=1, shuffle=True, num_workers=1)

    # criterion = nn.MSELoss()
    # regOptimizer = optim.SGD(stateEncoder.parameters(), lr = lr, momentum=0.9)
    # # regOptimizer = optim.Adam(stateEncoder.parameters(), lr = lr)

    # lossplot = []
    # encodesumplot = []
    # ind = 0
    # for sample in dataloader:
    #     ind += 1
    #     inputVar = Variable(sample.squeeze()).cuda()
    #     # print inputVar.size()
    #     x, encode, pred = stateEncoder(inputVar)
    #     # print encode.size(), x.size(), pred.size()

    #     # print encode

    #     pred_target = encode[unlabel_batch/2:,:].detach()

    #     loss_pred = criterion(pred, pred_target)

    #     # # loss = loss_label + loss_pred * lamb #+ normloss * lamb2

    #     # # zero the parameter gradients
    #     regOptimizer.zero_grad()
    #     loss_pred.backward()
    #     # # loss.backward()
    #     regOptimizer.step()

    #     lossplot.append(loss_pred.data[0])
    #     encodesumplot.append(encode.mean().data[0])
    #     print ind,loss_pred.data[0], encode.mean().data[0]

    #     if ind>=1000:
    #         break


    # import matplotlib.pyplot as plt
    # plt.plot(lossplot)
    # plt.plot(encodesumplot)
    # plt.grid()
    # plt.show()
