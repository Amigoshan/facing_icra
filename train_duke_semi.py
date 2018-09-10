import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from os.path import join
import random

from utils import loadPretrain2, loadPretrain
from labelData import LabelDataset
from folderUnlabelData import FolderUnlabelDataset
from trackingLabelData import TrackingLabelDataset
from MobileReg import MobileReg

import sys
sys.path.append('../WorkFlow')
from workflow import WorkFlow

exp_prefix = '1_1_'
Batch = 128
UnlabelBatch = 24 #32
Lr = 0.0005
Trainstep = 50000
Lamb = 0.05
Thresh = 0.005


Snapshot = 5000 # do a snapshot every Snapshot steps
TestIter = 50 # do a testing every TestIter steps
ShowIter = 50 # print to screen

datasetdir = '/datadrive/datasets'
trainfile = 'trainval_duke.txt'
saveModelName = 'facing'

pre_mobile_model = 'pretrained_models/mobilenet_v1_0.50_224.pth'
LoadPreMobile = True
pre_model = ''
LoadPreTrain = False

LogParamList= ['Batch', 'UnlabelBatch', 'Lr', 'Trainstep', 'Lamb', 'Thresh'] # these params will be log into the file

class MyWF(WorkFlow.WorkFlow):
    def __init__(self, workingDir, prefix = "", suffix = ""):
        super(MyWF, self).__init__(workingDir, prefix, suffix)

        # === Custom member variables. ===
        logstr = ''
        for param in LogParamList: # record useful params in logfile 
            logstr += param + ': '+ str(globals()[param]) + ', '
        self.logger.info(logstr) 

        self.countEpoch = 0
        self.countTrain = 0
        self.device = 'cuda'

        # Dataloader for the training and testing
        labeldataset = LabelDataset(balence=True)
        unlabeldataset = FolderUnlabelDataset(batch = UnlabelBatch, data_aug=True, datafile='duke_unlabeldata.pkl')
        testdataset = TrackingLabelDataset(filename='test_duke.txt', data_aug=True)
        self.train_loader = DataLoader(labeldataset, batch_size=Batch, shuffle=True)
        self.train_unlabel_loader = DataLoader(unlabeldataset, batch_size=1, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=Batch, shuffle=True)

        self.train_data_iter = iter(self.train_loader)
        self.train_unlabeld_iter = iter(self.train_unlabeld_loader)
        self.test_data_iter = iter(self.test_loader)

        self.model = MobileReg()
        if LoadPreMobile:
            self.model.load_pretrained_pth(pre_mobile_model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=Lr)
        self.criterion = nn.NLLLoss() 

        self.AV['loss'].avgWidth = 10 # there's a default plotter for 'loss'
        self.add_accumulated_value('accuracy', 10) # second param is the number of average data
        self.add_accumulated_value('test') 
        self.add_accumulated_value('test_accuracy')

        self.AVP.append(WorkFlow.VisdomLinePlotter("train_loss", self.AV, ['loss'], [False])) # False: no average line
        self.AVP.append(WorkFlow.VisdomLinePlotter("test_loss", self.AV, ['test'], [False]))
        self.AVP.append(WorkFlow.VisdomLinePlotter("train_test_accuracy", self.AV, ['accuracy', 'test_accuracy'], [True, False]))
        self.AVP.append(WorkFlow.VisdomLinePlotter("train_test_loss", self.AV, ['loss', 'test'], [True, False]))

    def initialize(self, device):
        super(MyWF, self).initialize()

        # === Custom code. ===
        self.logger.info("Initialized.")
        self.device = device
        self.model.to(device)


    def train(self):
        super(MyWF, self).train()
        self.countTrain += 1

        # === Custom code for training ===
        self.model.train()

        try:
            (data, target) = self.train_data_iter.next()
        except:
            self.train_data_iter = iter(self.train_loader)
            (data, target) = self.train_data_iter.next()
            self.countEpoch += 1

        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        accuracy = self.get_accuracy(output, target)

        # print and visualization
        self.AV['loss'].push_back(loss.item())
        self.AV['accuracy'].push_back(accuracy)
        if self.countTrain % ShowIter == 0:
            losslogstr = self.get_log_str()
            self.logger.info("%s #%d - %s lr: %.6f" % (exp_prefix[:-1], 
                self.countTrain, losslogstr, Lr))

        if ( self.countTrain % Snapshot == 0 ):
            self.write_accumulated_values()
            self.draw_accumulated_values()
            self.save_model(self.model, saveModelName+'_'+str(self.countTrain))

    def test(self):
        super(MyWF, self).test()

        self.model.eval()

        try:
            (data, target) = self.test_data_iter.next()
        except:
            self.test_data_iter = iter(self.test_loader)
            (data, target) = self.test_data_iter.next()

        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        test_loss = self.criterion(output, target).item() # sum up batch loss
        accuracy = self.get_accuracy(output, target)

        self.AV['test'].push_back(test_loss, self.countTrain)
        self.AV['test_accuracy'].push_back(accuracy, self.countTrain)

    def finalize(self):
        super(MyWF, self).finalize()
        self.print_delimeter('finalize ...')
        self.write_accumulated_values()
        self.draw_accumulated_values()
        self.save_model(self.model, saveModelName+'_'+str(self.countTrain))


    def get_accuracy(self, output, target):
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        return float(correct)/output.size()[0]

    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        preTrainDict = {k:v for k,v in preTrainDict.items() if k in model_dict}
        for item in preTrainDict:
            print('  Load pretrained layer: ',item )
        model_dict.update(preTrainDict)
        model.load_state_dict(model_dict)
        return model    

    def save_model(self, model, modelname):
        modelname = self.prefix + modelname + self.suffix + '.pkl'
        torch.save(model.state_dict(), self.modeldir+'/'+modelname)

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    try:
        # Instantiate an object for MyWF.
        wf = MyWF("./", prefix = exp_prefix)

        # Initialization.
        wf.initialize(device)

        while True:
            

            wf.train()

            if wf.countTrain % TestIter == 0:
                wf.test()

            if (wf.countTrain>=Trainstep):
                break

        wf.finalize()

    except WorkFlow.SigIntException as e:
        wf.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )


