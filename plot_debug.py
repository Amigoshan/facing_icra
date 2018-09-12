import matplotlib.pyplot as plt
import numpy as np
# from utils import groupPlot
from os.path import join

def groupPlot(datax, datay, group=10):
    datax, datay = np.array(datax), np.array(datay)
    if len(datax)%group>0:
        datax = datax[0:len(datax)/group*group]
        datay = datay[0:len(datay)/group*group]
    datax, datay = datax.reshape((-1,group)), datay.reshape((-1,group))
    datax, datay = datax.mean(axis=1), datay.mean(axis=1)
    return (datax, datay)

exp_ind = '1_1_'
datadir = 'logdata'
filelist = [['loss','test_loss'],
			['label_loss','test_label'],
			['unlabel_loss','test_unlabel'],
			]
labellist = [['training loss','validation loss'],
			 ['training loss','validation loss'],
			 ['training loss','validation loss'],
			 ]
titlelist = ['loss',
			 'label',
			 'unlabel',
			 ]
imgoutdir = 'resimg_facing'
AvgNum = 100

for ind,files in enumerate(filelist):
	print ind, files
	ax=plt.subplot(int('22'+str(ind+1)))
	# lines = []

	for k,filename in enumerate(files):

		filename = exp_ind+filename+'.npy'
		loss = np.load(join(datadir,filename))
		print loss.shape
		if k==1: # test data
			loss[:,0]=loss[:,0]*10
			datax, datay = groupPlot(loss[:,0],loss[:,1], group=1)
			ax.plot(datax, datay,label=labellist[ind][k])
		# ax.plot(loss[:,0],loss[:,1], label=labellist[ind][k])
		if k==0:
			datax, datay = groupPlot(loss[:,0],loss[:,1], group=1)
			ax.plot(datax, datay, label=labellist[ind][k])


	ax.grid()
	ax.legend()
	# ax.set_ylim(0,0.8)
	ax.set_xlabel('number of iterations')
	ax.set_ylabel('loss')
	ax.set_title(titlelist[ind])

plt.show()

