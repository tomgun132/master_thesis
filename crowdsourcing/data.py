import csv
import collections
import numpy as np
from sklearn.utils import shuffle

def read_training_data(type):
	worker_label_reader = csv.reader(open("train_data.csv",'rb'))
	worker_label_list = list(worker_label_reader)
	self_label_reader = csv.reader(open("train_data_.csv",'rb'))
	self_label_list = list(self_label_reader)
	worker_dict = {}
	label_list = []
	data_list = []
	for i,wlabels in enumerate(worker_label_list[1:]):
		data_list.append(wlabels[1])
		if wlabels[0] == type:
			label_list.append(1.)
		elif wlabels[0] == 'unsure':
			label_list.append(-1.)
		else:
			label_list.append(0.)
		workers = wlabels[2].split(',')
		for worker in workers:
			wid,label = worker.split(':')
			if label == type: label = 1.
			else: label = 0
			if wid in worker_dict:
				worker_dict[wid].append((label,i))
			else:
				worker_dict[wid] = [(label,i)]
				
	sdata_list = []
	slabel_list = []
	for slabels in self_label_list[1:]:
		sdata_list.append(slabels[1])
		if slabels[0] == type:
			slabel_list.append(1.)
		else:
			slabel_list.append(0.)

	self_N = len(sdata_list)
	sdata_list += data_list
	slabel_list += label_list
	ord_worker_dict = collections.OrderedDict(sorted(worker_dict.items()))
	J = len(worker_dict)
	N = len(sdata_list)
	y = np.zeros([N,J+1])
	y[:,0] = slabel_list
	j = 1
	print J
	print N
	for wid,labels in ord_worker_dict.iteritems():
		for label,pos in labels:
			y[self_N + pos,j] = label
		j+=1
		
	x = np.asarray(sdata_list)
	new_x, new_y = shuffle(x,y,random_state=10)
	
	return new_x, new_y, ord_worker_dict
