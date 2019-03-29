import numpy as np
import pickle

from time import time
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from nltk.corpus import stopwords

from PCMethod import *
from data import *

stopwords_list = stopwords.words("english")
stopwords_list.remove('not')

def calcPrecRecF1Acc(Y_true,Y_predicted):
	######### comments ############
	#tp_plus_fp = #(True Positive) + #(False Positive)
	#tp_plus_fn = #(True Positive) + #(False Negative)
	#tp = #(True Positive)
	##############################
	
	tp_plus_fp = np.array([Y_predicted[i] for i in range(Y_predicted.size) if Y_predicted[i]==1]).sum() * 1.0
	print "tp+fp =", tp_plus_fp
	tp_plus_fn = np.array([Y_true[i] for i in range(Y_true.size) if Y_true[i]==1]).sum() * 1.0
	print "tp+fn =", tp_plus_fn
	tp = np.array([Y_true[i] for i in range(Y_true.size) if Y_predicted[i]==1]).sum() * 1.0
	print "tp =", tp
	match = np.array([1 for i in range(Y_true.size) if Y_predicted[i] == Y_true[i]]).sum() * 1.0
	acc = match/Y_true.size
	
	precision = tp/tp_plus_fp
	recall = tp/tp_plus_fn
	f_measure = (2.0*recall*precision)/(recall + precision)
	print "precision =", precision
	print "recall =", recall
	print "f_measure =", f_measure
	print "accuracy =", acc
	return precision,recall,f_measure,acc
	
def nfoldval(n = 10):
	vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b',stop_words = stopwords_list)
	x,y, worker = read_training_data('hard')
	x_ = x[np.where(y[:,0] != -1)[0]]
	y_ = y[np.where(y[:,0] != -1)[0]]
	x_u = x[np.where(y[:,0] == -1)[0]]
	y_u = y[np.where(y[:,0] == -1)[0]]
	cv = KFold(x_.shape[0],n_folds=n)
	prec_t = 0
	rec_t = 0
	f1_t = 0
	acc_t = 0
	subset = x_.shape[0] / n
	print "begin training easy"
	i = 1
	start = time()
	log = open("log_hard_bi_.txt",'w')
	for train, test in cv:
		print "training fold %d"%i
		log.write("training fold %d \n"%i)
		x_train = vectorizer.fit_transform(list(np.concatenate((x_[train], x_u))))
		y_train = np.concatenate((y_[train], y_u))
		print "training model..."
		log_file = "training_log_hard_bi_fold%d.txt"%i
		model = SimplePCMethod(x_train,y_train,0.01,0.0122,0.00036,0.0001)
		model.train(log_file)
		x_test = vectorizer.transform(list(x_[test]))
		y_test = y_[test]
		y_predict = sparse_logist(x_test, model.W[0])
		y_predict[y_predict > 0.5] = 1
		y_predict[y_predict <= 0.5] = 0
		prec, rec, f1, acc = calcPrecRecF1Acc(y_test[:,0],y_predict)
		prec_t += prec
		rec_t += rec
		f1_t += f1
		acc_t += acc
		log.write("Precision = %.3f \n"%prec)
		log.write("Recall = %.3f \n"%rec)
		log.write("F-Measure = %.3f \n"%f1)
		log.write("Accuracy = %.3f \n"%acc)
		log.write("\n")
		i+=1
	
	mprec = prec_t/n
	mrec = rec_t/n
	mf1 = f1_t/n
	macc = acc_t/n
	print "Mean Precision = ", mprec
	print "Mean Recall = ", mrec
	print "Mean FMeasure = ", mf1
	print "Mean Accuracy = ", macc
	log.write("Final result : \n")
	log.write("Mean Precision = %.3f \n"%mprec)
	log.write("Mean Recall = %.3f \n"%mrec)
	log.write("Mean FMeasure = %.3f \n"%mf1)
	log.write("Mean Accuracy = %.3f \n"%macc)
	total_time = start - time()
	hour = int(total_time) / 3600
	minutes = (int(total_time) % 3600) / 60
	seconds = (int(total_time) % 3600) % 60
	log.write("Total time {0} hours {1} minutes {2} seconds".format(hour, minutes, seconds))
	log.close()
	

# Utility function to report best scores
def report(results, n_top=5):
	paramlog = open("param_log1.txt",'w')
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			paramlog.write("Model with rank: {0}\n".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
				  results['mean_test_score'][candidate],
				  results['std_test_score'][candidate]))
			paramlog.write("Mean validation score: {0:.3f} (std: {1:.3f}) \n".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))
			paramlog.write("Parameters: {0} \n".format(results['params'][candidate]))
			print("")
	paramlog.close()
			
def paramTune():
	print "tuning parameter"
	vectorizer = CountVectorizer(stop_words = 'english')
	x,y, worker = read_training_data('hard')
	x_ = x[np.where(y[:,0] != -1)[0]]
	y_ = y[np.where(y[:,0] != -1)[0]]
	x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size = 0.2, random_state=526)
	clf = SKFriendlyPCMethod()
	param_dist = {"alpha": stats.expon(scale=0.01),
				  "eta": stats.expon(scale=0.01),
				  "lamda": stats.expon(scale=0.01)}
	random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter = 20)
	start = time()
	random_search.fit(vectorizer.fit_transform(list(x_train)), y_train)
	print("RandomizedSearchCV took %.2f seconds for 20 candidates parameter settings." % (time() - start))
	report(random_search.cv_results_)

def createPickle():
	x,y_hard, worker = read_training_data('hard')
	x,y_easy, worker = read_training_data('easy')
	clf_hard = SKFriendlyPCMethod(alpha=0.01,eta=0.0122,lamda=0.00036)
	clf_easy = SKFriendlyPCMethod(alpha=0.01,eta=0.0122,lamda=0.00036)
	clf_hard.fit(list(x),y_hard)
	clf_easy.fit(list(x),y_easy)
	pkl_hard = open("PCMethod_hard.pkl",'wb')
	pkl_easy = open("PCMethod_easy.pkl",'wb')
	pickle.dump(clf_hard, pkl_hard, -1)
	pickle.dump(clf_easy, pkl_easy, -1)
	pkl_hard.close()
	pkl_easy.close()
	
"""
clf = SKFriendlyPCMethod(alpha = 0.000392, eta = 0.00692, lamda = 0.00331
precision :  0.873015873016
recall :  0.632183908046
f_measure :  0.733333333333

clf = SKFriendlyPCMethod(alpha = 0.0155, eta = 0.00157, lamda = 0.000461)
precision :  0.786764705882
recall :  0.614942528736
f_measure :  0.690322580645
"""	
if __name__ == "__main__":
	createPickle()
