import numpy as np

from data import getTrainTest
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression

def main(mode = 1, type = 1):
	x_train, y_train, x_test, y_test = getTrainTest(type)
  if mode==1:
    """Parameter Tuning"""
    clf = LogisticRegression()
    param_grid = {'C': [0.001, 0.05, 0.01,0.5, 0.1, 1, 5, 10,50, 100,500 ,1000] }
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    report(grid_search.cv_results_)
	else:
    """testing on test data"""
    clf = LogisticRegression(C=0.5)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print clf.score(x_test, y_test)
    calcPrecRecF1Acc(np.asarray(y_test),y_pred)
	
	# """Writing to file"""
	# csvOut = open("user_clf_result.csv",'wb')
	# writer = csv.writer(csvOut)
	# # writer.writerow(['title','reviews','sent_labels','true_y','pred_y'])
	# writer.writerow(['title','reviews','true_y','pred_y'])
	# # sent_dist = np.cumsum(review_len_test)
	# cur_pos = 0
	# for i in xrange(len(y_test)):
		# # next_pos = sent_dist[i]
		# # easy_prob = easy_clf.predict_prob(test_[cur_pos:next_pos])
		# # hard_prob = hard_clf.predict_prob(test_[cur_pos:next_pos])
		# # review = ''
		# # sent_labels = ''
		# # for j,sent in enumerate(test_[cur_pos:next_pos]):
			# # review += str(j) + '. ' + sent + '; '
			# # if easy_prob[j] > 0.5:
				# # sent_labels += 'easy_' + str(j)
			# # if hard_prob[j] > 0.5:
				# # sent_labels += 'hard_' + str(j)
		# # writer.writerow([title_test[i],review.encode('utf-8'), sent_labels, str(y_test[i]), str(y_pred[i])])
		# # cur_pos = next_pos
		# writer.writerow([title_test[i],test_[i].encode('utf-8'), str(y_test[i]), str(y_pred[i])])
	
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
	
if __name__== "__main__":
  main()
