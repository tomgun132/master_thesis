import numpy as np
from time import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse, stats
from scipy.optimize import minimize
from nltk.corpus import stopwords

stopwords_list = stopwords.words("english")
stopwords_list.remove('not')

class SKFriendlyPCMethod(BaseEstimator, ClassifierMixin):

	def __init__(self, alpha = 0.01, eta = 0.01, lamda = 0.015):
		"""
		Variables:
			N : int
				number of data instances
			J : int
				number of workers, excluding target classifier
			d : int
				number of the feature space
			alpha : float
				coefficient for optimization learning rate
			eta : float
				coefficient for regularization of w0
			lamda : float
				coefficient for regularization of weight difference
			W : numpy.array
				Weight matrix with size [J+1, d]. W[j] is the parameter of the j-th worker, W[0] is the parameter of target model
			
		"""
		self.alpha = alpha
		self.eta = eta
		self.lamda = lamda
		print "Training with parameter alpha = {0}".format(self.alpha)
		print "Training with parameter eta = {0}".format(self.eta)
		print "Training with parameter lamda = {0}".format(self.lamda)
		
	def _logist(self,x,w):
		return np.exp(-np.logaddexp(0,-(x*w)))
	
	def _neg_log_post(self):
		loss = -(self.y * np.log(self._logist(self.x, self.W_.T)) + (1 - self.y) * np.log(1 - self._logist(self.x, self.W_.transpose()))).sum()
		model_relation = 0.5 * self.lamda * np.sum(np.linalg.norm(self.W_[1:] - self.W_[0], axis=1)**2)
		prior = 0.5 * self.eta * (np.linalg.norm(self.W_[0]) ** 2)
		return loss + model_relation + prior

	def _optimize_w0(self):
		w_in = self.W_[0]
		# y_in = np.split(self.y, np.where(self.y[:,0] == -1.)[0])[0][:,0]
		y_in = self.y[np.where(self.y[:,0] != -1)[0]][:,0]
		x_in = self.x[np.where(self.y[:,0] != -1.)[0]]
		grad = self._loss_grad(x_in,y_in,w_in) + self.lamda * np.sum(w_in - self.W_[1:], axis=0) + self.eta * w_in
		self.W_[0] = w_in - self.alpha * grad
		
	def _optimize_W(self):
		w_in = self.W_[1:] #Jxd
		y_in = self.y[:,1:] #NxJ
		x_in = self.x #Nxd
		grad = self._loss_grad(x_in,y_in,w_in.T) + self.lamda * (w_in - self.W_[0])
		self.W_[1:] = w_in - self.alpha * grad
	
	def _loss_grad(self, x_in,y_in,w_in):
		"""
		Return 1xd vector
		"""
		p = self._logist(x_in,w_in)
		return -((y_in - p).T * x_in)		
		
	def _train(self):
		iter = 0
		convergence = 0
		print "start training with max iter = %d"%self.iter
		log_post_old = self._neg_log_post()
		log_post_new = log_post_old
		print "Initialization obj function = ", log_post_new
		while iter < self.iter:
			log_post_old = log_post_new
			self._optimize_w0()
			self._optimize_W()
			log_post_new = self._neg_log_post()
			print "new obj function value = {0}, with {1} iteration".format(log_post_new,iter)
			iter += 1
			if np.abs(log_post_old - log_post_new)/log_post_new < self.eps:
				convergence = 1
				print "certificate : ", np.abs(log_post_old - log_post_new)/log_post_new
				break
				
		print "train finished with %d iteration" % iter
		
	def fit(self, X, y, eps = 0.0001, iter = 4000, ngrams = 2):
		try:
			if ngrams == 1:
				print "train with unigram"
				self.vectorizer = CountVectorizer(stop_words = stopwords_list)
			else:
				print "train with ngram"
				self.vectorizer = CountVectorizer(ngram_range=(1, ngrams),token_pattern=r'\b\w+\b',stop_words = stopwords_list)
			self.x = self.vectorizer.fit_transform(X)
			self.y = y
			self.N = self.x.shape[0]
			self.J = self.y.shape[1] - 1
			self.d = self.x.shape[1]
			self.eps = eps
			self.iter = iter
			self.W_ = np.zeros((self.J+1,self.d))
			self._train()
		except IndexError:
			print "x should be a 2d sparse matrix"
		
		return self
	
	def transform(self, X):
		return self.vectorizer.transform(X)
	
	def predict_prob(self,X):
		try:
			getattr(self, "W_")
		except AttributeError:
			raise RuntimeError("Train the classifier first")
		
		return self._logist(self.transform(X),self.W_[0])
		
	def predict(self,X):
		predicted = self.predict_prob(X)
		predicted[predicted > 0.5] = 1
		predicted[predicted <= 0.5] = 0
		return predicted
		
	def score(self, X, y):
		Y_predicted = self.predict(X)
		y_in = y[:,0]
		tp_plus_fp = np.array([Y_predicted[i] for i in range(Y_predicted.size) if Y_predicted[i]==1]).sum() * 1.0
		tp_plus_fn = np.array([y_in[i] for i in range(y_in.size) if y_in[i]==1]).sum() * 1.0
		tp = np.array([y_in[i] for i in range(y_in.size) if Y_predicted[i]==1]).sum() * 1.0
		precision = tp/tp_plus_fp
		recall = tp/tp_plus_fn
		f_measure = (2.0*recall*precision)/(recall + precision)
		print "precision : ", precision
		print "recall : ", recall
		print "f_measure : ", f_measure
		return f_measure
