import csv
import pickle
import nltk

from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin

class RationalesFeature(BaseEstimator, ClassifierMixin):
	"""
	need: 
	sent_len: array with length N = number of reviews, which contains the number of sentences in each review
	x_sent: count vectorizer for every sentence
	hard_prob/easy_prob: array for every sentencew
	"""
	def __init__(self, hard_clf, easy_clf):
		self.hard_clf = hard_clf
		self.easy_clf = easy_clf
		
	def rationales_prob_clf(self, sent_reviews, sent_dist, y_train, reviewers, types):
		hard_prob = self.hard_clf.predict_prob(sent_reviews)
		easy_prob = self.easy_clf.predict_prob(sent_reviews)
		print types
		if types == 'user_model':
			print "user model commence"
			results_reviewer, result_ID = readTaggingResults(self.hard_clf, self.easy_clf)
			user_score = userAbilities(results_reviewer)
			j = 0
			for i in xrange(len(sent_reviews)):
				if i >= sent_dist[j] : j+=1
				try:
					if hard_prob[i] > easy_prob[i]:
						if user_score[reviewers[j]][0] > 0 and hard_prob[i] > 0.5: hard_prob[i] *= 2.0
					else:
						if user_score[reviewers[j]][0] < 0 and easy_prob[i] > 0.5: easy_prob[i] *= 2.0
				except KeyError:
					"reviewer not found"
		prob = np.maximum(easy_prob,hard_prob)
		return prob
		
	def extract_features(self, x_sent, sent_reviews, review_len, y_train, reviewers, types):
		"""matrix multplication of len_reviews x len_sents * len_sents x len_features"""
		print "extracting features"
		sent_dist = np.cumsum(review_len)
		prob = self.rationales_prob_clf(sent_reviews, sent_dist, y_train, reviewers, types)
		S = np.zeros((len(review_len), len(sent_reviews)))
		cur_pos = 0
		for i,pos in enumerate(sent_dist):
			next_pos = pos
			S[i][cur_pos:next_pos] += prob[cur_pos:next_pos]
			cur_pos = pos
		print S
		print "converting and calculating"
		S_sparse = sparse.csr_matrix(S)	
		return S_sparse * x_sent
    
  def readTaggingResults(self):
    review_reader = csv.reader(open("reviews_Video_Games_filtered_10.csv",'rb'))
    review_list = list(review_reader)
    asin_reader = csv.reader(open("game_list_full_filtered.csv",'rb'))
    asin_list = list(asin_reader)
    results_reviewer = {}
    result_ID = {}
    for review in review_list[1:]:
      for asin in asin_list[1:]:
        if review[1] == asin[0]:
          ID = "{0}_{1}".format(review[0],review[1])
          reviewer = review[0]
          review_text = preProcess(review[5])
          sents = nltk.sent_tokenize(review_text)
          hard_prob = self.hard_clf.predict_prob(sents)
          easy_prob = self.easy_clf.predict_prob(sents)
          label_temp = []
          for i in xrange(len(sents)):
            if hard_prob[i]>easy_prob[i]:
              if hard_prob[i]>0.5: label_temp.append("hard_%d"%i)
            else:
              if easy_prob[i]>0.5: label_temp.append("easy_%d"%i)
          if reviewer in results_reviewer:
            results_reviewer[reviewer].append((label_temp, review[1]))
          else:
            results_reviewer[reviewer] = [(label_temp, review[1])]
          result_ID[ID] = label_temp

    return results_reviewer, result_ID  
