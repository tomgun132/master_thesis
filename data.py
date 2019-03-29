import nltk
import csv
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from feature_extractore import RationalesFeature
stopwords_list = stopwords.words("english")
stopwords_list.remove('not')

class Data:
	def __init__(self):
		self.loadChosenReviews()
		
	def loadChosenReviews(self):
		review_reader = csv.reader(open("reviews_Video_Games_filtered_10.csv",'rb'))
		review_list = list(review_reader)
		asin_reader = csv.reader(open("game_list_full_filtered.csv",'rb'))
		asin_list = list(asin_reader)
		per_game_review = {}
		for review in review_list:
			for asin in asin_list[1:]:
				if review[1] == asin[0]:
					game_ID = "{0}_{1}".format(review[0],review[1])
					review_text = self.preProcess(review[5])
					sents = nltk.sent_tokenize(review_text)
					if len(sents) < 10: continue
					if asin[1] in per_game_review:
						per_game_review[asin[1]][0].append(review_text)
						per_game_review[asin[1]][1].append(game_ID)
					else:
						per_game_review[asin[1]] = ([review_text],[game_ID],asin[4])
						
		game_list = {}
		label_count = {}
		reviews = []
		easy_game_list = {}
		hard_game_list = {}
		reviewer_count = {}
		id_file_out = open("test_id_list.txt",'w')
		j = 0
		for title,item in per_game_review.iteritems():
			reviews_len = [len(review) for review in item[0]]
			len_index = sorted(range(len(reviews_len)), key=lambda k: reviews_len[k], reverse = True)
			chosen_review = []
			chosen_ID = []
			each_review = []
			if item[2] == 'easy': easy_game_list[title] = []
			elif item[2] == 'hard': hard_game_list[title] = []
			for i in len_index[:30]:
				id_file_out.write(item[1][i] + '\n')
				label = 0 if item[2] == 'hard' else 1
				each_review.append((item[0][i],item[1][i],label, title))
				if item[2] == 'hard':
					hard_game_list[title].append((item[0][i],item[1][i]))
				elif item[2] == 'easy':
					easy_game_list[title].append((item[0][i],item[1][i]))
					
				reviewer_ID, asin = item[1][i].split("_")
				if reviewer_ID in reviewer_count:
					reviewer_count[reviewer_ID] += 1
				else:
					reviewer_count[reviewer_ID] = 1
				if item[2] in label_count:
					label_count[item[2]] += 1
				else:
					label_count[item[2]] = 1
					
			reviews.append(each_review)

		self.reviews = reviews
    
	def preProcess(self, text):
		regex = r'[a-zA-Z0-9]+.\.([A-Z][A-Za-z0-9]*)' #Match string like "this.I" or "that.Because", error for like "We like GTA V.We also..."
		p = re.compile(regex)
		h = HTMLParser.HTMLParser()
		problem_words = p.findall(text)
		text = text.replace("&#8217;","'")
		text = text.replace("&#8216;","'")
		text = text.replace("&#8242;","'")
		text = text.replace("&#8482;","")
		text = text.replace("&#61514;","")
		text = text.replace("&#8211;","")
		text = text.replace("&#8207;","")
		text = text.replace("&#9650;","")
		text = text.replace("&#9632;","")
		text = text.replace("&eacute;","e")
		text = text.replace("&#1082;","k")
		text = text.replace("&#8230;",",")
		text = text.replace("&#8220;",'"')
		text = text.replace("&#8221;",'"')
		text = text.replace("won't","will not")
		text = text.replace("can't","can not")
		text = text.replace("n't",' not')
		sym = r"""([-!$%^&*()_+|~=`{}\[\]:";'<>?,\/]{2,})""" #Match symbols except period 2 or more
		sym2 = r"""(\.)([-!$%^&*()_+|~=`{}\[\]:";'<>?,\/])"""
		# sepsym = r"""([a-zA-Z0-9]+[-!$%^&*()_+|~=`{}\[\]:";'<>?,\/])([A-Za-z0-9]+)"""
		text = re.sub(r'(\.[A-Z])(\.)(\W[a-z]+)',r'\1\3',text) #Match and remove period after abbreviation with a lot of periods, e.g. B.O.W. become B.O.W
		text = re.sub(sym,r' ',text)
		text = re.sub(sym2,r'\1 \2',text)
		# text = re.sub(sepsym,r'\1 \2',text)
		text = re.sub(r'(\. ){2,}',', ',text)
		text = h.unescape(text)
		if len(problem_words) != 0:
			for word in problem_words:
				if len(word) == 1 and word != "I": continue
				else: text = text.replace(word," "+word)
		text = re.sub(' +',' ',text)
		return text
    
def getTrainTest(type):
	vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b',stop_words = stopwords_list)
	data = Data()
	train, test = train_test_split(data.reviews, test_size = 0.2, random_state=526)
	train_ = []
	test_ = []
	y_train = []
	y_test = []
	title_test = []
	if type == 1:
		for item in train:
			for tuple in item:
				train_.append(tuple[0])
				y_train.append(tuple[2])
		for item in test:
			for tuple in item:
				test_.append(tuple[0])
				y_test.append(tuple[2])
				title_test.append(tuple[3])
		x_train = vectorizer.fit_transform(train_)
		x_test = vectorizer.transform(test_)
	elif type == 2:
		easy_clf = pickle.load(open('PCMethod_easy.pkl','rb'))
		hard_clf = pickle.load(open('PCMethod_hard.pkl','rb'))
		feat_extractor = RationalesFeature(hard_clf,easy_clf)
		review_len_train = []
		review_len_test = []
		reviewer_train = []
		reviewer_test = []		
		for item in train:
			for tuple in item:
				sents = nltk.sent_tokenize(tuple[0])
				reviewer_train.append(tuple[1].split('_')[0])
				train_.extend(sents)
				review_len_train.append(len(sents))
				y_train.append(tuple[2])
		for item in test:
			for tuple in item:
				sents = nltk.sent_tokenize(tuple[0])
				reviewer_test.append(tuple[1].split('_')[0])
				test_.extend(sents)
				review_len_test.append(len(sents))
				y_test.append(tuple[2])
				title_test.append(tuple[3])
		x_sents_train = vectorizer.fit_transform(train_)
		x_sents_test = vectorizer.transform(test_)
		x_train = feat_extractor.extract_features(x_sents_train, train_, review_len_train, y_train, reviewer_train, types = 'user_model')
		x_test = feat_extractor.extract_features(x_sents_test, test_, review_len_test, y_test, reviewer_test, types = 'user_model')

	return x_train, y_train, x_test, y_test  
