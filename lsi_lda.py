from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel
import numpy as np
from gensim.corpora import Dictionary
import read_ap
from gensim.test.utils import get_tmpfile
import os
import pickle
from gensim import similarities



class LSIRetrieval:
	def __init__(self, model_type):
		assert model_type == "binary" or model_type == "tfidf", "acccepted model_type: 'binary' or 'tfidf'"
		self.model_type = model_type
		self.dictionary = self.get_dictionary()
		self.model = self.get_model()
		self.index = self.get_index()
		self.doc_index_map = {i : doc_id for i, (doc_id,_) in enumerate(read_ap.get_processed_docs().items())}


	def get_dictionary(self):
		tmp_fname = get_tmpfile(self.model_type + ".dictionary")

		if os.path.exists(tmp_fname):
			return Dictionary.load_from_text(tmp_fname)

		else:
			print("Creating dictionary.")
			docs_by_id = read_ap.get_processed_docs()
			docs = [doc for doc_id, doc in docs_by_id.items()]
			dictionary = Dictionary(docs)
			dictionary.save_as_text(tmp_fname)
			return dictionary


	def get_model(self):
		tmp_fname = get_tmpfile(self.model_type + ".model")

		if os.path.exists(tmp_fname):
			return LsiModel.load(tmp_fname)

		else:
			print("Training model.")
			return self.train_model()


	def train_model(self):
		corpus = self.get_corpus()
		model = LsiModel(corpus, num_topics=500)
		tmp_fname = get_tmpfile(self.model_type + ".model")
		model.save(tmp_fname)

		return model


	def get_corpus(self):
		docs_by_id = read_ap.get_processed_docs()
		docs = [doc for doc_id, doc in docs_by_id.items()]
		doc_bows = [self.dictionary.doc2bow(doc) for doc in docs]

		if self.model_type == "binary":
			corpus = [[(idx, 1) for idx,_ in bow] for bow in doc_bows]

		elif self.model_type == "tfidf":
			df = self.dictionary.dfs
			corpus = [[(idx, (np.log(1 + tf) / df[idx])) for idx,tf in bow] for bow in doc_bows]

		return corpus


	def get_index(self):
		tmp_fname = get_tmpfile(self.model_type + ".index")

		if os.path.exists(tmp_fname):
			return similarities.MatrixSimilarity.load(tmp_fname)

		else:
			print("Creating index.")
			corpus = self.get_corpus()
			index = similarities.MatrixSimilarity(self.model[corpus])
			index.save(tmp_fname)
			return index


	def search(self, query, max_docs=1000):
		vec_bow = self.dictionary.doc2bow(query)
		vec_lsi = self.model[vec_bow]
		sims = self.index[vec_lsi]
		sims = sorted(enumerate(sims), key=lambda item: -item[1])
		results = [(self.doc_index_map[doc_id], score) for doc_id, score in sims[:max_docs]]

		return results
	




	
x = LSIRetrieval("tfidf")

# d = read_ap.get_processed_docs()
q = ['prime', 'fli', 'close']
r = x.search(q)	
docs = read_ap.get_processed_docs()

for i in range(10):
	s = r[i][0]
	print(docs[s])

