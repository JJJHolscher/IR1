from gensim.test.utils import common_dictionary, common_corpus
from gensim.models import LsiModel, LdaModel
import numpy as np
from gensim.corpora import Dictionary
import read_ap
import os
import pickle
from gensim import similarities



class LSIRetrieval:
	def __init__(self, model_type, path='lsi/', num_topics=500):
		assert model_type == "binary" or model_type == "tfidf", "acccepted model_type: 'binary' or 'tfidf'"
		self.path = path
		self.model_type = model_type
		self.dictionary = self.get_dictionary()
		self.model = self.get_model(num_topics)
		self.index = self.get_index()
		self.doc_index_map = {i : doc_id for i, (doc_id,_) in enumerate(read_ap.get_processed_docs().items())}


	def get_dictionary(self):
		tmp_fname = self.path + self.model_type + "_dictionary"

		if os.path.exists(tmp_fname):
			return Dictionary.load_from_text(tmp_fname)

		else:
			print("Creating dictionary.")
			docs_by_id = read_ap.get_processed_docs()
			docs = [doc for doc_id, doc in docs_by_id.items()]
			dictionary = Dictionary(docs)
			dictionary.filter_extremes(no_below=20, no_above=0.5)
			dictionary.save_as_text(tmp_fname)
			return dictionary


	def get_model(self, num_topics):
		tmp_fname = self.path + self.model_type + "_model"

		if os.path.exists(tmp_fname):
			return LsiModel.load(tmp_fname)

		else:
			print("Training model.")
			return self.train_model(num_topics)


	def train_model(self, num_topics):
		corpus = self.get_corpus()
		model = LsiModel(corpus, num_topics=num_topics)
		tmp_fname = self.path + self.model_type + "_model"
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
		tmp_fname = self.path + self.model_type + "_index"

		if os.path.exists(tmp_fname):
			return similarities.MatrixSimilarity.load(tmp_fname)

		else:
			print("Creating index.")
			corpus = self.get_corpus()
			index = similarities.MatrixSimilarity(self.model[corpus])
			index.save(tmp_fname)
			return index


	def search(self, query, max_docs=1000):
		query_repr = read_ap.process_text(query)
		vec_bow = self.dictionary.doc2bow(query_repr)
		vec_lsi = self.model[vec_bow]
		sims = self.index[vec_lsi]
		sims = sorted(enumerate(sims), key=lambda item: -item[1])
		results = [(self.doc_index_map[doc_id], score.item()) for doc_id, score in sims[:max_docs]]

		return results


	def get_top_topics(self, num_topics, num_words):
		topics = self.model.print_topics(num_topics=num_topics, num_words=num_words)
		results = [[self.dictionary[int(x.split('*')[1].replace("\"", ""))] for x in content.split('+')] for _,content in topics]
		
		return results
			


class LDARetrieval:
	def __init__(self, path='lda/', num_topics=500):
		self.path = path
		self.dictionary = self.get_dictionary()
		self.model = self.get_model(num_topics)
		self.index = self.get_index()
		self.doc_index_map = {i : doc_id for i, (doc_id,_) in enumerate(read_ap.get_processed_docs().items())}


	def get_dictionary(self):
		tmp_fname = self.path + "lda.dictionary"

		if os.path.exists(tmp_fname):
			return Dictionary.load_from_text(tmp_fname)

		else:
			print("Creating dictionary.")
			docs_by_id = read_ap.get_processed_docs()
			docs = [doc for doc_id, doc in docs_by_id.items()]
			dictionary = Dictionary(docs)
			dictionary.save_as_text(tmp_fname)
			return dictionary


	def get_model(self, num_topics):
		tmp_fname = self.path + "lda.model"

		if os.path.exists(tmp_fname):
			return LdaModel.load(tmp_fname)

		else:
			print("Training model.")
			return self.train_model(num_topics)


	def train_model(self, num_topics):
		corpus = self.get_corpus()
		model = LdaModel(corpus, chunksize=2000, passes = 20, iterations = 200,  num_topics=num_topics, eval_every = None)
		tmp_fname = self.path + "lda.model"
		model.save(tmp_fname)

		return model


	def get_corpus(self):
		docs_by_id = read_ap.get_processed_docs()
		docs = [doc for doc_id, doc in docs_by_id.items()]
		doc_bows = [self.dictionary.doc2bow(doc) for doc in docs]
		corpus = [[(idx, 1) for idx,_ in bow] for bow in doc_bows]

		return corpus


	def get_index(self):
		tmp_fname = self.path + "lda.index"

		if os.path.exists(tmp_fname):
			return similarities.MatrixSimilarity.load(tmp_fname)

		else:
			print("Creating index.")
			corpus = self.get_corpus()
			index = similarities.MatrixSimilarity(self.model[corpus])
			index.save(tmp_fname)
			return index


	def search(self, query, max_docs=1000):
		query_repr = read_ap.process_text(query)
		vec_bow = self.dictionary.doc2bow(query_repr)
		vec_lda = self.model[vec_bow]
		sims = self.index[vec_lda]
		sims = sorted(enumerate(sims), key=lambda item: -item[1])
		results = [(self.doc_index_map[doc_id], score.item()) for doc_id, score in sims[:max_docs]]

		return results


	def get_top_topics(self, num_topics, num_words):
		topics = self.model.print_topics(num_topics=num_topics, num_words=num_words)
		results = [[self.dictionary[int(x.split('*')[1].replace("\"", ""))] for x in content.split('+')] for _,content in topics]

		return results
##
