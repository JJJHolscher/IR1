import pytrec_eval
import json
import read_ap
import lsi_lda
from tf_idf import TfIdfRetrieval
from tqdm import tqdm
import scipy.stats
import pickle
import csv
import analysis

qrels, queries = read_ap.read_qrels()
val_keys = [key for key in qrels.keys() if int(key) >= 76 and int(key) <= 100]
test_keys = [key for key in qrels.keys() if key not in val_keys]
test_queries = {key : queries[key] for key in test_keys}
test_qrels = {key : qrels[key] for key in test_keys}
val_queries = {key : queries[key] for key in val_keys}
val_qrels = {key : qrels[key] for key in val_keys}



def evaluate(model, queries, qrels):
	run = {}

	print("Collecting results.")
	for key in tqdm(queries.keys()):
		results = model.search(queries[key])
		run[key] = {ref : score for ref, score in results}
	
	evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})
	# data = json.dumps(, indent=1)

	return evaluator.evaluate(run)


def get_averages(data):
	mapp = 0
	ndcg = 0

	n = len(data)

	for i in data:
		mapp += data[i]['map']
		ndcg += data[i]['ndcg']

	return mapp/n, ndcg/n


def get_model(idx):
	# TF-IDF MODEL
	if idx == 1:
		docs_by_id = read_ap.get_processed_docs()
		model = TfIdfRetrieval(docs_by_id)
		return model

	# LSI BINARY MODEL
	elif idx == 2:
		return lsi_lda.LSIRetrieval('binary')

	# LSI TF-IDF MODEL
	elif idx == 3:
		return lsi_lda.LSIRetrieval('tfidf')

	# LDA MODEL
	elif idx == 4:
		return lsi_lda.LDARetrieval()

	elif idx == 5:
		return analysis.Word2Vec()

	# LSI BINARY 5 TOPICS
	elif idx == 12:
		return lsi_lda.LSIRetrieval('binary', path="lsi/5topics", num_topics=5)


def make_results_file(model, run_name, folder='result_files/'):
	file_lines = []
	for query_id in test_keys:
		results = model.search(test_queries[query_id])

		for i, content in enumerate(results):
			doc_id, score = content
			line = query_id + ' Q0 ' + doc_id + ' ' + str(i+1) + ' ' + "{:.6f}".format(score) + ' ' + run_name 
			file_lines.append(line)

	path = folder + run_name + '.txt'

	with open(path, 'w') as f:
	    for item in file_lines:
	        f.write("%s\n" % item)


def analysis3_1(model):
	topics = model.get_top_topcis(5, 10)

	for i,terms in enumerate(topics):
		s = ', '.join(terms)
		print('Topic ' + str(i+1) + ': ' + s)


def analysis4_2(model1, model2):
	res1 = evaluate(model2, queries, qrels)
	res2 = evaluate(model2, queries, qrels)

	query_ids = list(set(res1.keys()) & set(res2.keys()))

	first_scores = [res1[query_id]['map'] for query_id in query_ids]
	second_scores = [res2[query_id]['map'] for query_id in query_ids]

	print(scipy.stats.ttest_rel(first_scores, second_scores))


def analysis4_1(model):
	res_all = evaluate(model, queries, qrels)
	res_test = evaluate(model, val_queries, val_qrels)

	print('Results for val set:')
	print("Map : % .2f, nDCG : % .2f" %(get_averages(res_test)))  

	print('Results for all queries:')
	print("Map : % .2f, nDCG : % .2f" %(get_averages(res_all)))



##
# print('ANALISIS 4.1\n')

# print('TF-IDF MODEL:')
# analysis4_1(get_model(1))

# print('BINARY LSI MODEL')
# analysis4_1(get_model(2)
# 	)
# print('TFIDF LSI MODEL')
# analysis4_1(get_model(3))

# print('LDA MODEL')
# analysis4_1(get_model(4))

# analysis4_2(get_model(2), get_model(4))


# analysis4_1(get_model(12))

# make_results_file(get_model(5), 'word2vec')

# analysis4_1(get_model(5))
