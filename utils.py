import read_ap


def get_docs_by_id():
	return read_ap.get_processed_docs()


def get_token_indices(docs_by_id):
	bow = set()
	id2tok = {}
	tok2id = {}

	for doc in docs_by_id.values():
		for token in doc:
			bow.add(token)

	for i,token in enumerate(bow):
		id2tok[i] = token
		tok2id[token] = i

	return id2tok, tok2id



def get_binary_bow(doc, tok2id):
	bow = set()
	result = []

	for token in doc:
		bow.add(tok2id[token])

	for idx in bow:
		result.append((idx,1))

	return  result
