import torch
import my_word2vec
from my_word2vec import get_trained_doc2vec, SkipGram, search_SkipGram,\
                        search_doc2vec, preprocess


class Doc2Vec():
	def _init_():
		self.model = get_trained_doc2vec()

	def search(query):
        return search_doc2vec(self.model, query)


class Word2Vec():

	def _init_():
        self.model = SkipGram.create(tok2idx)

	def search(query):
        return search_SkipGram(self.model, query)


def search(query, doc2vec=False):
    if doc2vec:
        return search_doc2vec(get_trained_doc2vec(), query)
    else:
        return search_SkipGram(SkipGram.create(), query)


def similar_words(in_words, skipgram):
    out = {}

    for word in in_words:
        if word not in skipgram.tok2idx:
            print("The word '", word, "' has no embedding in the skipgram.")
            continue
        print("Input word is:", word)

        idx = skipgram.tok2idx[word]
        embd = skipgram.embeddings(torch.LongTensor([idx]))

        tokens = list(skipgram.tok2idx.keys())
        indices = torch.LongTensor([skipgram.tok2idx[t] for t in tokens])
        all_embd = skipgram.embeddings(indices)

        scores = torch.mm(embd, all_embd.T).squeeze(dim=0)
        scores = [(tokens[i], float(s)) for i, s in enumerate(scores)]
        scores.sort(key=lambda _: -_[1])

        print("Top 10 similar words are:", scores[:10])
        out[word] = scores[1:11]
    return out





if __name__ == '__main__':
    query = "sentient quiet start and woman"
    print(search(query, doc2vec=True)[:10])

    similar_words(['quiet', 'quite', 'start', 'stop', 'march', 'woman', 'trump'], SkipGram.create())

