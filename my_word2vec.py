""" 01-03-2020
A word2vec implementation by Fabrizio and Jochem
"""

import os
import random
import torch
import pickle
from collections import Counter
import gensim

# Own libraries
import download_ap
import read_ap
import utils

##### TUNABLE HYPERPARAMETERS FOR SKIPGRAM #####
# The minimum frequency a word should have for it to get an embedding.
OCCURANCE_THRESHHOLD = 100
# The number of features in any single word embedding.
EMBEDDING_SIZE = 100
# The number of words before or after a word that are considered its context.
CONTEXT_WINDOW = 2
# Number of documents in a batch.
BATCH_SIZE = 1
# The number of top documents returned for a given query
MAX_NUMBER_OF_RESULTS = 1000


##### TUNABLE HYPERPARAMETERS FOR DOC2VEC #####
D2V_CONTEXT_WINDOW = 2
D2V_EPOCHS = 2
D2V_EMBEDDING_SIZE = 50


##### EXTERNAL DOCUMENTS PATHS #####
# Pickle storage path
PROCESSED_DOCS_PATH = "processed_docs_SkipGram.pkl"
# SkipGram model storage path
SKIPGRAM_PATH = "my_skipgram.pth"
DOC2VEC_PATH = "my_doc2vec.gsm"
# The storage of a dictionary mapping doc_id to a vector representation made
# by the skipgram.
SKIPGRAM_DOC2VEC_PATH = "skipgrams_doc2vec.pkl"


class SkipGram(torch.nn.Module):

    def __init__(self, tok2idx, embedding_size=EMBEDDING_SIZE):
        super(SkipGram, self).__init__()
        self.tok2idx = tok2idx
        self.embeddings = torch.nn.Embedding(len(self.tok2idx), embedding_size,
                                             sparse=True)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, center_id, context_id):
        """ Forward propagation step """
        center_emb = self.embeddings(center_id)
        context_emb = self.embeddings(context_id)
        out = torch.mm(center_emb, context_emb.T)
        return torch.nn.functional.logsigmoid(out)

    def train(self, corpus, save_path=SKIPGRAM_PATH):
        """ Train """
        num_batches = len(corpus) // BATCH_SIZE + 1
        optimizer = torch.optim.SparseAdam(self.parameters())
        try:
            for i, batch in enumerate(self.to_batches(corpus)):
                centers, contexts, targets = batch[:, 0], batch[:, 1], batch[:, 2]

                optimizer.zero_grad()
                out = self(centers, contexts)
                out = torch.sum(out * torch.eye(len(out)), dim=0)
                loss = self.loss_func(out, targets.float())
                print("Loss of batch", i, "/", num_batches, "\t", float(loss))
                loss.backward()
                optimizer.step()
        finally:
            print(" ~ oops UwU")
            optimizer.zero_grad()
            torch.save(self.state_dict(), save_path)

        optimizer.zero_grad()
        torch.save(self.state_dict(), save_path)

    def to_batches(self, corpus, size=BATCH_SIZE):
        corpus = corpus.copy()
        random.shuffle(corpus)
        end_i = 0
        for start_i in range(0, len(corpus) - size, size):
            end_i = start_i + size
            yield sample(corpus[start_i : end_i])
        yield sample(corpus[end_i:])

    def doc2vec(self, doc):
        """ Return a vector representation of the input document by taking the
        average word embedding of all words in the document.
        """
        word_embeddings = []
        for token_idx in doc:
            word_embeddings.append(self.embeddings(token_idx))

        vec = torch.Tensor(len(word_embeddings[0]))
        for word_embd in word_embeddings:
            vec += word_embd

        return vec / len(word_embeddings)


def all_words_to_indices(docs_by_id):
    """ Create a corpus where all string representations of words are replaced
    by index representations.
    """
    id2corpus = {}
    tok2idx = {}

    for doc_id, doc in docs_by_id.items():
        doc_repr = []

        for token in doc:
            # Create a new index if the token has not occured before.
            if not (token in tok2idx):
                tok2idx[token] = len(tok2idx)
            doc_repr.append(tok2idx[token])

        if len(doc_repr) > 1:
            id2corpus[doc_id] = torch.LongTensor(doc_repr)

    return tok2idx, id2corpus


def preprocess(path=PROCESSED_DOCS_PATH):
    # Load the preprocessed docs_by_id file if it exists.
    if os.path.exists(path):
        print("Loading the preprocessed files...")
        with open(path, "rb") as reader:
            return pickle.load(reader)

    # (Down)load the dataset from the ap files and get it in the right form.
    download_ap.download_dataset()
    docs_by_id = read_ap.get_processed_docs()
    print("Filtering infrequent words...")
    docs_by_id = filter_infrequent(docs_by_id)
    print("Converting words to indices...")
    tok2idx, id2corpus = all_words_to_indices(docs_by_id)

    # Store the preprocessing results for faster future retrieval.
    print("Storing the preprocessed files...")
    with open(path, "wb") as writer:
        pickle.dump((tok2idx, id2corpus), writer)
    return tok2idx, id2corpus


def filter_infrequent(docs_by_id, occurance_threshhold=OCCURANCE_THRESHHOLD):
    """Create a dictionary similar to docs_by_id, but without any words
    occurring less then 'OCCURANCE_THRESHHOLD' times.
    """
    # Count all tokens.
    counter = Counter()
    for tokens in docs_by_id.values():
        counter.update(tokens)

    # Filter any token that doesn't meet the minimal frequency threshhold.
    for doc_id, doc in docs_by_id.items():
        filtered_doc = []
        for token in doc:
            if counter[token] >= occurance_threshhold:
                filtered_doc.append(token)
        docs_by_id[doc_id] = filtered_doc

    return docs_by_id


def sample(corpus, context_window=CONTEXT_WINDOW):
    dataset = []

    # Each row of the dataset is a center word and a context word and the
    # target value for the dot product of their embeddings.
    for doc_repr in corpus:
        for i, token in enumerate(doc_repr):
            min_context_idx = max(0, i - context_window)
            max_context_idx = min(len(doc_repr), i + context_window + 1)

            # negative sampling
            num_neg_samples = max_context_idx - min_context_idx - 1

            for fake_doc_repr in random.choices(corpus, k=num_neg_samples):
                fake_context = random.choice(fake_doc_repr)
                dataset.append([token, fake_context, 0])

            # positive sampling
            for context_idx in range(min_context_idx, i):
                dataset.append([token, doc_repr[context_idx], 1])
            for context_idx in range(i + 1, max_context_idx):
                dataset.append([token, doc_repr[context_idx], 1])

    return torch.LongTensor(dataset)


def get_docs_as_vecs(model, id2corpus, path=SKIPGRAM_DOC2VEC_PATH):
    if os.path.exists(path):
        print("Loading the doc2vec dictionary of the skipgram...")
        with open(path, "rb") as reader:
            return pickle.load(reader)

    print("Creating a doc2vec dictionary using the skipgram...")
    id2vec = {}
    for doc_id, doc in id2corpus.items():
        id2vec[doc_id] = model.doc2vec(doc)

    print("Storing the doc2vec dictionary...")
    with open(path, "wb") as writer:
        pickle.dump(id2vec, writer)
    return id2vec


def search(model, query, id2corpus, result_len=MAX_NUMBER_OF_RESULTS):
    id2vec = get_docs_as_vecs(model, id2corpus)

    # Transform the query to a vector.
    query_repr = []
    for q_tok in read_ap.process_text(query):
        if q_tok in model.tok2id:
            query_repr = model.tok2id[q_tok]
    q_vec = model.doc2vec(query_repr)

    cos = torch.nn.CosineSimilarity()
    results = {doc_id: cos(q_vec, d_vec) for doc_id, d_vec in id2vec.items()}
    results = list(results.items())
    results.sort(key=lambda _: _[1])
    return results[result_len]


def train_doc2vec(docs_by_id, batch_size=BATCH_SIZE, path=DOC2VEC_PATH,
                  batched=False):
    corpus = []
    print("Creating training corpus for doc2vec...")
    for doc_id, doc in docs_by_id.items():
        corpus.append(gensim.models.doc2vec.TaggedDocument(doc, [doc_id]))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=D2V_EMBEDDING_SIZE,
                                          epochs=D2V_EPOCHS,
                                          context_window=D2V_CONTEXT_WINDOW)
    model.build_vocab(corpus)

    if batched:
        print("Start training", len(corpus) // batch_size, "batches for doc2vec...")
        batch_i = 0
        try:
            for batch_i in range(0, len(corpus), batch_size):
                batch = corpus[batch_i: batch_i + min(len(corpus), batch_size)]
                model.train(batch, total_examples=model.corpus_count, epochs=model.epochs)
        except:
            model.save('batched_incomplete_train_' + DOC2VEC_PATH)
            print('    owo?  Stopped at batch', str(batch_i) + '?')
        model.save('batched_' + DOC2VEC_PATH)

    else:
        print("Start training model with the complete corpus")
        try:
            model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        except:
            print('... but an error occured.', end='')
            model.save('whole_corpus_incomplete_train_' + DOC2VEC_PATH)
            print(' it is still saved though :)')
        model.save('whole_corpus_' + DOC2VEC_PATH)





if __name__ == "__main__":
    train_doc2vec(read_ap.get_processed_docs())

    tok2idx, id2corpus = preprocess()
    skipgram = SkipGram(tok2idx)
    skipgram.train(list(id2corpus.values()))
    print(search(skipgram, "How are you", id2corpus))

    train_doc2vec(read_ap.get_processed_docs(), batched=True)
