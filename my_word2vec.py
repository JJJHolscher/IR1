"""
A word2vec implementation by Fabrizio and Jochem
"""

import os
import random
import torch
import pickle
from collections import Counter

# Own libraries
import download_ap
import read_ap
import utils

# Pickle storage path
PROCESSED_DOCS_PATH = "processed_docs_SkipGram.pkl"

##### Tunable hyperparameters #####
# The minimum frequency a word should have for it to get an embedding.
OCCURANCE_THRESHHOLD = 50
# The number of features in any single word embedding.
EMBEDDING_SIZE = 100
# The number of words before or after a word that are considered its context.
CONTEXT_WINDOW = 2
#
BATCH_SIZE = 100


class SkipGram(torch.nn.Module):

    def __init__(self, tok2idx, embedding_size=EMBEDDING_SIZE):
        super(SkipGram, self).__init__()
        self.tok2idx = tok2idx
        self.embeddings = torch.nn.Embedding(len(self.tok2idx), embedding_size)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, center_id, context_id):
        center_emb = self.embeddings(center_id)
        context_emb = self.embeddings(context_id)
        out = torch.mm(center_emb, context_emb.T)
        return torch.nn.functional.logsigmoid(out)

    def train(self, dataset):
        for center, context, target in dataset:
            out = self(center, context)[0]
            self.loss_func(out, target)



def all_words_to_indices(docs_by_id):
    """ Create a corpus where all string representations of words are replaced
    by index representations.
    """
    corpus = []
    tok2idx = {}

    for doc in docs_by_id.values():
        doc_repr = []

        for token in doc:
            # Create a new index if the token has not occured before.
            if not (token in tok2idx):
                tok2idx[token] = len(tok2idx) + 1
            doc_repr.append(tok2idx[token])

        if len(doc_repr) > 1:
            corpus.append(torch.LongTensor(doc_repr))

    return torch.LongTensor(corpus), tok2idx


def preprocess(path=PROCESSED_DOCS_PATH,):
    # Load the preprocessed docs_by_id file if it exists.
    if os.path.exists(path):
        with open(path, "rb") as reader:
            return pickle.load(reader)

    # (Down)load the dataset from the ap files and get it in the right form.
    download_ap.download_dataset()
    docs_by_id = read_ap.get_processed_docs()
    docs_by_id = filter_infrequent(docs_by_id)
    corpus, tok2idx = all_words_to_indices(docs_by_id)
    dataset = sample(corpus)

    # Store the preprocessing results for faster future retrieval.
    with open(path, "wb") as writer:
        pickle.dump((corpus, tok2idx), writer)
    return corpus, tok2idx


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

            for fake_doc_repr in random.choices(corpus, k=2 * context_window):
                fake_context = random.choice(fake_doc_repr)
                dataset.append(torch.Tensor([token, fake_context, 0]))

            # positive sampling
            for context_idx in range(min_context_idx, i):
                dataset.append(torch.Tensor[token, doc_repr[context_idx], 1])

            if i == len(doc_repr) - 1: continue

            for context_idx in range(i + 1, max_context_idx:
                dataset.append(torch.Tensor[token, doc_repr[context_idx], 1])

    return dataset


if __name__ == "__main__":
    docs_by_id = filter_infrequent()
    tok2idx = Tok2Idx(docs_by_id)


