"""
A word2vec implementation by Fabrizio and Jochem
"""

import torch
import pickle
from collections import Counter

# Own libraries
import download_ap
import read_ap
import utils

# Pickle storage path
PROCESSED_DOCS_PATH = "processed_docs_SkipGram.pkl"

# Tunable hyperparameters
OCCURANCE_THRESHHOLD = 50
EMBEDDING_SIZE = 100
CONTEXT_WINDOW = 2


class SkipGram(torch.nn.Module):

    def __init__(self, V, embedding_size):
        super(SkipGram, self).__init__()

        self.out_size = embedding_size
        self.in_size = V
        self.tok2id = None

        if type(V) is dict:
            self.in_size = len(V)
            self.tok2id = V

        self.embeddings = torch.nn.Embedding(self.in_size, self.out_size)

    def forward(self, center_id, context_id):
        center_emb = self.embeddings(center_id)
        context_emb = self.embeddings(context_id)
        out = torch.mm(center_emb, context_emb.T)
        return torch.nn.functional.logsigmoid(out)

    def train(self, ):
        pass


def filter_infrequent(path=PROCESSED_DOCS_PATH, docs_by_id=None):
    """Load or create a dictionary similar to docs_by_id, but without any word
    occurring less then 'OCCURANCE_THRESHHOLD' times.
    """

    # Load the preprocessed docs_by_id file if it exists.
    if os.path.exists(path):
        with open(path, "rb") as reader:
            return pickle.load(reader)

    # Count all tokens.
    counter = Counter()
    for tokens in docs_by_id.values():
        counter.update(tokens)

    # Filter any token that doesn't meet the frequency threshhold.
    filtered_docs = {}
    for doc_id, tokens in docs_by_id.items():
        filtered_doc = []
        for token in tokens:
            if counter[token] >= OCCURANCE_THRESHHOLD:
                filtered_doc.append(token)
        filtered_docs[doc_id] = torch.LongTensor(filtered_doc)

    # Store the new filtered docs_by_id dictionary for faster future retrieval.
    with open(path, "wb") as writer:
        pickle.dump(filtered_docs, writer)
    return filtered_docs


if __name__ == "__main__":
    download_ap.download_dataset()
    docs_by_id = read_ap.get_processed_docs()
    docs_by_id = filter_infrequent(docs_by_id=docs_by_id)

