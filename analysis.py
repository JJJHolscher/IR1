from my_word2vec import get_trained_doc2vec, SkipGram, search_SkipGram,\
                        search_doc2vec, preprocess


def search(query, doc2vec=False):
    if doc2vec:
        model = get_trained_doc2vec()
        return search_doc2vec(model, query)
    else:
        tok2idx, id2corpus = preprocess()
        model = SkipGram.create(tok2idx)
        return search_SkipGram(model, query, id2corpus=id2corpus)


if __name__ == '__main__':
    print(search(input("Type a query for which documents are returned: ")))

