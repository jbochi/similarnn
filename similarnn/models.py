import gensim
import re


class LDAModel():
    def __init__(self, name, lda_path, dictionary, corpus):
        self.name = name
        self.dictionary = dictionary
        self.corpus = corpus
        self.lda = gensim.models.ldamodel.LdaModel.load(lda_path)

    @property
    def num_topics(self):
        return self.lda.num_topics

    def infer_topics(self, document):
        bow = doc2bow(self.dictionary, document)
        sparse_topics = self.lda.get_document_topics(bow)
        nparray = gensim.matutils.sparse2full(sparse_topics, self.num_topics)
        return nparray.tolist()


def doc2bow(dictionary, document):
    text = " ".join(v for k, v in document.items() if k != 'id')
    words = re.findall("\w+", text.lower())
    return dictionary.doc2bow(words)
