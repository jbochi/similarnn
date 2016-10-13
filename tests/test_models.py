import gensim

from similarnn import models

def test_lda_model():
    dictionary = gensim.corpora.Dictionary.load_from_text(
        "tests/data/model/dictionary.dict")
    corpus = gensim.corpora.MmCorpus("tests/data/model/corpus.mm")
    model = models.LDAModel("tests/data/model/lda/model", dictionary, corpus)

    assert model.num_topics == 10
    assert len(model.infer_topics({"body": "carne panela"})) == 10


def test_doc2bow():
    dictionary = gensim.corpora.Dictionary.load_from_text(
        "tests/data/model/dictionary.dict")
    document = {
        "body": "frango com arroz",
        "title": "frango delicioso"
    }
    bow = dict(models.doc2bow(dictionary, document))
    words2ids = {v: k for k, v in dictionary.items()}
    assert len(bow) == 2
    assert bow[words2ids["frango"]] == 2
    assert bow[words2ids["arroz"]] == 1
