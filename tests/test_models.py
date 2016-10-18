import gensim
import pytest

from similarnn import models


@pytest.fixture
def dictionary():
    path = "tests/data/model/dictionary.dict"
    return gensim.corpora.Dictionary.load_from_text(path)


def test_lda_model(dictionary):
    corpus = gensim.corpora.MmCorpus("tests/data/model/corpus.mm")
    lda_path = "tests/data/model/lda/model"
    model = models.LDAModel("lda", lda_path, dictionary, corpus)

    assert 10 == model.num_topics
    assert 10 == len(model.infer_topics({"body": "carne panela"}))


def test_doc2bow(dictionary):
    document = {
        "id": "frango",
        "body": "frango com arroz",
        "title": "frango delicioso"
    }
    bow = dict(models.doc2bow(dictionary, document))
    words2ids = {v: k for k, v in dictionary.items()}
    assert 2 == len(bow)
    assert 2 == bow[words2ids["frango"]]
    assert 1 == bow[words2ids["arroz"]]
