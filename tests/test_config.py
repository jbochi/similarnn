import gensim

from similarnn import config
from similarnn import models


def test_load_dictionaries():
    dictionaries = config.load_dictionaries({
        "dictionaries": {
            "mydict": {
                "path": "tests/data/model/dictionary.dict"
            }
        }})
    assert list(dictionaries.keys()) == ["mydict"]
    assert type(dictionaries["mydict"]) == gensim.corpora.dictionary.Dictionary


def test_load_corpora():
    corpora = config.load_corpora({
        "corpora": {
            "corpus": {
                "path": "tests/data/model/corpus.mm"
            }
        }})
    assert list(corpora.keys()) == ["corpus"]
    assert type(corpora["corpus"]) == gensim.corpora.mmcorpus.MmCorpus


def test_load_models():
    test_config = {
        "dictionaries": {
            "mydictionary": gensim.corpora.Dictionary.load_from_text(
                "tests/data/model/dictionary.dict")
        },
        "corpora": {
            "mycorpus": gensim.corpora.MmCorpus("tests/data/model/corpus.mm")
        },
        "models": {
            "model": {
                "path": "tests/data/model/lda/model",
                "dictionary": "mydictionary",
                "corpus": "mycorpus"
    }}}
    loaded_models  = config.load_models(test_config)
    assert list(loaded_models.keys()) == ["model"]
    assert type(loaded_models["model"]) == models.LDAModel


def test_load_config():
    loaded_config = config.load_config("tests/data/config.toml")
    assert type(loaded_config['models']['lda']) == models.LDAModel
