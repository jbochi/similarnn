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
    assert ["mydict"] == list(dictionaries.keys())
    assert gensim.corpora.dictionary.Dictionary == type(dictionaries["mydict"])


def test_load_corpora():
    corpora = config.load_corpora({
        "corpora": {
            "corpus": {
                "path": "tests/data/model/corpus.mm"
            }
        }})
    assert ["corpus"] == list(corpora.keys())
    assert gensim.corpora.mmcorpus.MmCorpus == type(corpora["corpus"])


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
    assert ["model"] == list(loaded_models.keys())
    assert models.LDAModel == type(loaded_models["model"])


def test_load_config():
    loaded_config = config.load_config("tests/data/config.toml")
    assert models.LDAModel == type(loaded_config['models']['lda'])
