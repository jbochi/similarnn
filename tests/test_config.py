import gensim

from similarnn import config
from similarnn import models
from similarnn import storage


def test_load_dictionaries():
    dictionaries = config.load_dictionaries({
        "dictionaries": {
            "mydict": {
                "path": "tests/data/model/dictionary.dict"
            }
        }})
    assert ["mydict"] == list(dictionaries.keys())
    assert isinstance(dictionaries["mydict"],
                      gensim.corpora.dictionary.Dictionary)


def test_load_corpora():
    corpora = config.load_corpora({
        "corpora": {
            "corpus": {
                "path": "tests/data/model/corpus.mm"
            }
        }})
    assert ["corpus"] == list(corpora.keys())
    assert isinstance(corpora["corpus"],
                      gensim.corpora.mmcorpus.MmCorpus)


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
                "corpus": "mycorpus"}}}
    loaded_models = config.load_models(test_config)
    assert ["model"] == list(loaded_models.keys())
    assert isinstance(loaded_models["model"], models.LDAModel)


def test_load_storage():
    loaded_storage = config.load_storage({
        "storage": {
            "type": "RedisStorage",
            "sync_interval": 0.5
        }
    })
    assert isinstance(loaded_storage, storage.RedisStorage)
    assert 0.5 == loaded_storage.sync_interval


def test_load_storage_with_no_config():
    loaded_storage = config.load_storage({})
    assert isinstance(loaded_storage, storage.MemoryStorage)


def test_load_config():
    loaded_config = config.load_config("tests/data/config.toml")
    assert isinstance(loaded_config['storage'], storage.MemoryStorage)
    assert isinstance(loaded_config['models']['lda'], models.LDAModel)
