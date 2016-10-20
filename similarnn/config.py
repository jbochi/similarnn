import gensim
import os
import toml

from similarnn.models import LDAModel
from similarnn import storage


def load_storage(config):
    if 'storage' not in config:
        return storage.MemoryStorage()
    klass = config['storage']['type']
    kwargs = config['storage']
    del kwargs['type']
    return getattr(storage, klass)(**kwargs)


def load_dictionaries(config, dirname=''):
    dictionaries = {}
    for name, attributes in config["dictionaries"].items():
        path = os.path.join(dirname, attributes['path'])
        dictionaries[name] = gensim.corpora.Dictionary.load_from_text(path)
    return dictionaries


def load_corpora(config, dirname=''):
    corpora = {}
    for name, attributes in config["corpora"].items():
        path = os.path.join(dirname, attributes['path'])
        corpora[name] = gensim.corpora.MmCorpus(path)
    return corpora


def load_models(config, dirname=''):
    models = {}
    for model, attributes in config["models"].items():
        path = os.path.join(dirname, attributes['path'])
        dictionary = config['dictionaries'][attributes['dictionary']]
        corpus = config['corpora'][attributes['corpus']]
        models[model] = LDAModel(model, path, dictionary, corpus)
    return models


def load_config(config_path):
    with open(config_path) as f:
        config = toml.load(f)
    dirname = os.path.dirname(config_path)
    config['storage'] = load_storage(config)
    config['dictionaries'] = load_dictionaries(config, dirname)
    config['corpora'] = load_corpora(config, dirname)
    config['models'] = load_models(config, dirname)
    return config
