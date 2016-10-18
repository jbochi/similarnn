import gensim
import toml

from similarnn.models import LDAModel


def load_dictionaries(config):
    dictionaries = {}
    for name, attributes in config["dictionaries"].items():
        path = attributes['path']
        dictionaries[name] = gensim.corpora.Dictionary.load_from_text(path)
    return dictionaries


def load_corpora(config):
    corpora = {}
    for name, attributes in config["corpora"].items():
        corpora[name] = gensim.corpora.MmCorpus(attributes["path"])
    return corpora


def load_models(config):
    models = {}
    for model, attributes in config["models"].items():
        path = attributes['path']
        dictionary = config['dictionaries'][attributes['dictionary']]
        corpus = config['corpora'][attributes['corpus']]
        models[model] = LDAModel(model, path, dictionary, corpus)
    return models


def load_config(config_path):
    with open(config_path) as f:
        config = toml.load(f)
    config['dictionaries'] = load_dictionaries(config)
    config['corpora'] = load_corpora(config)
    config['models'] = load_models(config)
    return config
