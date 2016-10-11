import hug
import json
import codecs
import os
from similarnn.config import load_config


config = load_config(os.environ.get("CONFIG_PATH", "config.yaml"))


def validate_model(f):
    def wrap(model, response, **kwargs):
        if model not in config['models']:
            response.status = hug.HTTP_NOT_FOUND
            return {
                "error": "Model {model} not found".format(model=model),
                "available_models": config['models'].keys()
            }
        return f(config['models'][model], response, **kwargs)
    return wrap


@hug.get('/models/{model}/num_topics')
@validate_model
def num_topics(model, response, **kwargs):
    """Returns the number of topics in a model"""
    return "This model has {num_topics} topics".format(num_topics=model.num_topics)


@hug.put('/models/{model}/infer_topics')
@validate_model
def infer_topics(model, response, **kwargs):
    """Returns the number of topics in a model"""
    vector = model.infer_topics(document=kwargs)
    return vector.tolist()
