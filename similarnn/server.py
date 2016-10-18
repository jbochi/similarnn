from functools import wraps
import hug
import json
import os

from similarnn.config import load_config
from similarnn.storage import get_model_db


config = load_config(os.environ.get("CONFIG_PATH", "config.toml"))


def validate_model(f):
    @wraps(f)
    def wrap(model, response, **kwargs):
        if model not in config['models']:
            response.status = hug.HTTP_NOT_FOUND
            return {
                "error": "Model {model} not found".format(model=model),
                "available_models": config['models'].keys()
            }
        return f(config['models'][model], response, **kwargs)
    return wrap


@hug.get('/models/{model}/topics')
@validate_model
def num_topics(model, response, **kwargs):
    """Returns the number of topics in a model"""
    return { "topics":  model.num_topics }


@hug.post('/models/{model}/documents')
@validate_model
def create_document(model, response, **kwargs):
    """Adds document and returns the number of topics in a model"""
    document = kwargs
    vector = model.infer_topics(document=document)
    storage = get_model_db(model)
    storage.add_item(str(document['id']), vector)
    return vector.tolist()


@hug.delete('/models/{model}/documents')
@validate_model
def delete_all_documents(model, response, **kwargs):
    """Removes all documents"""
    storage = get_model_db(model)
    storage.clean()


@hug.get('/models/{model}/documents')
@validate_model
def vector_knn_documents(model, response, **kwargs):
    """Get vector KNN documents"""
    storage = get_model_db(model)
    if 'vector' not in kwargs:
        return  {}
    vector = map(float, kwargs['vector'].split(","))
    return _similar_json(storage.vector_knn(vector))


@hug.get('/models/{model}/documents/{document_id}')
@validate_model
def get_document(model, response, document_id):
    """Get document vector"""
    storage = get_model_db(model)
    try:
        vector = storage.item_vector(document_id)
        return vector.tolist()
    except KeyError:
        response.status = hug.HTTP_NOT_FOUND
        return {
            "error": "Document {document_id} not found".format(document_id=document_id)
        }


@hug.delete('/models/{model}/documents/{document_id}')
@validate_model
def delete_document(model, response, document_id):
    """Removes document by id"""
    storage = get_model_db(model)
    try:
        vector = storage.remove_item(str(document_id))
    except KeyError:
        response.status = hug.HTTP_NOT_FOUND
        return {
            "error": "Document {document_id} not found".format(document_id=document_id)
        }


@hug.get('/models/{model}/documents/{document_id}/similar')
@validate_model
def similar_documents(model, response, document_id):
    """Get similar documents"""
    storage = get_model_db(model)
    try:
        return _similar_json(storage.item_knn(document_id))
    except KeyError:
        response.status = hug.HTTP_NOT_FOUND
        return {
            "error": "Document {document_id} not found".format(document_id=document_id)
        }


def _similar_json(items):
    return {
        "similar": [{"key": key, "distance": distance}
            for key, distance in items]
    }
