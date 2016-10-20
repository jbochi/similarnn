from functools import wraps
import hug
import os

from similarnn.config import load_config
from similarnn.storage import Storage


config = load_config(os.environ.get("CONFIG_PATH", "config.toml"))
storage = Storage()


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
    return {"topics":  model.num_topics}


@hug.post('/models/{model}/documents')
@validate_model
def create_document(model, response, **kwargs):
    """Adds document and returns the number of topics in a model"""
    document = kwargs
    vector = model.infer_topics(document=document)
    db = storage.get_model_db(model)
    db.add_item(str(document['id']), vector)
    return vector.tolist()


@hug.delete('/models/{model}/documents')
@validate_model
def delete_all_documents(model, response, **kwargs):
    """Removes all documents"""
    db = storage.get_model_db(model)
    db.clean()


@hug.get('/models/{model}/documents')
@validate_model
def vector_knn_documents(model, response, vector=None):
    """Get vector KNN documents"""
    db = storage.get_model_db(model)
    if vector is None:
        return {}
    vector = map(float, vector.split(","))
    return _similar_json(db.vector_knn(vector))


@hug.get('/models/{model}/documents/{document_id}')
@validate_model
def get_document(model, response, document_id):
    """Get document vector"""
    db = storage.get_model_db(model)
    try:
        vector = db.item_vector(document_id)
        return vector.tolist()
    except KeyError:
        response.status = hug.HTTP_NOT_FOUND
        return {
            "error": "Document {document_id} not found".format(
                document_id=document_id)}


@hug.delete('/models/{model}/documents/{document_id}')
@validate_model
def delete_document(model, response, document_id):
    """Removes document by id"""
    db = storage.get_model_db(model)
    try:
        db.remove_item(str(document_id))
    except KeyError:
        response.status = hug.HTTP_NOT_FOUND
        return {
            "error": "Document {document_id} not found".format(
                document_id=document_id)}


@hug.get('/models/{model}/documents/{document_id}/similar')
@validate_model
def similar_documents(model, response, document_id, k: hug.types.number=10):
    """Get similar documents"""
    db = storage.get_model_db(model)
    try:
        return _similar_json(db.item_knn(document_id, k=k))
    except KeyError:
        response.status = hug.HTTP_NOT_FOUND
        return {
            "error": "Document {document_id} not found".format(
                document_id=document_id)}


def _similar_json(items):
    return {"similar": [{
        "key": key,
        "distance": distance
    } for key, distance in items]}
