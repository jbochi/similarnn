import falcon
import hug
import os
import numpy as np

os.environ['CONFIG_PATH'] = 'tests/data/config.toml'


from similarnn import model2vec
from similarnn.storage import get_model_db


def test_num_topics_model_not_found():
    assert '404 Not Found' == \
        hug.test.get(model2vec, "models/invalidmodel/num_topics").status



def test_num_topics():
    assert '200 OK' == \
        hug.test.get(model2vec, "models/lda/num_topics").status


def test_num_topics():
    url = "models/lda/num_topics"
    response = hug.test.get(model2vec, url)
    assert '200 OK' == response.status
    assert 'This model has 10 topics' == response.data


def test_post_documents_method_not_allowed():
    assert '405 Method Not Allowed' == \
        hug.test.get(model2vec, "models/invalidmodel/documents").status


def test_post_documents_model_not_found():
    assert '404 Not Found' == \
        hug.test.post(model2vec, "models/invalidmodel/documents").status


def test_post_documents():
    db = get_model_db(model2vec.config['models']['lda'])
    assert db.n_items == 0

    url = "models/lda/documents"
    document = {"id": "document1", "body": "frango"}
    response = hug.test.post(model2vec, url, body=document)
    assert '200 OK' == response.status

    expected_topics = model2vec.config['models']['lda'].infer_topics(document)
    assert np.allclose(expected_topics, response.data, rtol=1e-3)
    assert db.n_items == 1


def test_get_document():
    topics = np.array(range(10))
    db = get_model_db(model2vec.config['models']['lda'])
    db.add_item("doc1", topics)

    response = hug.test.get(model2vec, 'models/lda/documents/doc1')
    assert '200 OK' == response.status
    assert np.allclose(topics, response.data, rtol=1e-3)


def test_get_document_404():
    response = hug.test.get(model2vec, 'models/lda/documents/doc2')
    assert '404 Not Found' == response.status
