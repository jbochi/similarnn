import falcon
import hug
import numpy as np
import os
import pytest

os.environ['CONFIG_PATH'] = 'tests/data/config.toml'


from similarnn import server
from similarnn.storage import get_model_db


@pytest.fixture
def db():
    db = get_model_db(server.config['models']['lda'])
    db.clean()
    return db


def test_num_topics_model_not_found():
    assert '404 Not Found' == \
        hug.test.get(server, "models/invalidmodel/topics").status


def test_topics():
    response = hug.test.get(server, "models/lda/topics")
    assert '200 OK' == response.status
    assert 10 == response.data['topics']


def test_post_documents_method_not_allowed():
    assert '405 Method Not Allowed' == \
        hug.test.get(server, "models/invalidmodel/documents").status


def test_post_documents_model_not_found():
    assert '404 Not Found' == \
        hug.test.post(server, "models/invalidmodel/documents").status


def test_post_documents():
    db = get_model_db(server.config['models']['lda'])
    assert db.n_items == 0

    url = "models/lda/documents"
    document = {"id": "document1", "body": "frango"}
    response = hug.test.post(server, url, body=document)
    assert '200 OK' == response.status

    expected_topics = server.config['models']['lda'].infer_topics(document)
    assert np.allclose(expected_topics, response.data, rtol=1e-3)
    assert db.n_items == 1


def test_get_document_404():
    response = hug.test.get(server, 'models/lda/documents/doc2')
    assert '404 Not Found' == response.status


def test_get_document(db):
    topics = np.array(range(10))
    db.add_item("doc1", topics)

    response = hug.test.get(server, 'models/lda/documents/doc1')
    assert '200 OK' == response.status
    assert np.allclose(topics, response.data, rtol=1e-3)


def test_get_document_404():
    response = hug.test.get(server, 'models/lda/documents/doc2/similar')
    assert '404 Not Found' == response.status


def test_get_document(db):
    topics = np.array(range(10))
    db.add_item("doc1", topics)
    db.add_item("doc2", topics)

    response = hug.test.get(server, 'models/lda/documents/doc1/similar')
    assert '200 OK' == response.status
    assert ['similar'] == list(response.data.keys())
    assert 1 == len(response.data['similar'])
    assert 'doc2' == response.data['similar'][0]['key']
    assert 0 == response.data['similar'][0]['distance']


def test_delete_all_documents(db):
    topics = np.array(range(10))
    db.add_item("doc1", topics)
    db.add_item("doc2", topics)

    assert db.n_items == 2

    response = hug.test.delete(server, 'models/lda/documents')
    assert '200 OK' == response.status
    assert db.n_items == 0


def test_delete_document_404():
    response = hug.test.delete(server, 'models/lda/documents/not_found')
    assert '404 Not Found' == response.status


def test_delete_document(db):
    db.add_item("doc1", np.array(range(10)))

    response = hug.test.delete(server, 'models/lda/documents/doc1')
    assert '200 OK' == response.status
    assert db.n_items == 0
