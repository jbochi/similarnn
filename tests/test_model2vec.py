import falcon
import hug
import os
import numpy as np

os.environ['CONFIG_PATH'] = 'tests/data/config.toml'
from similarnn import model2vec


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


def test_topics_method_not_allowed():
    assert '405 Method Not Allowed' == \
        hug.test.get(model2vec, "models/invalidmodel/topics").status


def test_topics_model_not_found():
    assert '404 Not Found' == \
        hug.test.put(model2vec, "models/invalidmodel/topics").status


def test_topics():
    url = "models/lda/topics"
    document = {"body": "frango"}
    response = hug.test.put(model2vec, url, body=document)
    assert '200 OK' == response.status

    expected_topics = model2vec.config['models']['lda'].infer_topics(document)
    assert np.allclose(expected_topics, response.data, rtol=1e-3)
