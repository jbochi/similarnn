import math

from similarnn import ann


def close(a, b, tol=1e-3):
    return abs(a - b) < tol

def test_knn():
    space = ann.NearestNeighbours(num_factors=2)
    space.add_vector("doc1", [0, 1])
    space.add_vector("doc2", [1, 0])
    space.add_vector("doc3", [1, 1])

    docs = [doc for doc, _ in space.item_knn("doc1", k=2)]
    assert ['doc3', 'doc2'] == docs


def test_knn_45_degrees():
    space = ann.NearestNeighbours(num_factors=2)
    space.add_vector("doc1", [0, 1])
    space.add_vector("doc2", [1, 0])
    space.add_vector("doc3", [0.5, 0.5])

    docs = space.item_knn("doc1", k=1)
    assert 1 == len(docs)

    similar_doc, distance = docs[0]
    assert 'doc3' == similar_doc
    assert close(1 - math.sqrt(2) / 2, distance)


def test_knn_90_degrees():
    space = ann.NearestNeighbours(num_factors=2)
    space.add_vector("doc1", [0, 1])
    space.add_vector("doc2", [1, 0])

    docs = space.item_knn("doc1", k=1)
    assert 1 == len(docs)

    similar_doc, distance = docs[0]
    assert 'doc2' == similar_doc
    assert close(1, distance)


def test_knn_0_degrees():
    space = ann.NearestNeighbours(num_factors=2)
    space.add_vector("doc1", [0, 1])
    space.add_vector("doc2", [0, 1])

    docs = space.item_knn("doc1", k=1)
    assert 1 == len(docs)

    similar_doc, distance = docs[0]
    assert 'doc2' == similar_doc
    assert close(0, distance)
