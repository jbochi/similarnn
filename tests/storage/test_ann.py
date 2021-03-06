import math
import pytest

from similarnn.storage.ann import NearestNeighbours


def close(a, b, tol=1e-3):
    return abs(a - b) < tol


def test_n_items():
    space = NearestNeighbours(n_factors=2)
    assert 0 == space.n_items

    space.add_item("doc1", [0, 1])
    space.add_item("doc2", [1, 0])
    space.add_item("doc3", [1, 1])
    assert 3 == space.n_items


def test_clean():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [0, 1])
    space.add_item("doc2", [1, 0])
    space.add_item("doc3", [1, 1])
    space.clean()
    assert 0 == space.n_items


def test_item_vector():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [0, 1])
    assert [0, 1] == space.item_vector("doc1")


def test_item_vector_unknown():
    space = NearestNeighbours(n_factors=2)
    with pytest.raises(KeyError):
        space.item_vector("doc2")


def test_item_knn():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [0, 1])
    space.add_item("doc2", [1, 0])
    space.add_item("doc3", [1, 1])

    docs = [doc for doc, _ in space.item_knn("doc1", k=2)]
    assert ['doc3', 'doc2'] == docs


def test_item_knn_45_degrees():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [0, 1])
    space.add_item("doc2", [1, 0])
    space.add_item("doc3", [0.5, 0.5])

    docs = space.item_knn("doc1", k=1)
    assert 1 == len(docs)

    similar_doc, distance = docs[0]
    assert 'doc3' == similar_doc
    assert close(1 - math.sqrt(2) / 2, distance)


def test_item_knn_90_degrees():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [0, 1])
    space.add_item("doc2", [1, 0])

    docs = space.item_knn("doc1", k=1)
    assert 1 == len(docs)

    similar_doc, distance = docs[0]
    assert 'doc2' == similar_doc
    assert close(1, distance)


def test_item_knn_0_degrees():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [0, 1])
    space.add_item("doc2", [0, 1])

    docs = space.item_knn("doc1", k=1)
    assert 1 == len(docs)

    similar_doc, distance = docs[0]
    assert 'doc2' == similar_doc
    assert close(0, distance)


def test_vector_with_no_index():
    space = NearestNeighbours(n_factors=2)
    doc_and_distances = space.vector_knn([1, 1])
    assert [] == doc_and_distances


def test_vector_knn():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [-1, -1])
    space.add_item("doc2", [1, 0])
    space.add_item("doc3", [1, 1])

    doc_and_distances = space.vector_knn([1, 1])
    docs = [doc for doc, _ in doc_and_distances]
    distances = [distance for _, distance in doc_and_distances]
    assert ['doc3', 'doc2', 'doc1'] == docs
    assert close(0, distances[0])
    assert close(1 - math.sqrt(2) / 2, distances[1])
    assert close(2, distances[2])


def test_remove_item():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [0, 1])
    space.add_item("doc2", [1, 1])
    space.add_item("doc3", [1, 0])

    assert 3 == space.n_items
    space.remove_item("doc2")
    assert 2 == space.n_items

    with pytest.raises(KeyError):
        space.item_vector("doc2")

    assert [0, 1] == space.item_vector("doc1")
    assert [1, 0] == space.item_vector("doc3")

    docs = space.item_knn("doc1", k=2)
    assert 1 == len(docs)
    similar_doc, distance = docs[0]
    assert 'doc3' == similar_doc
    assert close(1, distance)


def test_remove_item_unknown():
    space = NearestNeighbours(n_factors=2)
    with pytest.raises(KeyError):
        space.remove_item("doc2")


def test_update_vector():
    space = NearestNeighbours(n_factors=2)
    space.add_item("doc1", [0, 1])

    space.add_item("doc2", [1, 0])
    space.add_item("doc2", [0, 1])

    assert [0, 1] == space.item_vector("doc2")
    doc, distance = space.item_knn("doc1", k=2)[0]
    assert "doc2" == doc
    assert close(0, distance)
