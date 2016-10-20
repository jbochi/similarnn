import redis
import pytest

from similarnn.storage.redis_ann import RedisNearestNeighbours


NAMESPACE = 'mod'


@pytest.fixture
def redis_conn():
    r = redis.StrictRedis()
    r.delete(NAMESPACE)
    return r


def test_items_are_persisted(redis_conn):
    space = RedisNearestNeighbours(n_factors=2,
                                   redis_conn=redis_conn,
                                   namespace=NAMESPACE)
    assert 0 == space.n_items

    space.add_item("doc1", [0, 1])
    space.add_item("doc2", [1, 0])
    space.add_item("doc3", [1, 1])
    assert 3 == space.n_items

    space = RedisNearestNeighbours(n_factors=2,
                                   redis_conn=redis_conn,
                                   namespace=NAMESPACE)
    assert 3 == space.n_items


def test_rnn_is_synced_different_keys(redis_conn):
    space1 = RedisNearestNeighbours(n_factors=2,
                                    redis_conn=redis_conn,
                                    namespace=NAMESPACE)
    space2 = RedisNearestNeighbours(n_factors=2,
                                    redis_conn=redis_conn,
                                    namespace=NAMESPACE)
    assert space1.is_synced()
    assert space2.is_synced()

    space1.add_item("doc1", [0, 1])

    assert space1.is_synced()
    assert not space2.is_synced()


def test_rnn_is_synced_different_values(redis_conn):
    space1 = RedisNearestNeighbours(n_factors=2,
                                    redis_conn=redis_conn,
                                    namespace=NAMESPACE)
    space2 = RedisNearestNeighbours(n_factors=2,
                                    redis_conn=redis_conn,
                                    namespace=NAMESPACE)
    assert space1.is_synced()
    assert space2.is_synced()

    space1.add_item("doc1", [0, 1])
    space2.add_item("doc1", [1, 1])

    assert not space1.is_synced()
    assert space2.is_synced()
