from similarnn.storage import MemoryStorage, RedisStorage


class DummyModel(object):
    name = 'dummy'
    num_topics = 3


def test_storage():
    s = MemoryStorage()
    model = DummyModel()
    db = s.get_model_db(model)

    assert db.n_items == 0
    db.add_item('doc1', [1, 2, 3])
    db.add_item('doc2', [1, 1, 1])

    db2 = s.get_model_db(model)
    assert db2.n_items == 2


def test_redis_storage():
    s = RedisStorage()
    model = DummyModel()
    db = s.get_model_db(model)
    db.clean()
    db.add_item('doc1', [1, 2, 3])
    db.add_item('doc2', [1, 1, 1])

    s2 = RedisStorage()
    db2 = s2.get_model_db(model)
    assert db2.n_items == 2
