from similarnn.storage import Storage


def test_storage():
    s = Storage()

    class DummyModel(object):
        name = 'dummy'
        num_topics = 3

    model = DummyModel()
    db = s.get_model_db(model)
    assert db.n_items == 0
    db.add_item('doc1', [1, 2, 3])
    db.add_item('doc2', [1, 1, 1])

    db2 = s.get_model_db(model)
    assert db2.n_items == 2
