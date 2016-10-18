from similarnn.storage.ann import NearestNeighbours

storage = {}


def get_model_db(model):
    global storage
    if model.name in storage:
        return storage[model.name]
    storage[model.name] = NearestNeighbours(model.num_topics)
    return storage[model.name]
