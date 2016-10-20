from similarnn.storage.ann import NearestNeighbours


class Storage(object):
    def __init__(self):
        self.storage = {}

    def get_model_db(self, model):
        if model.name in self.storage:
            return self.storage[model.name]
        self.storage[model.name] = NearestNeighbours(model.num_topics)
        return self.storage[model.name]
