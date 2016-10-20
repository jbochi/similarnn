from similarnn.storage.ann import NearestNeighbours


class Storage(object):
    def __init__(self):
        self.storage = {}

    def get_model_db(self, model):
        if model.name in self.storage:
            return self.storage[model.name]
        self.storage[model.name] = self._create_model_db(model)
        return self.storage[model.name]

    def _create_model_db(self, model):
        return NearestNeighbours(model.num_topics)
