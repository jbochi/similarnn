import os
import redis

from .ann import NearestNeighbours
from .redis_ann import RedisNearestNeighbours


class MemoryStorage(object):
    def __init__(self):
        self.storage = {}

    def get_model_db(self, model):
        if model.name in self.storage:
            return self.storage[model.name]
        self.storage[model.name] = self._create_model_db(model)
        return self.storage[model.name]

    def _create_model_db(self, model):
        return NearestNeighbours(model.num_topics)


class RedisStorage(MemoryStorage):
    def __init__(self, sync_interval=None):
        self.sync_interval = sync_interval
        super().__init__()

    def _create_model_db(self, model):
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = os.getenv('REDIS_PORT', 6379)
        redis_conn = redis.StrictRedis(redis_host, redis_port)
        return RedisNearestNeighbours(model.num_topics,
                                      redis_conn=redis_conn,
                                      namespace=model.name,
                                      sync_interval=self.sync_interval)
