import pickle

from .ann import NearestNeighbours


class RedisNearestNeighbours(NearestNeighbours):
    def __init__(self, n_factors, redis_conn, namespace, **kwargs):
        super().__init__(n_factors, **kwargs)
        self.redis_conn = redis_conn
        self.namespace = namespace
        self.sync()

    def add_item(self, key, vector):
        self._set_key_on_redis(key, vector)
        super().add_item(key, vector)

    def remove_item(self, key):
        self._del_key_on_redis(key)
        super().remove_item(key)

    def is_synced(self):
        local, remote = self._local_keys(), dict(self._keys_from_redis())
        return local == remote

    def sync(self):
        self.clean()
        for key, vector in self._keys_from_redis():
            self._add_new_vector(key, vector)
        self._rebuild_index()

    def _local_keys(self):
        return {key: self.vectors[i] for key, i in self.id_from_key.items()}

    def _set_key_on_redis(self, key, vector):
        self.redis_conn.hset(self.namespace,
                             key.encode('utf-8'),
                             pickle.dumps(vector))

    def _del_key_on_redis(self, key):
        self.redis_conn.hdel(self.namespace, key)

    def _keys_from_redis(self):
        for key, vector in self.redis_conn.hgetall(self.namespace).items():
            yield key.decode('utf-8'), pickle.loads(vector)
