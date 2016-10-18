from annoy import AnnoyIndex


class NearestNeighbours():
    def __init__(self, n_factors, n_trees=10):
        self.clean()
        self.n_factors = n_factors
        self.n_trees = n_trees

    def clean(self):
        "Cleans all documents"
        self.vectors = []
        self.key_from_id = {}
        self.id_from_key = {}
        self.index = None

    @property
    def n_items(self):
        "Returns the ammount of storage items"
        return len(self.vectors)

    def add_item(self, key, vector):
        "Add or update item"
        if key in self.id_from_key:
            self._update_vector(key, vector)
        else:
            self._add_new_vector(key, vector)
        self._rebuild_index()

    def remove_item(self, key):
        "Removes item"
        item_id = self.id_from_key[key]
        del self.key_from_id[item_id]
        new_ids = [(i, key) for i, (old_id, key) in enumerate(
            sorted(self.key_from_id.items()))]
        self.vectors.pop(item_id)
        self.id_from_key = dict((key, new_id) for new_id, key in new_ids)
        self.key_from_id = dict(new_ids)
        self._rebuild_index()

    def item_knn(self, key, k=10):
        "Returns K nearest neighbours from item"
        item_id = self.id_from_key[key]
        items, distances = self.index.get_nns_by_item(item_id,
                                                      n=k + 1,
                                                      include_distances=True)
        key_dists = self._get_items_keys_and_cosine_distances(items, distances)
        return [(k, d) for k, d in key_dists if k != key][:k]

    def vector_knn(self, vector, k=10):
        "Returns K nearest neighbours from vector"
        items, distances = self.index.get_nns_by_vector(vector,
                                                        n=k,
                                                        include_distances=True)
        return self._get_items_keys_and_cosine_distances(items, distances)

    def item_vector(self, key):
        "Returns item vector"
        return self.vectors[self.id_from_key[key]]

    def _get_items_keys_and_cosine_distances(self, items, distances):
        cosine_distances = map(self._euclidean_from_cosine_distance, distances)
        zipped = zip(items, cosine_distances)
        return [(self.key_from_id[i], d) for i, d in zipped]

    def _euclidean_from_cosine_distance(self, distance):
        # Annoy uses Euclidean distance of normalized vectors for its angular
        # distance, which for two vectors u,v is equal to sqrt(2(1-cos(u,v)))
        return (distance ** 2) / 2

    def _rebuild_index(self):
        index = AnnoyIndex(self.n_factors, metric='angular')
        for i, vector in enumerate(self.vectors):
            index.add_item(i, vector)
        index.build(self.n_trees)
        self.index = index

    def _update_vector(self, key, vector):
        self.vectors[self.id_from_key[key]] = vector

    def _add_new_vector(self, key, vector):
        vector_id = len(self.vectors)
        self.vectors.append(vector)
        self.id_from_key[key] = vector_id
        self.key_from_id[vector_id] = key
