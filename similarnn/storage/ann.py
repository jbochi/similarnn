from annoy import AnnoyIndex


class NearestNeighbours():
    def __init__(self, num_factors, n_trees=10):
        self.clean()
        self.num_factors = num_factors
        self.n_trees = n_trees

    def clean(self):
        self.vectors = []
        self.key_from_id = {}
        self.id_from_key = {}
        self.index = None

    def add_item(self, key, vector):
        "Add or update item"
        if key in self.id_from_key:
            self._update_vector(key, vector)
        else:
            self._add_new_vector(key, vector)
        self._rebuild_index()

    def item_knn(self, key, k=10):
        "K nearest neighbours from item"
        item_id = self.id_from_key[key]
        items, distances = self.index.get_nns_by_item(item_id,
            n=k + 1,
            include_distances=True)
        cosine_distances = map(self._euclidean_from_cosine_distance, distances)
        print(items)
        return [(self.key_from_id[item], distance) for item, distance in
            zip(items, cosine_distances) if item != item_id][:k]

    @property
    def n_items(self):
        return len(self.vectors)

    def item_vector(self, key):
        return self.vectors[self.id_from_key[key]]

    def _euclidean_from_cosine_distance(self, distance):
        # Annoy uses Euclidean distance of normalized vectors for its angular
        # distance, which for two vectors u,v is equal to sqrt(2(1-cos(u,v)))
        return (distance ** 2) / 2

    def _rebuild_index(self):
        index = AnnoyIndex(self.num_factors, metric='angular')
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
