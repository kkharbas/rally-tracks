import bz2
import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_K: int = 1000
SO_VECTOR_DIR = os.path.join(os.path.dirname(__file__), "..", "so_vector")
QUERIES_FILENAME = "queries.json.bz2"


def _load_queries() -> List:
    queries = []
    with bz2.open(os.path.join(SO_VECTOR_DIR, QUERIES_FILENAME), "r") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


class BruteForceSearchParamSource:
    """Param source for the dense_vector brute force KNN query."""

    def __init__(self, track, params, **kwargs):
        if len(track.indices) == 1:
            default_index = track.indices[0].name
        else:
            default_index = "_all"

        self._index_name = params.get("index", default_index)
        self._params = params
        self._queries = _load_queries()
        self._iters = 0
        self._maxIters = len(self._queries)
        self.infinite = True

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        query_vec = self._queries[self._iters]
        self._iters = (self._iters + 1) % self._maxIters
        k = self._params.get("k", DEFAULT_K)

        return {
            "index": self._index_name,
            "cache": self._params.get("cache", False),
            "size": k,
            "body": {
                "query": {
                    "dense_vector": {
                        "field": "titleVector",
                        "query_vector": query_vec,
                        "similarity_function": "dot_product"
                    }
                }
            },
        }


class BruteForceESQLParamSource:
    """Param source for the ESQL brute force KNN query."""

    def __init__(self, track, params, **kwargs):
        if len(track.indices) == 1:
            default_index = track.indices[0].name
        else:
            default_index = "_all"

        self._index_name = params.get("index", default_index)
        self._params = params
        self._queries = _load_queries()
        self._iters = 0
        self._maxIters = len(self._queries)
        self.infinite = True

    def partition(self, partition_index, total_partitions):
        return self

    def params(self):
        query_vec = self._queries[self._iters]
        self._iters = (self._iters + 1) % self._maxIters
        k = self._params.get("k", DEFAULT_K)

        query = (
            f"FROM {self._index_name} METADATA _score "
            '| WHERE KNN(titleVector, ?query, {"similarity_function": "dot_product"}) '
            f"| SORT _score DESC | LIMIT {k} "
        )

        return {
            "query": query,
            "body": {"params": [{"query": query_vec}]},
        }


def register(registry):
    registry.register_param_source("brute-force-dense-vector-param-source", BruteForceSearchParamSource)
    registry.register_param_source("brute-force-esql-knn-param-source", BruteForceESQLParamSource)
