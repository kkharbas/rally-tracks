import base64
import copy
import logging
import os

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_QUERY_IMAGES = []
for _fname in sorted(os.listdir(_DATA_DIR)):
    if _fname.lower().endswith((".jpeg", ".jpg", ".png")):
        with open(os.path.join(_DATA_DIR, _fname), "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode("ascii")
        _QUERY_IMAGES.append(f"data:image/jpeg;base64,{_b64}")


class SemanticSearchParamSource:
    """
    Cycles through sample query images and issues knn searches against the
    semantic image field using query_vector_builder so the inference endpoint
    generates the query embedding at search time.
    Each client partition starts at a different offset to vary queries.
    """

    def __init__(self, track, params, **kwargs):
        self._index = params.get("index", "semantic-image-test")
        self._size = params.get("size", 10)
        self._inference_id = params.get("inference_id", ".jina-embeddings-v5-omni-small")
        self._counter = 0
        self.infinite = True

    def partition(self, partition_index, total_partitions):
        source = copy.copy(self)
        source._counter = partition_index
        return source

    def params(self):
        data_uri = _QUERY_IMAGES[self._counter % len(_QUERY_IMAGES)]
        self._counter += 1
        return {
            "index": self._index,
            "size": self._size,
            "body": {
                "size": self._size,
                "query": {
                    "knn": {
                        "field": "image",
                        "num_candidates": self._size * 4,
                        "query_vector_builder": {
                            "embedding": {
                                "inference_id": self._inference_id,
                                "input": {
                                    "type": "image",
                                    "value": data_uri,
                                },
                            }
                        },
                    }
                },
            },
        }


class SemanticSearchRunner:
    """
    Executes one semantic search per invocation.  Returns hit count and
    server-side latency as custom metrics; failures are logged and counted.
    """

    async def __call__(self, es, params):
        success = True
        hits = 0
        took = 0

        try:
            response = await es.search(
                body=params["body"],
                index=params["index"],
                request_cache=False,
                request_timeout=30,
            )
            hits = response["hits"]["total"]["value"]
            took = response["took"]
        except Exception as exc:
            logger.error(
                "Search request failed (%s): %s", type(exc).__name__, exc
            )
            success = False

        return {
            "weight": 1,
            "unit": "ops",
            "success": success,
            "hits": hits,
            "took": took,
        }

    def __repr__(self):
        return "semantic-image-search"


def register(registry):
    registry.register_param_source("semantic-search-param-source", SemanticSearchParamSource)
    registry.register_runner("semantic-image-search", SemanticSearchRunner(), async_runner=True)
