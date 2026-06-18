import base64
import copy
import logging
import mimetypes
import os

logger = logging.getLogger(__name__)


class ImageBulkIndexParamSource:
    """
    Generates bulk index requests from local image files.
    Both images are read once at init (as base64 data URIs) and cycled
    indefinitely so the operation can run for an arbitrary time-period.
    Each client partition gets a distinct document-ID space to avoid
    cross-client collisions.
    """

    def __init__(self, track, params, **kwargs):
        self._index = params.get("index", "semantic-image-test")
        self._bulk_size = params.get("bulk-size", 1)

        cwd = os.path.dirname(__file__)
        image_paths = params.get(
            "image-paths",
            [
                os.path.join(cwd, "data", "galaxy-1.jpeg"),
                os.path.join(cwd, "data", "galaxy-2.jpeg"),
            ],
        )

        self._images = []
        for path in image_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            name = os.path.splitext(os.path.basename(path))[0]
            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type:
                mime_type = "image/jpeg"
            with open(path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("utf-8")
            self._images.append(
                {
                    "name": name,
                    "data_uri": f"data:{mime_type};base64,{encoded}",
                }
            )
            logger.info(
                "Loaded image %s (%d bytes on disk, %d base64 chars)",
                path,
                os.path.getsize(path),
                len(encoded),
            )

        self._counter = 0
        self.infinite = True

    def partition(self, partition_index, total_partitions):
        """Each Rally client gets its own counter, offset by 10M IDs."""
        source = copy.copy(self)
        source._counter = partition_index * 10_000_000
        return source

    def params(self):
        bulk_body = []
        for _ in range(self._bulk_size):
            img = self._images[self._counter % len(self._images)]
            bulk_body.append({"index": {"_index": self._index, "_id": str(self._counter)}})
            bulk_body.append(
                {
                    "name": f"{img['name']}-{self._counter}",
                    "image": {
                        "type": "image",
                        "value": img["data_uri"],
                    },
                }
            )
            self._counter += 1
        return {
            "index": self._index,
            "bulk-body": bulk_body,
            "bulk-size": self._bulk_size,
        }


class ImageBulkIndexRunner:
    """
    Sends one bulk request per invocation and returns per-call success/failure
    counts as custom Rally metrics.  Errors are logged at ERROR level so they
    surface in the Rally log regardless of log-level settings.
    """

    async def __call__(self, es, params):
        bulk_body = params["bulk-body"]
        expected = params.get("bulk-size", len(bulk_body) // 2)
        docs_indexed = 0
        docs_failed = 0

        try:
            response = await es.bulk(body=bulk_body, request_timeout=300)
            if response.get("errors"):
                for item in response.get("items", []):
                    op_type = list(item.keys())[0]
                    result = item[op_type]
                    if result.get("status", 200) >= 400:
                        docs_failed += 1
                        error = result.get("error", {})
                        logger.error(
                            "Document index failed: id=%s status=%d type=%s reason=%s",
                            result.get("_id"),
                            result.get("status"),
                            error.get("type"),
                            error.get("reason"),
                        )
                    else:
                        docs_indexed += 1
            else:
                docs_indexed = len(response.get("items", []))
        except Exception as exc:
            logger.error(
                "Bulk request failed (%s): %s", type(exc).__name__, exc
            )
            docs_failed = expected

        return {
            "weight": docs_indexed + docs_failed,
            "unit": "docs",
            "success": docs_failed == 0,
            "docs-indexed": docs_indexed,
            "docs-failed": docs_failed,
        }

    def __repr__(self):
        return "bulk-index-images"


class SemanticSearchParamSource:
    """
    Cycles through a small set of text queries that describe typical
    astronomical image content.  The semantic field's inference endpoint
    will embed each query and compare against stored image embeddings.
    Each client partition starts at a different offset to vary queries.
    """

    _QUERIES = [
        "galaxy",
        "spiral galaxy",
        "nebula",
        "deep space",
        "stars",
        "cosmic dust",
        "milky way",
        "universe",
        "astronomy",
        "stellar",
    ]

    def __init__(self, track, params, **kwargs):
        self._index = params.get("index", "semantic-image-test")
        self._size = params.get("size", 10)
        self._counter = 0
        self.infinite = True

    def partition(self, partition_index, total_partitions):
        source = copy.copy(self)
        source._counter = partition_index
        return source

    def params(self):
        query = self._QUERIES[self._counter % len(self._QUERIES)]
        self._counter += 1
        return {
            "index": self._index,
            "size": self._size,
            "body": {
                "query": {
                    "semantic": {
                        "field": "image",
                        "query": query,
                    }
                },
                "size": self._size,
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
    registry.register_param_source("image-bulk-param-source", ImageBulkIndexParamSource)
    registry.register_runner("bulk-index-images", ImageBulkIndexRunner(), async_runner=True)
    registry.register_param_source("semantic-search-param-source", SemanticSearchParamSource)
    registry.register_runner("semantic-image-search", SemanticSearchRunner(), async_runner=True)
