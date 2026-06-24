import base64
import copy
import logging
import os

logger = logging.getLogger(__name__)

_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_VIDEO_DATA_URIS = []

for _fname in sorted(os.listdir(_DATA_DIR)):
    _ext = os.path.splitext(_fname.lower())[1]
    if _ext in _MIME_TYPES:
        _fpath = os.path.join(_DATA_DIR, _fname)
        with open(_fpath, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode("ascii")
        _VIDEO_DATA_URIS.append(f"data:{_MIME_TYPES[_ext]};base64,{_b64}")
        logger.info("Loaded video file: %s (%d bytes base64)", _fname, len(_b64))

if not _VIDEO_DATA_URIS:
    raise FileNotFoundError(
        f"No video files found in '{_DATA_DIR}'. "
        "Place one or more .mp4/.webm/.mov/.avi/.mkv files there before running."
    )


class SemanticVideoIndexParamSource:
    """
    Generates bulk-index request bodies from video files in the data/ directory.
    Each call produces `bulk_size` documents, cycling through the available
    videos. Each partition starts at an offset to avoid duplicate document names
    across concurrent clients.
    """

    def __init__(self, track, params, **kwargs):
        self._index = params.get("index", "semantic-video-test")
        self._bulk_size = int(params.get("bulk_size", 1))
        self._request_timeout = int(params.get("request_timeout", 300))
        self._counter = 0
        self.infinite = True

    def partition(self, partition_index, total_partitions):
        source = copy.copy(self)
        # Offset each partition so document names never collide
        source._counter = partition_index * 10_000_000
        return source

    def params(self):
        bulk_body = []
        for _ in range(self._bulk_size):
            data_uri = _VIDEO_DATA_URIS[self._counter % len(_VIDEO_DATA_URIS)]
            bulk_body.append({"index": {"_index": self._index}})
            bulk_body.append({"name": f"video-{self._counter}", "video": data_uri})
            self._counter += 1
        return {
            "index": self._index,
            "bulk_size": self._bulk_size,
            "bulk_body": bulk_body,
            "request_timeout": self._request_timeout,
        }


class SemanticVideoBulkRunner:
    """
    Executes one bulk-index call per invocation against the semantic video index.
    Counts per-item successes and failures from the bulk response, and surfaces
    server-side took as a custom metric.
    """

    async def __call__(self, es, params):
        successes = 0
        failures = 0
        took = 0

        try:
            response = await es.bulk(
                body=params["bulk_body"],
                index=params["index"],
                request_timeout=params.get("request_timeout", 300),
            )
            took = response.get("took", 0)
            for item in response.get("items", []):
                action = next(iter(item))
                if item[action].get("error"):
                    logger.warning("Document index error: %s", item[action]["error"])
                    failures += 1
                else:
                    successes += 1
        except Exception as exc:
            logger.error("Bulk indexing request failed (%s): %s", type(exc).__name__, exc)
            failures = params["bulk_size"]

        total = successes + failures
        return {
            "weight": total,
            "unit": "docs",
            "success": failures == 0,
            "took": took,
            "successes": successes,
            "failures": failures,
        }

    def __repr__(self):
        return "semantic-video-bulk-index"


def register(registry):
    registry.register_param_source(
        "semantic-video-index-param-source", SemanticVideoIndexParamSource
    )
    registry.register_runner(
        "semantic-video-bulk-index", SemanticVideoBulkRunner(), async_runner=True
    )
