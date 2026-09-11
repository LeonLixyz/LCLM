"""Bound Hub request volume for the pinned 0.35.3 bulk uploader."""
import json
import re
import sys
import threading
import time
from urllib.parse import urlparse

import requests
from huggingface_hub import HfApi, configure_http_backend

_lock = threading.Lock()
_next_request = 0.0
_blocked_until = 0.0


def retry_delay(headers):
    value = re.search(r"(?:^|;)\s*t=(\d+)", headers.get("RateLimit", ""))
    if value:
        return int(value.group(1)) + 3
    try:
        return max(1, int(headers.get("Retry-After", "300"))) + 3
    except ValueError:
        return 303


class HubSession(requests.Session):
    def send(self, request, **kwargs):
        global _next_request, _blocked_until
        if urlparse(request.url).hostname != "huggingface.co":
            return super().send(request, **kwargs)
        body = request.body
        position = body.tell() if hasattr(body, "tell") and hasattr(body, "seek") else None
        while True:
            # Reserve a distinct time slot so busy upload threads cannot starve
            # a commit or upload-mode request that has already been waiting.
            with _lock:
                slot = max(_next_request, _blocked_until, time.monotonic())
                _next_request = slot + 3.0
            while True:
                with _lock:
                    now = time.monotonic()
                    if _blocked_until > slot:
                        slot = max(_next_request, _blocked_until, now)
                        _next_request = slot + 3.0
                    delay = slot - now
                    if delay <= 0:
                        break
                time.sleep(min(delay, 1.0))
            response = super().send(request, **kwargs)
            if response.status_code != 429:
                return response
            delay = retry_delay(response.headers)
            with _lock:
                _blocked_until = max(_blocked_until, time.monotonic()+delay)
            print(json.dumps({"event": "hf_rate_limit_backoff", "seconds": delay}), flush=True)
            response.close()
            if position is not None:
                body.seek(position)


def install():
    configure_http_backend(backend_factory=HubSession)


class UploadAPI(HfApi):
    """Cache only repository transfer capability, never file lists or commits."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capabilities = {}
        self._capability_lock = threading.Lock()

    def repo_info(self, *args, **kwargs):
        if kwargs.get("expand") != "xetEnabled":
            return super().repo_info(*args, **kwargs)
        key = (kwargs.get("repo_id", args[0] if args else None), kwargs.get("repo_type"))
        with self._capability_lock:
            if key not in self._capabilities:
                self._capabilities[key] = super().repo_info(*args, **kwargs)
            return self._capabilities[key]


if __name__ == "__main__":
    import huggingface_hub
    import huggingface_hub._upload_large_folder as bulk
    if huggingface_hub.__version__ != "0.35.3":
        raise RuntimeError("Recheck upload batching before changing the pinned Hub version")
    # The pinned SDK otherwise makes a Hub capability/batch request per file.
    # Batch files to reduce API traffic while preserving its upload/checksum logic.
    bulk.UPLOAD_BATCH_SIZE_LFS = 16
    install()
    UploadAPI().upload_large_folder(**json.load(open(sys.argv[1])))
