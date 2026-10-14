"""A local HTTP server that answers like FR-DOPAMS, built from its source (api/match_api.py, app.py).

Response bodies mirror the real ones: success bodies spread the service result next to ``data``
and ``errors``; failures are ``{"detail": ...}``. Gallery-person names and crime fields are included
in /search hits on purpose, so tests can prove the client discards them.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

SEARCH_HIT = {
    "rank": 1,
    "person_id": "CRIM000001",
    "name": "SENSITIVE GALLERY NAME",
    "similarity": 0.8123,
    "image": "media/CRIM000001/front.jpg",
    "crime_type": "SENSITIVE CRIME",
    "police_station": "SENSITIVE PS",
    "status": "Active",
}


def search_success(hits=(SEARCH_HIT,)) -> dict[str, Any]:
    result: dict[str, Any] = {"success": True, "results": list(hits), "threshold": 0.4}
    return {"success": True, "message": "Search completed.", "data": result["results"], "errors": [], **result}


def openapi_document(title: str = "Face Enrollment and Matching API", file_field: bool = True) -> dict[str, Any]:
    """The shape FastAPI 0.115 generates for ``POST /search`` with ``file: UploadFile = File(...)``."""
    ref = "#/components/schemas/Body_search_search_post"
    properties = {"file": {"type": "string", "format": "binary", "title": "File"}} if file_field else {}
    return {
        "openapi": "3.1.0",
        "info": {"title": title, "version": "2.0.0"},
        "paths": {
            "/health": {"get": {"responses": {"200": {}}}},
            "/search": {"post": {"requestBody": {"content": {"multipart/form-data": {"schema": {"$ref": ref}}}, "required": True}}},
            "/compare": {
                "post": {
                    "security": [{"APIKeyHeader": []}],
                    "requestBody": {
                        "content": {"multipart/form-data": {"schema": {"$ref": "#/components/schemas/Body_compare_compare_post"}}},
                        "required": True,
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "Body_search_search_post": {"type": "object", "properties": properties, "required": ["file"] if file_field else []},
                "Body_compare_compare_post": {
                    "type": "object",
                    "properties": {"file_a": {"type": "string", "format": "binary"}, "file_b": {"type": "string", "format": "binary"}},
                    "required": ["file_a", "file_b"],
                },
            }
        },
    }


API_KEY = "emulator-key-0123456789abcdef"


def compare_success(similarity: float = 0.63, threshold: float = 0.45) -> dict[str, Any]:
    """The body FR-DOPAMS ``POST /compare`` returns (api/compare_api.py, src/face_service.py)."""
    return {
        "success": True,
        "message": "Comparison completed.",
        "data": {
            "status": "compared", "similarity": similarity, "metric": "cosine", "similarity_range": [-1.0, 1.0],
            "same_person": similarity >= threshold, "threshold": threshold, "threshold_name": "PERSON_MATCH_THRESHOLD",
            "model": "buffalo_l", "embedding_dim": 512,
            "faces": {"a": {"face_width": 110, "face_height": 134, "detection_score": 0.9},
                      "b": {"face_width": 101, "face_height": 121, "detection_score": 0.87}},
        },
        "errors": [],
    }


def compare_rejection(**sides: str) -> dict[str, Any]:
    """422 body for e.g. ``compare_rejection(a="no_face", b="low_quality:too_small")``."""
    errors = []
    for side, spec in sorted(sides.items()):
        code, _, reason = spec.partition(":")
        errors.append({"side": side, "code": code, "reason": reason or None, "message": f"{code} message"})
    return {"success": False, "message": "One or both images could not be used for comparison.", "data": None,
            "errors": errors, "detail": "; ".join(f"{e['side']}: {e['code']}" for e in errors)}


class FrEmulator(BaseHTTPRequestHandler):
    openapi: Any = openapi_document()
    # script: list of (status, body, extra) consumed one per /search call; falls back to success.
    script: list = []
    requests: list[dict[str, Any]] = []
    delay: float = 0.0
    health_status: int = 200
    health_body: Any = {"status": "ok"}

    def log_message(self, *args):
        pass

    def _send(self, status: int, body: Any, content_type: str = "application/json") -> None:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    compare_script: list = []

    def _compare(self, body: bytes) -> None:
        parts = re.findall(
            rb'name="(\w+)"; filename="([^"]*)"\r\nContent-Type: ([^\r]*)\r\n\r\n(.*?)\r\n--[0-9a-f]{32}(?:--)?\r\n?', body, re.S
        )
        self.requests.append(
            {
                "method": "POST", "path": self.path, "headers": dict(self.headers),
                "fields": [p[0].decode() for p in parts], "filenames": [p[1].decode() for p in parts],
                "payloads": {p[0].decode(): p[3] for p in parts},
            }
        )
        if self.delay:
            time.sleep(self.delay)
        if self.headers.get("X-API-Key") != API_KEY:  # the real service checks this before reading the body
            self._send(401, {"success": False, "message": "Missing or invalid API key.", "data": None,
                             "errors": [{"side": None, "code": "unauthorized", "reason": None, "message": "x"}]})
            return
        if {p[0] for p in parts} != {b"file_a", b"file_b"}:
            self._send(422, {"detail": [{"type": "missing", "loc": ["body", "file_b"], "msg": "Field required", "input": None}]})
            return
        status, payload = self.compare_script.pop(0) if self.compare_script else (200, compare_success())
        self._send(status, payload)

    def do_GET(self):
        self.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
        if self.path == "/health":
            self._send(self.health_status, self.health_body)
        elif self.path == "/openapi.json":
            self._send(200, self.openapi)
        else:
            self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.path == "/compare":
            self._compare(body)
            return
        match = re.search(rb'name="(\w+)"; filename="([^"]*)"\r\nContent-Type: ([^\r]*)\r\n\r\n(.*)\r\n--[0-9a-f]+--\r\n$', body, re.S)
        self.requests.append(
            {
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "field": match.group(1).decode() if match else None,
                "filename": match.group(2).decode() if match else None,
                "part_type": match.group(3).decode() if match else None,
                "payload": match.group(4) if match else None,
            }
        )
        if self.delay:
            time.sleep(self.delay)
        if self.path != "/search":
            self._send(404, {"detail": "Not Found"})
            return
        if not match:  # FastAPI's reply when the multipart field is missing
            self._send(422, {"detail": [{"type": "missing", "loc": ["body", "file"], "msg": "Field required", "input": None}]})
            return
        status, payload = self.script.pop(0) if self.script else (200, search_success())
        self._send(status, payload)


def start_emulator():
    FrEmulator.script, FrEmulator.requests, FrEmulator.delay = [], [], 0.0
    FrEmulator.compare_script = []
    FrEmulator.health_status, FrEmulator.health_body = 200, {"status": "ok"}
    FrEmulator.openapi = openapi_document()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FrEmulator)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"
