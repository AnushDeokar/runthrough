"""A small local web server for labelling judged conversations. Standard library only."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any

from pydantic import ValidationError

from runthrough.calibrate.items import JudgedItem, group_by_conversation
from runthrough.calibrate.labels import Label, LabelStore


def make_server(
    items: list[JudgedItem], store: LabelStore, *, host: str = "127.0.0.1", port: int = 7700
) -> ThreadingHTTPServer:
    """Create (but don't start) the labelling server. Use ``port=0`` for any free port."""
    page = files("runthrough.calibrate").joinpath("ui.html").read_bytes()
    by_key = {item.key: item for item in items}
    conversations = [c.model_dump() for c in group_by_conversation(items)]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
            elif self.path == "/api/conversations":
                labels = {
                    key: label.model_dump() for key, label in store.load().items() if key in by_key
                }
                self._json(
                    HTTPStatus.OK,
                    {"conversations": conversations, "labels": labels},
                )
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/api/labels":
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))))
                item = by_key[body["key"]]
                label = Label(
                    key=item.key,
                    scorer=item.scorer,
                    sample_id=item.sample_id,
                    epoch=item.epoch,
                    label=body["label"],
                    note=str(body.get("note", "")),
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid label: {exc}"})
                return
            store.add(label)
            self._json(HTTPStatus.OK, label.model_dump())

        def log_message(self, format: str, *args: Any) -> None:  # keep the terminal quiet
            pass

        def _json(self, status: HTTPStatus, data: Any) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)
