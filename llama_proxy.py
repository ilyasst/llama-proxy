#!/usr/bin/env python3
"""llama-proxy — Priority queue proxy for llama-server.

Sits between clients and llama-server's multi-model manager. Provides:
- Priority queuing (smallest prompt first) when slots are busy
- GPU conflict management (weight-based budget prevents OOM)
- /slots endpoint for external load balancers (caproute)
- X-No-Queue header support for instant rejection when busy
- Request logging with Tailscale hostname resolution
- Ollama-compatible /api/ps and /api/tags endpoints

Configuration via environment variables or proxy-config.json.
"""

import http.client
import http.server
import json
import os
import socketserver
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────
# All settings can be overridden via env vars or proxy-config.json.

_CONFIG_PATH = Path(os.environ.get(
    "LLAMA_PROXY_CONFIG",
    Path(__file__).parent / "proxy-config.json",
))

def _load_file_config():
    """Load config from JSON file if it exists."""
    try:
        if _CONFIG_PATH.exists():
            return json.loads(_CONFIG_PATH.read_text())
    except Exception:
        pass
    return {}

_file_config = _load_file_config()

def _cfg(key, default, cast=str):
    """Read config: env var > config file > default."""
    env_val = os.environ.get(key)
    if env_val is not None:
        return cast(env_val)
    file_val = _file_config.get(key)
    if file_val is not None:
        return cast(file_val)
    return default

UPSTREAM_HOST = _cfg("LLAMA_PROXY_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = _cfg("LLAMA_PROXY_UPSTREAM_PORT", 8081, int)
LISTEN_HOST = _cfg("LLAMA_PROXY_HOST", "0.0.0.0")
LISTEN_PORT = _cfg("LLAMA_PROXY_PORT", 8080, int)
POLL_INTERVAL = _cfg("LLAMA_PROXY_POLL_INTERVAL", 0.5, float)
MAX_WAIT = _cfg("LLAMA_PROXY_MAX_WAIT", 300, int)  # queue timeout (seconds)
LOG_DIR = _cfg("LLAMA_PROXY_LOG_DIR", os.path.expanduser("~/logs/llama"))

# ── GPU conflict rules ───────────────────────────────────────────────
# Weight-based GPU budget system. Each model has a weight representing
# its GPU resource footprint. Only models fitting within MAX_GPU_WEIGHT
# may be loaded simultaneously. When a new request would exceed budget,
# the proxy waits for conflicting heavy models to drain before admitting.
#
# Configure via proxy-config.json:
#   { "MODEL_WEIGHTS": {"qwen2.5:32b": 1, "qwen2.5:72b": 2}, "MAX_GPU_WEIGHT": 3 }
# Or set to empty {} to disable conflict management.

MODEL_WEIGHTS = _file_config.get("MODEL_WEIGHTS", {})
DEFAULT_WEIGHT = _cfg("LLAMA_PROXY_DEFAULT_WEIGHT", 1, int)
MAX_GPU_WEIGHT = _cfg("LLAMA_PROXY_MAX_GPU_WEIGHT", 3, int)

_conflict_events = []
_conflict_lock = threading.Lock()
_MAX_CONFLICT_LOG = 200


def _model_weight(name):
    """Return GPU weight for a model. Defaults to DEFAULT_WEIGHT if unknown."""
    return MODEL_WEIGHTS.get(name, DEFAULT_WEIGHT)


def _check_gpu_budget(target_model, loaded_models):
    """Check if loading target_model would exceed GPU budget.

    Returns (ok, conflict_models):
      ok=True  → safe to coexist
      ok=False → conflict_models must drain first
    """
    if not MODEL_WEIGHTS:
        return True, set()  # conflict management disabled
    target_w = _model_weight(target_model)
    other_w = sum(_model_weight(m) for m in loaded_models if m != target_model)
    total = target_w + other_w
    if total <= MAX_GPU_WEIGHT:
        return True, set()
    conflicting = set()
    budget_remaining = MAX_GPU_WEIGHT - target_w
    for m in sorted(loaded_models - {target_model}, key=lambda x: -_model_weight(x)):
        if budget_remaining < 0:
            conflicting.add(m)
        budget_remaining -= _model_weight(m)
    return False, conflicting


def _log_conflict(target, loaded, conflicting):
    """Log a conflict event for the /conflicts introspection endpoint."""
    entry = {
        "ts": time.time(),
        "target": target,
        "loaded": list(loaded),
        "conflicting": list(conflicting),
    }
    with _conflict_lock:
        _conflict_events.append(entry)
        if len(_conflict_events) > _MAX_CONFLICT_LOG:
            _conflict_events.pop(0)


# ── Request logging ──────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
REQUEST_LOG = os.path.join(LOG_DIR, "requests.jsonl")

MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB per file
MAX_LOG_FILES = 7  # keep 7 rotated files (~35 MB max)
_log_lock = threading.Lock()


def _rotate_log(path):
    """Rotate log file if it exceeds MAX_LOG_BYTES. Keep MAX_LOG_FILES backups."""
    try:
        if os.path.getsize(path) < MAX_LOG_BYTES:
            return
    except OSError:
        return
    for i in range(MAX_LOG_FILES - 1, 0, -1):
        src = f"{path}.{i}"
        dst = f"{path}.{i + 1}"
        if os.path.exists(src):
            if i + 1 >= MAX_LOG_FILES:
                os.remove(src)
            else:
                os.rename(src, dst)
    os.rename(path, f"{path}.1")


def log_request(entry):
    """Append a JSON line to the request log. Non-blocking, fire-and-forget."""
    with _log_lock:
        try:
            _rotate_log(REQUEST_LOG)
            with open(REQUEST_LOG, "a") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except Exception:
            pass


# ── Tailscale hostname resolution ────────────────────────────────────

_ts_names = {}
_ts_lock = threading.Lock()


def _refresh_tailscale_names():
    """Refresh Tailscale IP to hostname mapping."""
    global _ts_names
    try:
        out = subprocess.check_output(
            ["tailscale", "status", "--json"], timeout=5, text=True
        )
        data = json.loads(out)
        names = {}
        self_ips = data.get("TailscaleIPs", [])
        self_name = data.get("Self", {}).get("HostName", "")
        for ip in self_ips:
            if ":" not in ip and self_name:
                names[ip] = self_name
        for _, peer in data.get("Peer", {}).items():
            name = peer.get("HostName", "")
            if not name:
                continue
            for ip in peer.get("TailscaleIPs", []):
                if ":" not in ip:
                    names[ip] = name
        with _ts_lock:
            _ts_names = names
    except Exception:
        pass


def resolve_client(ip):
    """Resolve a Tailscale IP to its hostname, or return the IP as-is."""
    with _ts_lock:
        name = _ts_names.get(ip)
    return name if name else ip


def _ts_refresher():
    """Background thread that refreshes Tailscale names every 5 minutes."""
    while True:
        _refresh_tailscale_names()
        time.sleep(300)


# ── Queue infrastructure ─────────────────────────────────────────────

_seq_lock = threading.Lock()
_seq = 0


def next_seq():
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq


class QueueEntry:
    __slots__ = ("priority", "seq", "event", "admitted", "model")

    def __init__(self, priority, model=None):
        self.priority = priority
        self.seq = next_seq()
        self.event = threading.Event()
        self.admitted = False
        self.model = model

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.seq < other.seq


class SlotDispatcher:
    """Monitors slot availability and admits queued requests smallest-first."""

    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
        self._has_work = threading.Event()
        self._active_requests = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def poll_slots_and_model(self):
        """Returns (free_slots, busy_slots, loaded_models, is_loading)"""
        try:
            req = urllib.request.Request(
                f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/v1/models"
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                models_data = json.loads(resp.read())

            loaded_models = set()
            is_loading = False

            for m in models_data.get("data", []):
                status = m.get("status", {}).get("value", "")
                if status == "loaded":
                    loaded_models.add(m.get("id"))
                elif status in ("loading", "unloading"):
                    is_loading = True

            free, busy = 0, 0
            query_model = next(iter(loaded_models), None)
            if query_model:
                try:
                    req = urllib.request.Request(
                        f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/slots?"
                        + urllib.parse.urlencode({"model": query_model})
                    )
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        slots = json.loads(resp.read())
                    free = sum(1 for s in slots if not s.get("is_processing", True))
                    busy = len(slots) - free
                except Exception:
                    pass

            return free, busy, loaded_models, is_loading
        except Exception:
            return 0, 0, set(), False

    def get_slot_availability(self, model_name):
        """Query /slots for a specific model. Returns (free, busy)."""
        try:
            req = urllib.request.Request(
                f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/slots?"
                + urllib.parse.urlencode({"model": model_name})
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                slots = json.loads(resp.read())
            free = sum(1 for s in slots if not s.get("is_processing", True))
            busy = len(slots) - free
            return free, busy
        except Exception:
            return 0, 0

    def enqueue(self, entry):
        with self._lock:
            inserted = False
            for i, e in enumerate(self._queue):
                if entry < e:
                    self._queue.insert(i, entry)
                    inserted = True
                    break
            if not inserted:
                self._queue.append(entry)
        self._has_work.set()

    def remove(self, entry):
        with self._lock:
            try:
                self._queue.remove(entry)
            except ValueError:
                pass

    def mark_request_started(self):
        with self._lock:
            self._active_requests += 1

    def mark_request_finished(self):
        with self._lock:
            if self._active_requests > 0:
                self._active_requests -= 1
            self._has_work.set()

    def _run(self):
        while True:
            self._has_work.wait()

            while True:
                with self._lock:
                    if not self._queue:
                        self._has_work.clear()
                        break

                free, busy, loaded_models, is_loading = self.poll_slots_and_model()

                if is_loading:
                    time.sleep(POLL_INTERVAL)
                    continue

                with self._lock:
                    if not self._queue:
                        continue
                    top_entry = self._queue[0]
                    target_model = top_entry.model
                    active = self._active_requests

                # GPU conflict check
                if target_model and loaded_models and target_model in loaded_models:
                    ok, conflicting = _check_gpu_budget(target_model, loaded_models)
                    if not ok:
                        if not hasattr(top_entry, "_conflict_logged"):
                            _log_conflict(target_model, loaded_models, conflicting)
                            top_entry._conflict_logged = True
                        if busy > 0 or active > 0:
                            time.sleep(POLL_INTERVAL)
                            continue

                if target_model and loaded_models and target_model not in loaded_models:
                    ok, conflicting = _check_gpu_budget(target_model, loaded_models)
                    if not ok and not hasattr(top_entry, "_conflict_logged"):
                        _log_conflict(target_model, loaded_models, conflicting)
                        top_entry._conflict_logged = True
                    if busy > 0 or active > 0:
                        time.sleep(POLL_INTERVAL)
                        continue
                    with self._lock:
                        entry = self._queue.pop(0)
                        entry.admitted = True
                        entry.event.set()
                elif not loaded_models:
                    if busy > 0 or active > 0:
                        time.sleep(POLL_INTERVAL)
                        continue
                    with self._lock:
                        entry = self._queue.pop(0)
                        entry.admitted = True
                        entry.event.set()
                else:
                    per_model_free, _ = self.get_slot_availability(target_model)
                    if per_model_free > 0:
                        with self._lock:
                            entry = self._queue.pop(0)
                            entry.admitted = True
                            entry.event.set()
                    else:
                        time.sleep(POLL_INTERVAL)


dispatcher = SlotDispatcher()


# ── HTTP handler ─────────────────────────────────────────────────────

def _safe_err(msg):
    """Ensure error messages are ASCII-safe for HTTP status lines."""
    return str(msg).encode("ascii", "replace").decode("ascii")


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/api/ps":
            self._handle_ps()
        elif self.path == "/api/tags":
            self._handle_tags()
        elif self.path == "/conflicts":
            self._handle_conflicts()
        elif self.path.startswith("/slots"):
            self._handle_slots()
        elif self.path == "/v1/models":
            self._proxy_pass("GET")
        else:
            self._proxy_pass("GET")

    def _handle_conflicts(self):
        """Return conflict rules config + recent conflict events."""
        with _conflict_lock:
            events = list(_conflict_events)
        self._send_json({
            "config": {
                "model_weights": MODEL_WEIGHTS,
                "default_weight": DEFAULT_WEIGHT,
                "max_gpu_weight": MAX_GPU_WEIGHT,
            },
            "recent_conflicts": events[-20:],
            "total_conflicts": len(events),
        })

    def _handle_slots(self):
        """Expose slot state for external load balancers (caproute)."""
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        model_name = qs.get("model", [None])[0]
        if model_name:
            free, busy = dispatcher.get_slot_availability(model_name)
        else:
            free, busy, _, _ = dispatcher.poll_slots_and_model()
        slots = []
        for i in range(free + busy):
            slots.append({"id": i, "is_processing": i >= free})
        self._send_json(slots)

    def _handle_ps(self):
        """Return loaded models in Ollama /api/ps format."""
        try:
            req = urllib.request.Request(
                f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/v1/models"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = []
            for m in data.get("data", []):
                if m.get("status", {}).get("value") == "loaded":
                    models.append({"name": m.get("id")})
            self._send_json({"models": models})
        except Exception as e:
            self.send_error(502, _safe_err(e))

    def _handle_tags(self):
        """Return model list in Ollama /api/tags format."""
        try:
            req = urllib.request.Request(
                f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/v1/models"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                status = m.get("status", {}).get("value", "unloaded")
                models.append({
                    "name": mid,
                    "model": mid,
                    "modified_at": "",
                    "size": 0,
                    "details": {"family": "", "parameter_size": "", "format": "gguf"},
                    "status": status,
                })
            self._send_json({"models": models})
        except Exception as e:
            self.send_error(502, _safe_err(e))

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _extract_model(self, body):
        try:
            return json.loads(body).get("model", None)
        except Exception:
            return None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        is_probe = self._is_probe_request(body)
        target_model = self._extract_model(body)

        if self.path in (
            "/v1/chat/completions",
            "/v1/completions",
            "/chat/completions",
            "/completions",
        ):
            size = self._estimate_size(body)
            t_arrival = time.monotonic()
            queue_wait = 0.0

            free, busy, loaded_models, is_loading = dispatcher.poll_slots_and_model()

            # Fast path: slot available and model loaded
            if (
                free > 0
                and not is_loading
                and target_model in loaded_models
                and not is_probe
            ):
                dispatcher.mark_request_started()
                try:
                    resp_body = self._forward_post(body)
                finally:
                    dispatcher.mark_request_finished()
            else:
                # X-No-Queue: reject immediately instead of queuing.
                # Lets external routers (caproute) try other backends.
                if self.headers.get("X-No-Queue"):
                    self.send_error(503, "All slots busy (no-queue mode)")
                    return

                # Slow path: queue and wait
                entry = QueueEntry(size, target_model)
                dispatcher.enqueue(entry)
                entry.event.wait(timeout=MAX_WAIT)

                if not entry.admitted:
                    dispatcher.remove(entry)
                    self.send_error(503, "Queue timeout - all slots busy")
                    return

                queue_wait = time.monotonic() - t_arrival

                dispatcher.mark_request_started()
                try:
                    resp_body = self._forward_post(body)
                finally:
                    dispatcher.mark_request_finished()

            total_time = time.monotonic() - t_arrival
            client = resolve_client(self.client_address[0])
            self._log_completion(resp_body, client, queue_wait, total_time, target_model)
        else:
            self._forward_post(body)

    def do_OPTIONS(self):
        self._proxy_pass("OPTIONS")

    def _is_probe_request(self, body):
        """Detect probe requests (very short, just testing responsiveness)."""
        try:
            data = json.loads(body)
            content = ""
            if "messages" in data:
                for m in data.get("messages", []):
                    c = m.get("content", "")
                    if c:
                        content += c
            elif "prompt" in data:
                content = str(data.get("prompt", ""))
            if len(content) <= 10 and data.get("max_tokens", 999) <= 10:
                return True
        except Exception:
            pass
        return False

    def _estimate_size(self, body):
        try:
            data = json.loads(body)
            if "messages" in data:
                return sum(len(str(m.get("content", ""))) for m in data["messages"])
            elif "prompt" in data:
                return len(str(data["prompt"]))
        except Exception:
            pass
        return len(body)

    def _proxy_pass(self, method):
        """Simple proxy for GET/OPTIONS."""
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=10)
            conn.request(method, self.path, headers=self._upstream_headers())
            resp = conn.getresponse()
            self._send_response(resp)
            conn.close()
        except Exception as e:
            self.send_error(502, _safe_err(e))

    def _forward_post(self, body):
        """Forward POST, handle both regular and streaming responses.
        Returns response body bytes for logging, or None for streaming."""
        try:
            conn = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=600)
            headers = self._upstream_headers()
            headers["Content-Length"] = str(len(body))
            conn.request("POST", self.path, body=body, headers=headers)
            resp = conn.getresponse()
            resp_body = self._send_response(resp)
            conn.close()
            return resp_body
        except Exception as e:
            try:
                self.send_error(502, _safe_err(e))
            except Exception:
                pass
            return None

    def _log_completion(self, resp_body, client, queue_wait, total_time, model_name="unknown"):
        """Extract metrics from llama-server response and log them."""
        if not resp_body:
            return
        try:
            data = json.loads(resp_body)
            usage = data.get("usage", {})
            timings = data.get("timings", {})
            log_request({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "client": client,
                "model": data.get("model", model_name),
                "prompt_n": usage.get("prompt_tokens", 0),
                "gen_n": usage.get("completion_tokens", 0),
                "prompt_tps": round(timings.get("prompt_per_second", 0), 1),
                "gen_tps": round(timings.get("predicted_per_second", 0), 1),
                "queue_wait_s": round(queue_wait, 2),
                "total_s": round(total_time, 2),
            })
        except Exception:
            pass

    def _send_response(self, resp):
        """Stream upstream response back to client.
        Returns body bytes for non-streaming, None for streaming."""
        self.send_response(resp.status)
        content_type = None
        for header, value in resp.getheaders():
            h = header.lower()
            if h in ("transfer-encoding", "connection", "keep-alive"):
                continue
            if h == "content-type":
                content_type = value
            self.send_header(header, value)

        is_streaming = content_type and "text/event-stream" in content_type

        try:
            if is_streaming:
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return None
            else:
                body = resp.read()
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                return body
        except (BrokenPipeError, ConnectionResetError):
            return None

    def _upstream_headers(self):
        """Copy relevant client headers for the upstream request."""
        headers = {}
        for key in self.headers:
            k = key.lower()
            if k in ("host", "connection", "keep-alive", "transfer-encoding"):
                continue
            headers[key] = self.headers[key]
        headers["Host"] = f"{UPSTREAM_HOST}:{UPSTREAM_PORT}"
        return headers

    def log_message(self, fmt, *args):
        pass  # quiet


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


if __name__ == "__main__":
    # Show config
    print(f"[llama-proxy] {LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM_HOST}:{UPSTREAM_PORT}")
    if MODEL_WEIGHTS:
        print(f"[llama-proxy] GPU conflict management: {MODEL_WEIGHTS} (max={MAX_GPU_WEIGHT})")
    else:
        print(f"[llama-proxy] GPU conflict management: disabled")
    print(f"[llama-proxy] Queue timeout: {MAX_WAIT}s, log: {LOG_DIR}")

    _refresh_tailscale_names()
    threading.Thread(target=_ts_refresher, daemon=True).start()

    # Model keep-alive: ensure all preset models stay loaded.
    # Polls /v1/models periodically; any unloaded model gets a warmup
    # request to force-load it. Prevents cold starts after crashes/restarts.
    KEEPALIVE_INTERVAL = int(_cfg("LLAMA_PROXY_KEEPALIVE_INTERVAL", 60, int))

    def _model_keepalive():
        """Background thread: check for unloaded models and warm them up."""
        time.sleep(10)  # initial grace period for manager startup
        while True:
            try:
                req = urllib.request.Request(
                    f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/v1/models"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                for m in data.get("data", []):
                    mid = m.get("id", "")
                    status = m.get("status", {}).get("value", "")
                    preset = m.get("status", {}).get("preset", "")
                    # Only warm models that have a preset (configured, not discovered)
                    if status == "unloaded" and preset:
                        print(f"[keepalive] {mid} is unloaded, warming up...")
                        try:
                            warmup = json.dumps({
                                "model": mid,
                                "messages": [{"role": "user", "content": "hi"}],
                                "max_tokens": 3,
                            }).encode()
                            wr = urllib.request.Request(
                                f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}/v1/chat/completions",
                                data=warmup,
                                headers={"Content-Type": "application/json"},
                            )
                            urllib.request.urlopen(wr, timeout=180)
                            print(f"[keepalive] {mid} loaded successfully")
                        except Exception as e:
                            print(f"[keepalive] {mid} warmup failed: {e}")
            except Exception:
                pass
            time.sleep(KEEPALIVE_INTERVAL)

    threading.Thread(target=_model_keepalive, daemon=True).start()
    print(f"[llama-proxy] Model keepalive: every {KEEPALIVE_INTERVAL}s")

    server = ThreadedHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
