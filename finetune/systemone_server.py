#!/usr/bin/env python3
"""A System-One sidecar: Laya behind the Jev wire (`POST /v1/systemone`).

The router's brain is a PROVIDER, not a code path — so it has to be reachable over a wire that
more than one implementation speaks. This serves the one TypeSafe/Jev defines:

    POST /v1/systemone   {"state": …, "questions": {…}, "model"?: "…"}
                      -> {"model": …, "answers": {…}, "usage": {"input_tokens": n, "output_tokens": 0}}
    GET  /v1/models   -> {"data": [{"id": …, "object": "model"}, …]}     (the provider probe)
    GET  /health      -> {"status": "ok", …}

Three things speak it, so the Rust client is written once and never again:
  * this file — Laya on CPU, for the local A/B and for a box with no GPU;
  * `laya.cpp --server` — the same checkpoints in C++/ggml on CUDA or Vulkan;
  * TypeSafe's hosted Jev.

Swapping between them is a provider record edit (one approval), not a deploy.

Run:
    ~/venvs/laya/bin/python scripts/laya/systemone_server.py            # 127.0.0.1:8099, english ckpt
    LAYA_SUBFOLDER=typed-decisions LAYA_PORT=8100 …/python …            # the 1024-ctx checkpoint
    LAYA_CAPTURE_PATH=/tmp/…/capture.jsonl …                            # also record every request body

See README.md in this directory for the install recipe and the measured numbers.
"""
import hashlib
import json
import os
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Laya is torch-only, and transformers' TensorFlow probe costs seconds of import time for nothing.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_BODY = 1 << 20  # a System-One state is a request plus a little session context, never a corpus
LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _pin_threads():
    """Choose torch's thread counts explicitly. Leaving them to torch costs 10x on this box.

    Measured 2026-09-22 on the WSL box (AMD Ryzen 9 6900HX, 5 physical cores / 10 vCPUs, avx2, no
    GPU), 24 requests, the 3-question schema, p50:

        torch defaults (10 intra-op, 5 inter-op)   9,396 ms   <- and a 21,092 ms max
        8 intra-op, 1 inter-op                       783 ms
        6 intra-op, 1 inter-op                       882 ms
        5 intra-op, 1 inter-op                     1,051 ms

    The dominant term is inter-op: torch defaults it to half the vCPUs, and each of those threads
    then opens its own intra-op pool, so a 5-core machine ends up scheduling ~50 runnable threads
    for one forward pass. laya issues exactly one forward pass per call, so there is nothing for an
    inter-op pool to overlap -- 1 is not a tuning choice, it is the shape of the work.

    Both are overridable because the right intra-op count is a property of the machine, and
    `scripts/laya/thread_sweep.py` is how a new machine measures its own instead of inheriting ours.
    """
    import torch

    interop = int(_env("LAYA_INTEROP_THREADS", "1"))
    threads = int(_env("LAYA_THREADS", "8"))
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        # Only settable before the first parallel op. Reaching here means something already ran,
        # which is worth saying out loud rather than silently serving at the slow default (rule 1).
        print("laya system-one: WARNING inter-op threads already fixed at %d; wanted %d"
              % (torch.get_num_interop_threads(), interop), file=sys.stderr)
    torch.set_num_threads(threads)
    return torch.get_num_threads(), torch.get_num_interop_threads()


def served_id(model, subfolder, explicit):
    """The NAME a provider record's `model_id` must match. A local export has no name of its own:
    it must be given one, or a fine-tuned checkpoint would answer as `laya` and the record could
    not tell the two apart (rule 8)."""
    if explicit:
        return explicit
    if model and os.path.isdir(model):
        sys.exit("LAYA_MODEL=%s is a local checkpoint: set LAYA_SERVED_ID to the id the provider record "
                 "declares — a fine-tuned export must not answer as `laya`" % model)
    return subfolder or "laya"


def checkpoint_dir(model, subfolder):
    """The directory the agent's weights were loaded from, resolved the way the SDK resolves it."""
    if os.path.isdir(model):
        base = model
    else:
        from huggingface_hub import snapshot_download  # already cached by the agent's own load
        prefix = f"{subfolder}/" if subfolder else ""
        base = snapshot_download(model, allow_patterns=[prefix + "model.safetensors"])
    return os.path.join(base, subfolder) if subfolder else base


def checkpoint_sha256(model_dir):
    digest = hashlib.sha256()
    with open(os.path.join(model_dir, "model.safetensors"), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Brain:
    """One resident checkpoint, and a lock in front of it.

    One model on one device does not parallelise: concurrent forward passes contend for the same
    cores and the latency of BOTH goes up. Serialising here is what makes the p50 in the README a
    number you can plan with rather than a number that depends on who else called.

    It also says WHAT it serves: the checkpoint's sha256, the temperatures it applies (after the
    SDK's clamp), and the export's MANIFEST.json when there is one — so a provider record can pin
    the checkpoint it was measured with and the probe can refuse any other (rule 8).
    """

    def __init__(self, agent=None, *, model=None, subfolder=None, served=None, device=None):
        self.model_id = model or _env("LAYA_MODEL", "convaiinnovations/laya")
        self.subfolder = subfolder if subfolder is not None else _env("LAYA_SUBFOLDER")
        self.served = served_id(self.model_id, self.subfolder, served or _env("LAYA_SERVED_ID"))
        started = time.perf_counter()
        if agent is None:
            import laya  # imported late so --help and a bad port fail in milliseconds, not seconds

            self.threads, self.interop_threads = _pin_threads()
            agent = laya.load(self.model_id, device=device or _env("LAYA_DEVICE"), subfolder=self.subfolder)
        else:
            self.threads, self.interop_threads = 0, 0
        self.agent = agent
        self.load_seconds = time.perf_counter() - started
        self.device = str(getattr(agent, "device", "cpu"))
        self.max_len = int(agent.cfg.get("max_len", 512))
        self.checkpoint_dir = checkpoint_dir(self.model_id, self.subfolder)
        self.checkpoint_sha256 = checkpoint_sha256(self.checkpoint_dir)
        self.temperature = {**{"type:%d" % i: t for i, t in enumerate(getattr(agent, "temperature", []))},
                            **getattr(agent, "temperature_by_options", {})}
        self.temperature_raw = {**{"type:%d" % i: t for i, t in enumerate(getattr(agent, "temperature_raw", []))},
                                **getattr(agent, "temperature_by_options_raw", {})}
        manifest = os.path.join(self.checkpoint_dir, "MANIFEST.json")
        self.manifest = json.load(open(manifest)) if os.path.exists(manifest) else None
        self.combiner = None
        if _env("LAYA_COMBINER"):
            import combine
            self.combiner = combine.load(_env("LAYA_COMBINER"))
        self._lock = threading.Lock()
        self.calls = 0

    def warmup(self, message="Write a Python function that parses this CSV and returns rows above a threshold."):
        """Answer one representative request so the first real caller is not the slow one.

        The first forward pass allocates every intermediate buffer and pays lazy-init costs that no
        later call pays. Measured on this box it is several times the steady-state latency, so a
        server that skips this reports its own cold start as the router's p99.
        """
        questions = {"warm": {"type": "score", "instructions": "How hard is `request`?",
                              "criteria": ["trivial", "easy", "moderate", "hard"]}}
        started = time.perf_counter()
        self.ask({"request": message}, questions)
        return time.perf_counter() - started

    def ask(self, state, questions):
        import torch

        with self._lock, torch.inference_mode():
            self.calls += 1
            result = self.agent.system_one(state, questions)
        if self.combiner is not None:
            import combine
            result = combine.apply(result, self.combiner)
        return result


def health_payload(brain):
    return {"status": "ok", "model": brain.served, "checkpoint_sha256": brain.checkpoint_sha256,
            "checkpoint_dir": os.path.basename(os.path.normpath(brain.checkpoint_dir or "")),
            "device": brain.device, "max_len": brain.max_len, "load_seconds": round(brain.load_seconds, 3),
            "threads": brain.threads, "interop_threads": brain.interop_threads, "calls": brain.calls,
            "temperature": brain.temperature, "temperature_raw": brain.temperature_raw,
            "manifest": brain.manifest,
            "combiner_sha256": (brain.combiner or {}).get("sha256")}


def models_payload(brain):
    return {"object": "list", "data": [{"id": brain.served, "object": "model", "owned_by": "laya",
                                        "checkpoint": brain.checkpoint_sha256}]}


BRAIN = None
TOKEN = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "laya-systemone/1"

    def log_message(self, fmt, *args):  # one line per call, on stderr, with the request id
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def handle_one_request(self):
        # A client that walks away mid-response (curl -sf on a readiness loop, a browser reload) is
        # not a fault of this server: report it as one line, never a traceback, so the log keeps
        # showing real failures. Anything else still raises.
        try:
            BaseHTTPRequestHandler.handle_one_request(self)
        except (ConnectionResetError, BrokenPipeError) as exc:
            self.close_connection = True
            sys.stderr.write("%s client went away (%s)\n" % (self.address_string(), type(exc).__name__))

    # -- plumbing ----------------------------------------------------------------
    def _send(self, status, payload, request_id=None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        if request_id:
            # The same header TypeSafe returns, so a caller can correlate a slow turn with this
            # server's own log line without either side inventing an id.
            self.send_header("x-typesafe-request-id", request_id)
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self):
        if TOKEN is None:
            return True
        header = self.headers.get("authorization", "")
        return header.startswith("Bearer ") and header[7:] == TOKEN

    def _body(self):
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            raise ValueError("empty body")
        if length > MAX_BODY:
            raise ValueError("body larger than %d bytes" % MAX_BODY)
        return json.loads(self.rfile.read(length))

    # -- routes ------------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send(200, health_payload(BRAIN))
        elif path in ("/v1/models", "/models"):
            # The provider probe. A SystemOne provider declares no chat models, so this list exists
            # to prove the wire answers at all — and to let the record's model_id be checked against
            # what is actually served rather than against what someone typed (rule 8).
            if not self._authorised():
                self._send(401, {"error": "unauthorised"})
                return
            self._send(200, models_payload(BRAIN))
        else:
            self._send(404, {"error": "no route %s" % path})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/v1/systemone", "/systemone"):
            self._send(404, {"error": "no route %s" % path})
            return
        if not self._authorised():
            self._send(401, {"error": "unauthorised"})
            return
        request_id = self.headers.get("x-typesafe-request-id") or uuid.uuid4().hex
        try:
            payload = self._body()
        except Exception as error:
            # The measured reason, not "bad request": a body too large, a truncated read and invalid
            # JSON are three different callers' bugs (rule 6).
            self._send(400, {"error": "could not read the request body: %s" % error}, request_id)
            return
        capture = _env("LAYA_CAPTURE_PATH")
        if capture:
            # Exactly what the caller asked, in the order it asked it (dicts keep insertion order):
            # the only way to learn the router's real state shape and option order is to record
            # the bytes, not to read the code that is supposed to produce them.
            with open(capture, "a") as f:
                f.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
        state, questions = payload.get("state"), payload.get("questions")
        if state is None or not isinstance(questions, dict) or not questions:
            self._send(400, {"error": "a System-One call needs a `state` and a non-empty `questions` map"},
                       request_id)
            return
        wanted = payload.get("model")
        if wanted and wanted not in (BRAIN.served, "laya-latest", "jev-latest"):
            # Refusing by name beats silently answering from another checkpoint: the two differ in
            # context window and in accuracy, and a caller that got the other one would never know.
            self._send(404, {"error": "this sidecar serves %r, not %r" % (BRAIN.served, wanted)}, request_id)
            return
        started = time.perf_counter()
        try:
            result = BRAIN.ask(state, questions)
        except Exception as error:
            self._send(422, {"error": "%s: %s" % (type(error).__name__, error)}, request_id)
            return
        elapsed_ms = (time.perf_counter() - started) * 1000
        result["model"] = BRAIN.served
        result["latency_ms"] = round(elapsed_ms, 2)
        result["request_id"] = request_id
        self._send(200, result, request_id)


def main():
    global BRAIN, TOKEN
    host = _env("LAYA_HOST", "127.0.0.1")
    port = int(_env("LAYA_PORT", "8099"))
    TOKEN = _env("LAYA_API_TOKEN")
    # Loopback by default and off-loopback only with a token: this process answers anyone who can
    # reach it, and its answers pick which model spends an agent's authority.
    if host not in LOOPBACK and not TOKEN:
        sys.exit("refusing to bind %s without LAYA_API_TOKEN set (loopback needs no token)" % host)
    BRAIN = Brain()
    warm_ms = BRAIN.warmup() * 1000
    print("laya system-one: %s @ sha256 %s on %s, ctx %d, %d/%d threads, loaded in %.1fs, first call %.0fms"
          " -> http://%s:%d"
          % (BRAIN.served, BRAIN.checkpoint_sha256[:12], BRAIN.device, BRAIN.max_len, BRAIN.threads,
             BRAIN.interop_threads, BRAIN.load_seconds, warm_ms, host, port),
          flush=True)
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    if hasattr(socket, "TCP_NODELAY"):
        server.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
