"""Local GUI server for the E1 validation harness.

Standard-library only (http.server + threading + tkinter for native file
pickers). Run it and it opens a browser tab with an instrument-style control
panel: pick your images/reference, choose a backend, hit Run, watch the live
log, and inspect the 3D error heatmap — all on localhost, no cloud.

    python gui/server.py            # opens http://localhost:8000
    python gui/server.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from e1_harness.config import Config          # noqa: E402
from e1_harness.pipeline import run as run_pipeline  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared run state (one run at a time — this is a single-user local tool).
# --------------------------------------------------------------------------- #
class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.status = "idle"      # idle | running | done | error
        self.log: list[str] = []
        self.report: dict | None = None
        self.error: str | None = None
        self.work_dir: str | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "status": self.status,
                "log": list(self.log),
                "report": self.report,
                "error": self.error,
            }


STATE = RunState()


class _LogWriter(io.TextIOBase):
    """Tee stdout from the pipeline into the shared log and the real console."""

    def __init__(self, real: io.TextIOBase) -> None:
        self._real = real
        self._buf = ""

    def write(self, s: str) -> int:
        self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            with STATE.lock:
                STATE.log.append(line)
        return len(s)


def _run_thread(payload: dict) -> None:
    try:
        cfg = Config.from_dict(payload)
        reuse = payload.get("reuse_mesh") or None
        with STATE.lock:
            STATE.work_dir = cfg.work_dir
        writer = _LogWriter(sys.stdout)
        with contextlib.redirect_stdout(writer):
            report = run_pipeline(cfg, skip_reconstruction=Path(reuse) if reuse else None)
        with STATE.lock:
            STATE.report = report
            STATE.status = "done"
    except Exception:  # noqa: BLE001 — surface any failure to the UI
        with STATE.lock:
            STATE.error = traceback.format_exc()
            STATE.status = "error"
            STATE.log.append("ERROR: " + (STATE.error.strip().splitlines() or ["unknown"])[-1])


# --------------------------------------------------------------------------- #
# Native OS file/folder pickers (run on the user's own machine).
# --------------------------------------------------------------------------- #
def _native_pick(kind: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "dir":
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(
                filetypes=[("Meshes", "*.ply *.obj"), ("All files", "*.*")]
            )
    finally:
        root.destroy()
    return path or None


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # quiet console
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/status":
            self._json(STATE.snapshot())
        elif path == "/heatmap.ply":
            wd = STATE.work_dir
            ply = Path(wd) / "error_heatmap.ply" if wd else None
            if ply and ply.exists():
                self._send(200, ply.read_bytes(), "text/plain")
            else:
                self._send(404, b"no heatmap yet", "text/plain")
        elif path in ("/pick-dir", "/pick-file"):
            picked = _native_pick("dir" if path == "/pick-dir" else "file")
            self._json({"path": picked, "available": picked is not None})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            return self._json({"error": f"bad JSON: {exc}"}, code=400)

        if path == "/run":
            with STATE.lock:
                if STATE.status == "running":
                    return self._json({"error": "a run is already in progress"}, code=409)
                STATE.reset()
                STATE.status = "running"
            threading.Thread(target=_run_thread, args=(payload,), daemon=True).start()
            return self._json({"started": True})
        self._json({"error": "not found"}, code=404)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1 harness local GUI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"E1 harness GUI  →  {url}")
    print("Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
