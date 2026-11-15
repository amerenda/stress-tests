"""The runner side of the contract: what a test prints when it runs in a pod.

With --jsonl a test prints one JSON object per line on stdout:

  {"type":"event","t":12.3,"msg":"..."}        something happened (t = seconds since start)
  {"type":"status","t":15.0,...}                current state, every --status-interval seconds
  {"type":"report_part","n":0,"of":2,"gz":"…"}  the full report, gzipped, base64, in parts
  {"type":"result","result":{...},"parts":2}    small summary; always the last line

Anything else on stdout (tracebacks, library logging) is shown as "other output". A run stops
gracefully when --stop-file contains true/1/yes/stop; the web UI sets that through a pod
annotation and a downward API file.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import json

# The container runtime splits log lines over 16 KB, and a split line breaks tail-based reads.
# Reports go out gzipped and base64-encoded in parts well under that, then one small result line.
REPORT_PART = 12_000
STOP_WORDS = ("1", "true", "yes", "stop")


def emit(obj: dict) -> None:
    print(json.dumps(obj, separators=(",", ":"), default=str), flush=True)


def report_lines(report: dict, result: dict) -> list[str]:
    blob = base64.b64encode(gzip.compress(
        json.dumps(report, separators=(",", ":"), default=str).encode())).decode()
    parts = [blob[i:i + REPORT_PART] for i in range(0, len(blob), REPORT_PART)] or [""]
    lines = [json.dumps({"type": "report_part", "n": i, "of": len(parts), "gz": p}, separators=(",", ":"))
             for i, p in enumerate(parts)]
    lines.append(json.dumps({"type": "result", "result": result, "parts": len(parts)},
                            separators=(",", ":"), default=str))
    return lines


def emit_report(report: dict, result: dict) -> None:
    for line in report_lines(report, result):
        print(line, flush=True)


def read_report(lines) -> dict | None:
    """The report from log lines: reassembled parts, or a single legacy {"type":"report"} line."""
    parts: dict[int, str] = {}
    total = None
    for ln in lines:
        if not (ln.startswith("{") and '"report' in ln[:40]):
            continue
        try:
            obj = json.loads(ln)
        except ValueError:  # a legacy report line the runtime split
            continue
        if obj.get("type") == "report_part":
            parts[obj["n"]] = obj["gz"]
            total = obj["of"]
        elif obj.get("type") == "report":
            return obj.get("report")
    if total is None or len(parts) != total:
        return None
    return json.loads(gzip.decompress(base64.b64decode("".join(parts[i] for i in range(total)))))


def add_output_args(p: argparse.ArgumentParser, status_interval: float = 5) -> argparse._ArgumentGroup:
    """The flags every test takes for running in a pod. The web UI sets these itself."""
    g = p.add_argument_group("output (for running in a pod)")
    g.add_argument("--jsonl", action="store_true",
                   help="no terminal output: print JSON lines to stdout (event, status, and a final report)")
    g.add_argument("--status-interval", type=float, default=status_interval, help="--jsonl: s between status lines")
    g.add_argument("--stop-file", help="stop gracefully once this file contains true/1/yes/stop")
    return g


def stop_requested(path: str | None) -> bool:
    if not path:
        return False
    try:
        with open(path) as f:
            return f.read().strip().strip('"').lower() in STOP_WORDS
    except OSError:
        return False


async def watch_stop_file(path: str | None, stop: asyncio.Event, on_stop=None, interval: float = 2) -> None:
    """Set `stop` once the stop file says so (call on_stop first). Returns at once without a path."""
    if not path:
        return
    while not stop.is_set():
        if stop_requested(path):
            if on_stop:
                on_stop()
            stop.set()
            return
        await asyncio.sleep(interval)
