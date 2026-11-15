"""The web UI side of the contract: a small server that runs a test's CLI as Kubernetes Jobs.

A suite describes its test with a `Suite` subclass: the argparse parser (every option becomes a
form field, so the form and the CLI cannot drift), the script the Job runs, presets, and how to
summarize a run in the list. The library does the rest: validating the form with the CLI's own
parser, creating the Job (runner image, node placement, stop file), listing runs, reading live
status and the report from the pod log, stop, kill/delete, and the JSON API the page uses:

  GET  /api/options            form fields, presets, pod defaults (+ Suite.extra_options)
  POST /api/preview            {form} -> {command, error}
  GET  /api/runs               run list
  POST /api/runs               {form, pod, allow_concurrent} -> {name, command}
  GET  /api/runs/<name>        one run: summary, form, live status/events/report
  POST /api/runs/<name>/stop   graceful stop
  DELETE /api/runs/<name>      kill and delete
  GET  /api/runs/<name>/report.json, /api/runs/<name>/log.txt

Config comes from the environment (set by tofu): NAMESPACE, RUNNER_IMAGE, UI_DEFAULTS,
NODE_SELECTOR, TOLERATIONS, RUNNER_RESOURCES, RUNNER_ENV (a JSON list of container env entries,
valueFrom allowed), JOB_TTL_SECONDS, MAX_RUN_SECONDS.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import shlex
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import httpx

from .output import read_report

STOP_FILE = "/podinfo/stop"
NAME_OK = set("abcdefghijklmnopqrstuvwxyz0123456789-")
LOG_LIMIT = 64_000_000
ACTIVE = ("starting", "pending", "running", "stopping")
ALWAYS_HIDDEN = {"help", "jsonl", "status_interval", "stop_file"}


def env_json(name: str, default):
    raw = os.environ.get(name)
    return json.loads(raw) if raw else default


class Suite:
    """What a test suite tells the UI. Subclass it and set the class attributes."""

    name = "stress"                 # Job name prefix and container name
    group = "stress.amer.dev"       # annotation and label domain
    title = "Stress test"
    script = "/app/run.py"          # what the Job runs
    cli_name = "run.py"             # how the command line is shown
    command_prefix = ""             # shown before the command line, e.g. "JF_PASSWORD=... "
    default_namespace = "default"
    hidden: set = set()             # options the UI sets itself or must not take from a browser
    group_titles = {"options": "other"}
    presets: list = []              # [{"group", "name", "note", "values": {dest: value}}]
    preset_keep: tuple = ()         # fields a preset never changes (e.g. server, username)
    html_dir = "."                  # holds index.html

    def __init__(self) -> None:
        self.run_labels = {"app.kubernetes.io/name": f"{self.name}-run",
                           "app.kubernetes.io/managed-by": f"{self.name}-ui"}
        self.run_selector = ",".join(f"{k}={v}" for k, v in self.run_labels.items())
        self.stop_ann = f"{self.group}/stop"

    def build_parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError

    def parse_args(self, argv: list[str], parser: argparse.ArgumentParser) -> argparse.Namespace:
        """Parse and check argv as the CLI would; parser.error raises ValueError here."""
        return parser.parse_args(argv)

    def runner_env(self, cfg: "Config") -> list[dict]:
        """Extra env for run pods, on top of RUNNER_ENV."""
        return []

    def columns(self, form: dict, kind: str) -> dict:
        """Extra fields for a run in the list (shown by the suite's page)."""
        return {}

    def result_summary(self, report: dict) -> dict:
        """The small result kept on the Job, from a full report (runs that ended without a result line)."""
        return report

    def extra_options(self, cfg: "Config") -> dict:
        return {}

    def route(self, handler: "Handler", method: str, parts: list[str]) -> bool:
        """Extra routes; return True when handled."""
        return False


class Config:
    def __init__(self, suite: Suite) -> None:
        self.namespace = os.environ.get("NAMESPACE", suite.default_namespace)
        self.image = os.environ.get("RUNNER_IMAGE", "")
        self.defaults = env_json("UI_DEFAULTS", {})
        self.node_selector = env_json("NODE_SELECTOR", {})
        self.tolerations = env_json("TOLERATIONS", [])
        self.resources = env_json("RUNNER_RESOURCES", {})
        self.runner_env = env_json("RUNNER_ENV", [])
        self.ttl = int(os.environ.get("JOB_TTL_SECONDS", str(7 * 86400)))
        self.max_run = int(os.environ.get("MAX_RUN_SECONDS", str(3 * 3600)))


# ---------- options <-> argv ----------

def option_spec(suite: Suite, defaults: dict | None = None) -> list[dict]:
    """The CLI options as form fields, grouped like `--help`.

    Kinds: bool (with off_flag for a paired --no-x), list (append), choice, int, float, text,
    and textarea for options declared with metavar="TEXT" (multi-line values such as prompts)."""
    p = suite.build_parser()
    hidden = ALWAYS_HIDDEN | set(suite.hidden)
    negs = {a.dest: max(a.option_strings, key=len) for a in p._actions
            if isinstance(a, argparse._StoreFalseAction)}
    groups = []
    for g in p._action_groups:
        fields = []
        for a in g._group_actions:
            if a.dest in hidden or not a.option_strings or isinstance(a, argparse._StoreFalseAction):
                continue
            f = {"dest": a.dest, "flag": max(a.option_strings, key=len), "help": a.help or "",
                 "default": a.default, "cli_default": a.default, "required": bool(a.required)}
            if isinstance(a, argparse._StoreTrueAction):
                f["kind"] = "bool"
                if a.dest in negs:
                    f["off_flag"] = negs[a.dest]
            elif isinstance(a, argparse._AppendAction):
                f["kind"], f["default"], f["cli_default"] = "list", [], None
            elif a.choices:
                f["kind"], f["choices"] = "choice", list(a.choices)
            elif a.type is int:
                f["kind"] = "int"
            elif a.type is float:
                f["kind"] = "float"
            elif a.metavar == "TEXT":
                f["kind"] = "textarea"
            else:
                f["kind"] = "text"
            if defaults and a.dest in defaults:
                f["default"] = defaults[a.dest]
            fields.append(f)
        if fields:
            groups.append({"title": suite.group_titles.get(g.title, g.title), "fields": fields})
    return [g for g in groups if g["title"] != "other"] + [g for g in groups if g["title"] == "other"]


def _same_as_cli_default(f: dict, raw: str) -> bool:
    d = f["cli_default"]
    if d is None:
        return False
    try:
        conv = {"int": int, "float": float}.get(f["kind"], str)
        return conv(raw) == d
    except ValueError:
        return False


def build_argv(form: dict, spec: list[dict]) -> list[str]:
    argv: list[str] = []
    for g in spec:
        for f in g["fields"]:
            v = form.get(f["dest"])
            if f["kind"] == "bool":
                on = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "yes")
                if v is None:
                    on = bool(f["default"])
                if on and not f["default"]:
                    argv.append(f["flag"])
                elif not on and f["default"] and f.get("off_flag"):
                    argv.append(f["off_flag"])
            elif f["kind"] == "list":
                items = v if isinstance(v, list) else str(v or "").splitlines()
                argv += [f"{f['flag']}={s}" for s in (str(x).strip() for x in items) if s]
            elif f["kind"] == "textarea":
                text = str(v or "").replace("\r\n", "\n").strip()
                if text and text != str(f["cli_default"] or "").strip():
                    argv.append(f"{f['flag']}={text}")
            elif v is not None and str(v).strip() != "" and not _same_as_cli_default(f, str(v).strip()):
                argv.append(f"{f['flag']}={str(v).strip()}")
    return argv


def validate_argv(suite: Suite, argv: list[str]) -> argparse.Namespace:
    p = suite.build_parser()

    def fail(msg):
        raise ValueError(msg)

    p.error = fail
    return suite.parse_args(argv + ["--jsonl"], p)


def parse_pod(pod: dict, cfg: Config) -> dict:
    def load(key, default, typ):
        raw = pod.get(key)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return default
        val = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(val, typ):
            raise ValueError(f"{key} must be a JSON {typ.__name__}")
        return val

    ns = load("node_selector", cfg.node_selector, dict)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in ns.items()):
        raise ValueError("node_selector values must be strings")
    tol = load("tolerations", cfg.tolerations, list)
    if not all(isinstance(t, dict) for t in tol):
        raise ValueError("tolerations must be a list of objects")
    max_run = int(pod.get("max_run_seconds") or cfg.max_run)
    if max_run < 60:
        raise ValueError("max_run_seconds must be at least 60")
    return {"node_selector": ns, "tolerations": tol, "max_run_seconds": max_run}


def job_manifest(suite: Suite, name: str, argv: list[str], pod: dict, form: dict, cfg: Config,
                 script: str | None = None, kind: str = "stress") -> dict:
    labels = dict(suite.run_labels)
    g = suite.group
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name, "namespace": cfg.namespace, "labels": labels,
            "annotations": {f"{g}/kind": kind,
                            f"{g}/form": json.dumps(form, sort_keys=True),
                            f"{g}/argv": json.dumps(argv),
                            f"{g}/pod": json.dumps(pod, sort_keys=True)},
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": cfg.ttl,
            "activeDeadlineSeconds": pod["max_run_seconds"],
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "automountServiceAccountToken": False,
                    "enableServiceLinks": False,
                    "terminationGracePeriodSeconds": 60,
                    "nodeSelector": pod["node_selector"],
                    "tolerations": pod["tolerations"],
                    "securityContext": {"runAsNonRoot": True, "runAsUser": 65534, "runAsGroup": 65534,
                                        "seccompProfile": {"type": "RuntimeDefault"}},
                    "containers": [{
                        "name": suite.name,
                        "image": cfg.image,
                        "command": ["python", "-u", script or suite.script],
                        "args": argv + ["--jsonl", f"--stop-file={STOP_FILE}"],
                        "env": suite.runner_env(cfg) + list(cfg.runner_env),
                        "resources": cfg.resources,
                        "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                                            "capabilities": {"drop": ["ALL"]}},
                        "volumeMounts": [{"name": "podinfo", "mountPath": "/podinfo", "readOnly": True},
                                         {"name": "tmp", "mountPath": "/tmp"}],
                    }],
                    "volumes": [{"name": "podinfo", "downwardAPI": {"items": [{
                        "path": "stop",
                        "fieldRef": {"fieldPath": f"metadata.annotations['{suite.stop_ann}']"}}]}},
                        {"name": "tmp", "emptyDir": {"sizeLimit": "1Gi"}}],
                },
            },
        },
    }


def command_line(suite: Suite, argv: list[str], cli_name: str | None = None) -> str:
    return f"{suite.command_prefix}./{cli_name or suite.cli_name} " + " ".join(shlex.quote(a) for a in argv)


def run_state(job: dict, pod: dict | None, stop_ann: str) -> tuple[str, str]:
    st = job.get("status", {})
    conds = {c["type"]: c for c in st.get("conditions", []) if c.get("status") == "True"}
    if "Complete" in conds or st.get("succeeded"):
        return "succeeded", ""
    if "Failed" in conds or st.get("failed"):
        c = conds.get("Failed", {})
        return "failed", c.get("reason", "") + (": " + c["message"] if c.get("message") else "")
    if job["spec"].get("suspend"):
        return "suspended", ""
    if pod is None:
        return "starting", ""
    stopping = (pod["metadata"].get("annotations") or {}).get(stop_ann) == "true"
    phase = pod.get("status", {}).get("phase", "")
    if phase == "Pending":
        why = ""
        for c in pod["status"].get("conditions", []):
            if c.get("type") == "PodScheduled" and c.get("status") == "False":
                why = c.get("message", c.get("reason", ""))
        for cs in pod["status"].get("containerStatuses", []):
            w = cs.get("state", {}).get("waiting")
            if w:
                why = w.get("reason", "") + (": " + w["message"] if w.get("message") else "")
        return "pending", why
    if phase == "Running":
        return ("stopping" if stopping else "running"), ""
    return phase.lower() or "unknown", ""


def parse_lines(lines: list[str]) -> dict:
    status, result, events, other = None, None, [], []
    report = read_report(lines)
    for ln in lines:
        if ln.startswith(('{"type":"report', '{"type": "report')):
            continue
        obj = None
        if ln.startswith("{"):
            try:
                obj = json.loads(ln)
            except ValueError:
                obj = None
        if not isinstance(obj, dict):
            if ln.strip():
                other.append(ln)
            continue
        t = obj.get("type")
        if t == "status":
            status = obj
        elif t == "event":
            events.append(obj)
        elif t == "result":
            result = obj.get("result")
    return {"status": status, "report": report, "result": result,
            "events": events[-200:], "other": other[-60:]}


def is_run(suite: Suite, job: dict) -> bool:
    labels = job["metadata"].get("labels") or {}
    return all(labels.get(k) == v for k, v in suite.run_labels.items())


# ---------- Kubernetes ----------

class KubeError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


class Kube:
    SA = "/var/run/secrets/kubernetes.io/serviceaccount"

    def __init__(self, base: str | None = None):
        self.base = base or os.environ.get("KUBE_API", "https://kubernetes.default.svc")
        ca = f"{self.SA}/ca.crt"
        verify = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else True
        self.http = httpx.Client(verify=verify, timeout=20)

    def _headers(self, extra: dict | None = None) -> dict:
        h = dict(extra or {})
        try:
            with open(f"{self.SA}/token") as f:  # projected token rotates: read every call
                h["Authorization"] = "Bearer " + f.read().strip()
        except OSError:
            pass
        return h

    def req(self, method: str, path: str, *, headers=None, **kw) -> httpx.Response:
        r = self.http.request(method, self.base + path, headers=self._headers(headers), **kw)
        if r.status_code >= 400:
            try:
                msg = r.json().get("message", r.text)
            except ValueError:
                msg = r.text
            raise KubeError(r.status_code, msg[:500])
        return r

    def patch(self, path: str, body: dict) -> dict:
        return self.req("PATCH", path, headers={"Content-Type": "application/merge-patch+json"},
                        content=json.dumps(body)).json()


class Runs:
    def __init__(self, suite: Suite, cfg: Config, kube: Kube):
        self.suite, self.cfg, self.kube = suite, cfg, kube
        ns = cfg.namespace
        self.jobs = f"/apis/batch/v1/namespaces/{ns}/jobs"
        self.pods = f"/api/v1/namespaces/{ns}/pods"
        self.events = f"/api/v1/namespaces/{ns}/events"

    def _pods_by_job(self) -> dict[str, dict]:
        items = self.kube.req("GET", self.pods, params={"labelSelector": self.suite.run_selector}).json()["items"]
        out: dict[str, dict] = {}
        for p in sorted(items, key=lambda p: p["metadata"]["creationTimestamp"]):
            out[p["metadata"].get("labels", {}).get("job-name", "")] = p
        return out

    def _pod_for(self, name: str) -> dict | None:
        items = self.kube.req("GET", self.pods, params={"labelSelector": f"job-name={name}"}).json()["items"]
        return max(items, key=lambda p: p["metadata"]["creationTimestamp"]) if items else None

    def state(self, job: dict, pod: dict | None) -> tuple[str, str]:
        return run_state(job, pod, self.suite.stop_ann)

    def _report_from(self, pod: dict) -> dict | None:
        """Runs end with N report parts and a result line saying N: read just those.
        Older runs have one big report line, which only a read of the whole log gets intact."""
        last = self._log_lines(pod, tailLines=1)
        parts = None
        if last and last[-1].startswith('{"type":"result"'):
            try:
                parts = int(json.loads(last[-1]).get("parts") or 0)
            except ValueError:
                parts = None
        if parts:
            return read_report(self._log_lines(pod, tailLines=parts + 1))
        return read_report(self._log_lines(pod, limitBytes=LOG_LIMIT))

    def _log_lines(self, pod: dict, **params) -> list[str]:
        try:
            r = self.kube.req("GET", f"{self.pods}/{pod['metadata']['name']}/log",
                              params={"container": self.suite.name, **params})
        except KubeError:
            return []
        return r.text.splitlines()

    def _summary(self, job: dict, pod: dict | None) -> dict:
        g = self.suite.group
        ann = job["metadata"].get("annotations") or {}
        state, why = self.state(job, pod)
        form = json.loads(ann.get(f"{g}/form", "{}"))
        kind = ann.get(f"{g}/kind", "stress")
        result = json.loads(ann[f"{g}/result"]) if f"{g}/result" in ann else None
        if f"{g}/result" not in ann and state in ("succeeded", "failed") and pod is not None:
            result = parse_lines(self._log_lines(pod, tailLines=3))["result"]
            if result is None:  # runs without a result line: summarize the report
                rep = self._report_from(pod)
                result = self.suite.result_summary(rep) if rep else None
            try:  # cache, so finished runs are read once; "null" marks a run without a report
                self.kube.patch(f"{self.jobs}/{job['metadata']['name']}",
                                {"metadata": {"annotations": {f"{g}/result": json.dumps(result)}}})
            except KubeError:
                pass
        return {
            "name": job["metadata"]["name"], "kind": kind,
            "created": job["metadata"]["creationTimestamp"],
            "finished": job.get("status", {}).get("completionTime"),
            "state": state, "reason": why,
            "node": (pod or {}).get("spec", {}).get("nodeName"),
            **self.suite.columns(form, kind),
            "result": result,
        }

    def list(self) -> list[dict]:
        jobs = self.kube.req("GET", self.jobs, params={"labelSelector": self.suite.run_selector}).json()["items"]
        pods = self._pods_by_job()
        jobs.sort(key=lambda j: j["metadata"]["creationTimestamp"], reverse=True)
        return [self._summary(j, pods.get(j["metadata"]["name"])) for j in jobs]

    def get(self, name: str, *, report_inline: bool = True) -> dict:
        g = self.suite.group
        job = self.kube.req("GET", f"{self.jobs}/{name}").json()
        if not is_run(self.suite, job):
            raise KubeError(404, f"not a {self.suite.name} run")
        pod = self._pod_for(name)
        out = self._summary(job, pod)
        ann = job["metadata"].get("annotations") or {}
        out["form"] = json.loads(ann.get(f"{g}/form", "{}"))
        out["pod"] = json.loads(ann.get(f"{g}/pod", "{}"))
        out["command"] = command_line(self.suite, json.loads(ann.get(f"{g}/argv", "[]")),
                                      self.cli_name_for(out["kind"]))
        out["live"] = parse_lines(self._log_lines(pod, tailLines=400)) if pod else None
        finished = out["state"] in ("succeeded", "failed")
        if not report_inline:
            if out["live"]:
                out["live"]["report"] = None  # large; a report page fetches it
            out["has_report"] = finished and out["result"] is not None
        elif finished and pod and out["live"] is not None and out["live"]["report"] is None:
            out["live"]["report"] = self._report_from(pod)
        if out["state"] in ("pending", "starting") and pod:
            evs = self.kube.req("GET", self.events, params={
                "fieldSelector": f"involvedObject.name={pod['metadata']['name']}"}).json()["items"]
            out["pod_events"] = [f"{e.get('type')} {e.get('reason')}: {e.get('message')}" for e in evs[-10:]]
        return out

    def cli_name_for(self, kind: str) -> str:
        return self.suite.cli_name

    def create(self, body: dict) -> dict:
        form = body.get("form") or {}
        spec = option_spec(self.suite, self.cfg.defaults)
        argv = build_argv(form, spec)
        validate_argv(self.suite, argv)
        return self.start(argv, form, body)

    def start(self, argv: list[str], form: dict, body: dict, *, prefix: str | None = None,
              script: str | None = None, kind: str = "stress", cli_name: str | None = None) -> dict:
        pod = parse_pod(body.get("pod") or {}, self.cfg)
        if not body.get("allow_concurrent"):
            self._refuse_if_busy()
        name = self._new_name(prefix or self.suite.name)
        self.kube.req("POST", self.jobs, json=job_manifest(self.suite, name, argv, pod, form, self.cfg,
                                                            script=script, kind=kind))
        return {"name": name, "command": command_line(self.suite, argv, cli_name)}

    def _refuse_if_busy(self) -> None:
        busy = [r["name"] for r in self.list() if r["state"] in ACTIVE]
        if busy:
            raise KubeError(409, f"run {busy[0]} is still active; stop it first or allow concurrent runs")

    @staticmethod
    def _new_name(prefix: str) -> str:
        return f"{prefix}-{dt.datetime.now(dt.timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"

    def stop(self, name: str) -> None:
        self._own(name)
        pod = self._pod_for(name)
        if pod is None:
            raise KubeError(409, "run has no pod yet; delete it instead")
        self.kube.patch(f"{self.pods}/{pod['metadata']['name']}",
                        {"metadata": {"annotations": {self.suite.stop_ann: "true"}}})

    def delete(self, name: str) -> None:
        self._own(name)
        self.kube.req("DELETE", f"{self.jobs}/{name}", params={"propagationPolicy": "Background"})

    def report(self, name: str) -> dict | None:
        self._own(name)
        pod = self._pod_for(name)
        return self._report_from(pod) if pod else None

    def log(self, name: str) -> str:
        self._own(name)
        pod = self._pod_for(name)
        return "\n".join(self._log_lines(pod, limitBytes=LOG_LIMIT)) if pod else ""

    def _own(self, name: str) -> None:
        if not is_run(self.suite, self.kube.req("GET", f"{self.jobs}/{name}").json()):
            raise KubeError(404, f"not a {self.suite.name} run")


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    suite: Suite
    cfg: Config
    runs: Runs

    def log_message(self, fmt, *args):  # quieter: no probe noise
        if not self.path.startswith("/healthz"):
            super().log_message(fmt, *args)

    def send(self, code: int, body, ctype="application/json", headers=None):
        data = body if isinstance(body, bytes) else (
            body.encode() if isinstance(body, str) else json.dumps(body).encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def send_page(self, filename: str):
        with open(os.path.join(self.suite.html_dir, filename), "rb") as f:
            return self.send(200, f.read(), "text/html; charset=utf-8")

    def body(self) -> dict:
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            raise KubeError(415, "send application/json")
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def options(self) -> dict:
        return {"title": self.suite.title,
                "groups": option_spec(self.suite, self.cfg.defaults), "presets": self.suite.presets,
                "preset_keep": list(self.suite.preset_keep),
                "pod": {"node_selector": self.cfg.node_selector, "tolerations": self.cfg.tolerations,
                        "max_run_seconds": self.cfg.max_run},
                "namespace": self.cfg.namespace, "ttl_seconds": self.cfg.ttl,
                **self.suite.extra_options(self.cfg)}

    def route(self, method: str):
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        try:
            if self.suite.route(self, method, parts):
                return None
            if method == "GET" and not parts:
                return self.send_page("index.html")
            if parts == ["healthz"]:
                return self.send(200, "ok", "text/plain")
            if parts[:1] != ["api"]:
                return self.send(404, {"error": "not found"})
            rest = parts[1:]
            if method == "GET" and rest == ["options"]:
                return self.send(200, self.options())
            if method == "POST" and rest == ["preview"]:
                body = self.body()
                argv = build_argv(body.get("form") or {}, option_spec(self.suite, self.cfg.defaults))
                try:
                    validate_argv(self.suite, argv)
                    err = None
                except ValueError as e:
                    err = str(e)
                return self.send(200, {"command": command_line(self.suite, argv), "error": err})
            if rest == ["runs"]:
                if method == "GET":
                    return self.send(200, {"runs": self.runs.list()})
                if method == "POST":
                    return self.send(201, self.runs.create(self.body()))
            if len(rest) >= 2 and rest[0] == "runs" and set(rest[1]) <= NAME_OK:
                name, tail = rest[1], rest[2:]
                if method == "GET" and not tail:
                    return self.send(200, self.runs.get(name))
                if method == "DELETE" and not tail:
                    self.runs.delete(name)
                    return self.send(200, {"deleted": name})
                if method == "POST" and tail == ["stop"]:
                    self.body()
                    self.runs.stop(name)
                    return self.send(200, {"stopping": name})
                if method == "GET" and tail == ["report.json"]:
                    rep = self.runs.report(name)
                    if rep is None:
                        return self.send(404, {"error": "no report (yet)"})
                    return self.send(200, json.dumps(rep, indent=2), headers={
                        "Content-Disposition": f'attachment; filename="{name}.json"'})
                if method == "GET" and tail == ["log.txt"]:
                    return self.send(200, self.runs.log(name), "text/plain; charset=utf-8", {
                        "Content-Disposition": f'attachment; filename="{name}.log"'})
            return self.send(404, {"error": "not found"})
        except ValueError as e:
            return self.send(400, {"error": str(e)})
        except KubeError as e:
            return self.send(e.status if 400 <= e.status < 600 else 502, {"error": str(e)})
        except httpx.HTTPError as e:
            return self.send(502, {"error": f"kubernetes API: {e}"})

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")

    def do_DELETE(self):
        self.route("DELETE")


def serve(suite: Suite, cfg: Config | None = None, runs: Runs | None = None,
          handler: type[Handler] = Handler) -> None:
    cfg = cfg or Config(suite)
    h = type(f"{suite.name}Handler", (handler,), {"suite": suite, "cfg": cfg,
                                                  "runs": runs or Runs(suite, cfg, Kube()),
                                                  "server_version": f"{suite.name}-ui"})
    host, port = os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8080"))
    print(f"{suite.name}-ui on {host}:{port}, namespace {cfg.namespace}, image {cfg.image}", flush=True)
    ThreadingHTTPServer((host, port), h).serve_forever()
