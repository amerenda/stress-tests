#!/usr/bin/env python3
"""jfstress web UI, on the shared stresskit web UI library.

Serves a form with every jfstress option (read from jfstress's own argparse parser, so the
two cannot drift) and runs each test as a Kubernetes Job in this pod's namespace. The Job
runs `jfstress.py --jsonl`; the UI reads its pod log for live status and the final report.
A graceful stop sets a pod annotation that the runner sees through a downward API file.
The library adds the library compatibility report (jfcompat.py) as a second kind of run.

Config (env), on top of stresskit.webui's (NAMESPACE, RUNNER_IMAGE, UI_DEFAULTS, NODE_SELECTOR,
TOLERATIONS, RUNNER_RESOURCES, RUNNER_ENV, JOB_TTL_SECONDS, MAX_RUN_SECONDS):
  JF_SECRET_NAME/KEY   Secret holding the Jellyfin password (env JF_PASSWORD in the runner)
  SERVER_PRESETS       JSON list of server URLs offered in the form
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):  # the suite, then the repo
    if path not in sys.path:
        sys.path.insert(1, path)

import jfcompat  # noqa: E402
import jfstress  # noqa: E402
from stresskit import webui  # noqa: E402
from stresskit.webui import build_argv, env_json, parse_pod  # noqa: E402,F401 (API kept for tests)

# Options the UI sets itself or must not take from a browser.
HIDDEN = {"help", "password", "api_key", "out", "jsonl", "status_interval", "stop_file"}

# Pre-canned tests. Fields not listed reset to the CLI defaults (server and username are kept).
# Notes are for the gmktec nodes: Ryzen 7 7730U, Radeon Vega 8 (VCN 2: H.264/HEVC/VP9 decode,
# H.264/HEVC encode, no AV1 at all, and no GPU decodes 10-bit H.264), and for the encoding
# settings of dean-tofu modules/jellyfin (tone mapping off, AV1 encoding off).
LAN = 200_000_000  # bitrate cap high enough that remux/direct are never refused for bitrate
PRESETS = [
    {"group": "GPU transcode", "name": "Baseline 1080p", "values": {
        "ramp": True, "filter": "1080p,h264,sdr"},
     "note": "1080p H.264 to 1080p H.264, all on the GPU. The cheapest real transcode: the ceiling."},
    {"group": "GPU transcode", "name": "Remote 720p 4 Mbps", "values": {
        "ramp": True, "filter": "1080p,sdr", "max_height": 720, "bitrate": 4_000_000},
     "note": "What a remote viewer on a capped connection costs."},
    {"group": "GPU transcode", "name": "HEVC output", "values": {
        "ramp": True, "filter": "1080p,sdr", "codec": "hevc"},
     "note": "Same sources as the baseline, HEVC encoder instead of H.264."},
    {"group": "GPU transcode", "name": "4K HDR to 1080p", "values": {
        "ramp": True, "filter": "4k,hdr"},
     "note": "4K 10-bit HEVC decode and downscale to 1080p H.264 on the iGPU. Tone mapping is off in "
             "Jellyfin, so this measures decode and scaling only."},
    {"group": "GPU transcode", "name": "Dolby Vision", "values": {
        "ramp": True, "filter": "dovi"},
     "note": "Dolby Vision sources (some profile 7 with an enhancement layer) transcoded to 1080p H.264."},
    {"group": "GPU transcode", "name": "4K HDR, one file", "values": {
        "ramp": True, "filter": "4k,hdr", "same_item": True},
     "note": "Every stream plays the same 4K HDR file from a different spot: a repeatable number."},
    {"group": "CPU fallback", "name": "AV1 source", "values": {
        "ramp": True, "filter": "av1", "max_streams": 10},
     "note": "Vega 8 has no AV1 decoder, so every stream decodes AV1 in software (dav1d) on the CPU."},
    {"group": "CPU fallback", "name": "10-bit H.264 anime", "values": {
        "ramp": True, "filter": "hi10p", "max_streams": 10},
     "note": "No GPU decodes 10-bit H.264 (Hi10P): software decode on the CPU."},
    {"group": "Subtitle burn-in", "name": "ASS (anime)", "values": {
        "ramp": True, "filter": "ass", "burn_subs": True, "max_startup": 90},
     "note": "Styled ASS rendered by libass on the CPU. First play of a file extracts every subtitle "
             "track and font first, hence the 90 s startup limit."},
    {"group": "Subtitle burn-in", "name": "PGS on 4K HDR", "values": {
        "ramp": True, "filter": "4k,hdr,pgs", "burn_subs": True, "max_startup": 60},
     "note": "The worst common case: 4K HDR decode and downscale plus a Blu-ray image subtitle overlay."},
    {"group": "Subtitle burn-in", "name": "DVD subs", "values": {
        "ramp": True, "filter": "dvdsub", "burn_subs": True, "max_startup": 60},
     "note": "DVD bitmap subtitles overlaid on the video."},
    {"group": "Subtitle burn-in", "name": "SRT 1080p", "values": {
        "ramp": True, "filter": "1080p,srt", "burn_subs": True},
     "note": "Plain text subtitles: the cheapest burn-in, compare with the baseline."},
    {"group": "Throughput", "name": "High-bitrate remux", "values": {
        "ramp": True, "mix": "remux", "filter": "highbitrate", "bitrate": LAN},
     "note": "Files over 40 Mbps remuxed to HLS: disk, NFS and network, almost no GPU."},
    {"group": "Throughput", "name": "4K direct play", "values": {
        "ramp": True, "mix": "direct", "filter": "4k", "bitrate": LAN},
     "note": "4K files streamed as-is at real-time pace: storage and network only."},
    {"group": "Realistic", "name": "Random library", "values": {
        "ramp": True, "mix": "transcode"},
     "note": "Transcode ramp over random files: what an average stream costs."},
    {"group": "Realistic", "name": "Household mix (8)", "values": {
        "streams": 8, "mix": "transcode=2,remux=2,direct=4", "duration": 900, "bitrate": 60_000_000},
     "note": "8 viewers for 15 minutes: 2 transcodes, 2 remuxes, 4 direct plays."},
    {"group": "Realistic", "name": "Heavy mix ramp", "values": {
        "ramp": True, "mix": "transcode=3,remux=1", "filter": "4k", "burn_subs": True,
        "max_startup": 60, "bitrate": 20_000_000},
     "note": "4K sources, 3 of 4 transcoded with subtitles burned in, 1 remuxed."},
]


class JellyfinSuite(webui.Suite):
    name = "jfstress"
    group = "jfstress.amer.dev"
    title = "Jellyfin stress test"
    script = "/app/jfstress.py"
    cli_name = "jfstress.py"
    command_prefix = "JF_PASSWORD=... "
    default_namespace = "jellyfin-stress"
    hidden = HIDDEN
    presets = PRESETS
    preset_keep = ("server", "username", "insecure", "timeout")
    html_dir = HERE

    def build_parser(self):
        return jfstress.build_parser()

    def parse_args(self, argv, parser):
        return jfstress.parse_args(argv, parser=parser)

    def runner_env(self, cfg):
        return [{"name": "JF_PASSWORD", "valueFrom": {"secretKeyRef": {
            "name": cfg.secret_name, "key": cfg.secret_key}}}]

    def columns(self, form, kind):
        return {"server": form.get("server"),
                "mode": "compat" if kind == "compat"
                else "ramp" if form.get("ramp") in (True, "true", "on") else "fixed",
                "mix": form.get("mix"), "streams": form.get("streams"), "filter": form.get("filter")}

    def result_summary(self, rep):
        if rep.get("type") == "compat":
            return jfcompat.compat_result(rep)
        return {k: rep.get(k) for k in ("ramp_result_streams", "ramp_reason", "streams_total", "errors",
                                        "stalls_total", "stall_seconds_total", "startup_p50_s",
                                        "startup_max_s", "stopped_by_user", "duration_s")}

    def extra_options(self, cfg):
        return {"server_presets": cfg.presets,
                "devices": [{"id": d.id, "name": d.name, "app": d.app} for d in jfcompat.DEVICES]}

    def route(self, handler, method, parts):
        if method == "GET" and len(parts) == 2 and parts[0] == "compat" and set(parts[1]) <= webui.NAME_OK:
            handler.send_page("compat.html")
            return True
        if method == "POST" and parts == ["api", "compat"]:
            handler.send(201, handler.runs.create_compat(handler.body()))
            return True
        return False


SUITE = JellyfinSuite()
STOP_ANN = SUITE.stop_ann


class Config(webui.Config):
    def __init__(self) -> None:
        super().__init__(SUITE)
        self.secret_name = os.environ.get("JF_SECRET_NAME", "jellyfin-stress-jellyfin")
        self.secret_key = os.environ.get("JF_SECRET_KEY", "password")
        self.presets = env_json("SERVER_PRESETS", [])


def option_spec(defaults: dict | None = None) -> list[dict]:
    return webui.option_spec(SUITE, defaults)


def validate_argv(argv: list[str]):
    return webui.validate_argv(SUITE, argv)


def job_manifest(name, argv, pod, form, cfg, script="/app/jfstress.py", kind="stress"):
    return webui.job_manifest(SUITE, name, argv, pod, form, cfg, script=script, kind=kind)


def command_line(argv: list[str], script: str = "jfstress.py") -> str:
    return webui.command_line(SUITE, argv, script)


def _is_run(job: dict) -> bool:
    return webui.is_run(SUITE, job)


def compat_argv(body: dict, defaults: dict) -> tuple[list[str], dict]:
    """argv for jfcompat.py from the report form, checked by its own parser."""
    form = {"server": (body.get("server") or defaults.get("server") or "").strip(),
            "username": (body.get("username") or defaults.get("username") or "").strip(),
            "subtitles": body.get("subtitles", True) not in (False, "false", "off"),
            "concurrency": str(body.get("concurrency") or "8").strip(),
            "library": (body.get("library") or "").strip(),
            "types": (body.get("types") or "Movie,Episode").strip(),
            "devices": [d for d in body.get("devices") or [] if d],
            "limit": str(body.get("limit") or "").strip()}
    if not form["server"] or not form["username"]:
        raise ValueError("server and username are required")
    argv = [f"--server={form['server']}", f"--username={form['username']}", f"--types={form['types']}",
            f"--concurrency={form['concurrency']}"]
    if not form["subtitles"]:
        argv.append("--no-subtitles")
    if form["library"]:
        argv.append(f"--library={form['library']}")
    if form["limit"]:
        argv.append(f"--limit={form['limit']}")
    argv += [f"--device={d}" for d in form["devices"]]
    p = jfcompat.build_parser()

    def fail(msg):
        raise ValueError(msg)

    p.error = fail
    a = p.parse_args(argv)
    if not 1 <= a.concurrency <= 32:
        raise ValueError("concurrency must be 1-32")
    return argv, form


class Runs(webui.Runs):
    def __init__(self, cfg: Config, kube: webui.Kube):
        super().__init__(SUITE, cfg, kube)

    state = staticmethod(lambda job, pod: webui.run_state(job, pod, STOP_ANN))
    parse_lines = staticmethod(webui.parse_lines)
    result_summary = staticmethod(SUITE.result_summary)

    def cli_name_for(self, kind):
        return "jfcompat.py" if kind == "compat" else "jfstress.py"

    def get(self, name: str) -> dict:
        job = self.kube.req("GET", f"{self.jobs}/{name}").json()
        kind = (job["metadata"].get("annotations") or {}).get(f"{SUITE.group}/kind", "stress")
        return super().get(name, report_inline=kind != "compat")

    def create_compat(self, body: dict) -> dict:
        argv, form = compat_argv(body, self.cfg.defaults)
        return self.start(argv, form, body, prefix="jfcompat", script="/app/jfcompat.py",
                          kind="compat", cli_name="jfcompat.py")


def main() -> None:
    cfg = Config()
    webui.serve(SUITE, cfg, Runs(cfg, webui.Kube()))


if __name__ == "__main__":
    main()
