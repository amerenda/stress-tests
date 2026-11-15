# stress-tests

Tests that push a self-hosted service the way its users do and report what they would see. Each
suite is a CLI you can run by hand, and a small web UI that runs the same CLI as Kubernetes Jobs.

| Suite | What it tests | UI |
|---|---|---|
| [`jellyfin/`](jellyfin/) | Jellyfin playback under load: concurrent streams, transcodes, stalls; library compatibility per device | https://jellyfin.stress.amer.dev |
| [`mealie-parser/`](mealie-parser/) | Mealie's AI ingredient parser against a local model, with an editable prompt: Mealie's recipes and labelled lines | https://mealie.stress.amer.dev |

## The common interface

Every suite follows the same contract, implemented once in [`stresskit/`](stresskit/):

**The CLI** (`stresskit/output.py`)

- Options are a plain `argparse` parser from a `build_parser()` function, plus a `parse_args(argv, parser)`
  that does any extra checks through `parser.error`. Whatever the parser accepts is a valid run.
- `--jsonl` prints JSON lines on stdout instead of terminal output: `event` lines as things happen,
  a `status` line every `--status-interval` seconds, then the full report as gzipped `report_part`
  lines and a small `result` line last (`emit`, `emit_report`). Other output is shown as-is.
- `--stop-file PATH` ends the run gracefully, with a report, once the file says `true`
  (`watch_stop_file`, `stop_requested`).

**The web UI** (`stresskit/webui.py`)

A suite subclasses `Suite` (name, script, presets, how to summarize a run) and gets:

- a form built from the CLI's parser, so a new flag appears in the UI with no UI change
  (`metavar="TEXT"` makes a multi-line text area), validated by the CLI's own `parse_args`;
- each run as a Job (`--jsonl --stop-file=/podinfo/stop`), with node placement, run env and limits from tofu;
- run history (finished Jobs are kept for `JOB_TTL_SECONDS`), live status, Stop (pod annotation →
  downward API file → stop file), Kill/Delete, the report and the log as downloads;
- the JSON API listed in `webui.py`. Each suite keeps its own `index.html`.

UI configuration comes from the environment set by tofu: `NAMESPACE`, `RUNNER_IMAGE`, `UI_DEFAULTS`,
`NODE_SELECTOR`, `TOLERATIONS`, `RUNNER_RESOURCES`, `RUNNER_ENV` (container env entries for run pods,
`valueFrom` allowed), `JOB_TTL_SECONDS`, `MAX_RUN_SECONDS`.

## Adding a suite

1. `<suite>/<cli>.py` with `build_parser()`, `parse_args()` and `stresskit.output.add_output_args(p)`,
   writing `--jsonl` output and honouring `--stop-file`.
2. `<suite>/webui/app.py` with a `Suite` subclass and `webui.serve(SUITE)`, and `<suite>/webui/index.html`.
3. `<suite>/Dockerfile`, built from the repo root, copying `stresskit/` next to the suite.
4. A job in `.github/workflows/test.yaml`; an image build in `dean-amer-dev/k3s-images`; a dean-tofu root
   for the UI (Deployment, ServiceAccount with Job/Pod rights in its namespace, Service, Ingress).

## Images and deploys

Images are built by `dean-amer-dev/k3s-images` from a pinned commit of this repo
(`images/<suite>/source-ref`) and pushed to Artifact Registry; dean-tofu deploys them.
