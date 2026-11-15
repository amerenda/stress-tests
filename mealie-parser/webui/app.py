#!/usr/bin/env python3
"""mealie-parser web UI, on the shared stresskit web UI library.

Every mealie_parser.py option is a form field (the prompt is a text area), each run is a
Kubernetes Job running `mealie_parser.py --jsonl`, and the page shows live progress and the
result: recipes parsed, failure reasons, Mealie's confidence per ingredient and the gold scores.
Runs store their result in the model-eval Postgres through RUNNER_ENV (PGHOST, PGPASSWORD...).

Config (env): stresskit.webui's (NAMESPACE, RUNNER_IMAGE, UI_DEFAULTS, NODE_SELECTOR,
TOLERATIONS, RUNNER_RESOURCES, RUNNER_ENV, JOB_TTL_SECONDS, MAX_RUN_SECONDS).
"""
from __future__ import annotations

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))):  # the suite, then the repo
    if path not in sys.path:
        sys.path.insert(1, path)

import mealie_parser  # noqa: E402
from stresskit import webui  # noqa: E402

PROMPT_FILE = ("services", "openai", "prompts", "recipes", "parse-recipe-ingredients.txt")

PRESETS = [
    {"group": "Mealie recipes", "name": "Quick check", "values": {"recipes": "3", "gold_sample": 0},
     "note": "3 random recipes, nothing else: a fast look at a prompt change (a few minutes)."},
    {"group": "Mealie recipes", "name": "All recipes", "values": {"recipes": "all", "gold_sample": 0},
     "note": "Every recipe in Mealie (or every one with the tags below), one request each, like re-importing them."},
    {"group": "Labelled lines", "name": "Gold 100", "values": {"recipes": "none", "gold_sample": 100},
     "note": "100 labelled lines in batches of 10, scored on quantity, unit and food."},
    {"group": "Labelled lines", "name": "Gold 300", "values": {"recipes": "none", "gold_sample": 300},
     "note": "300 labelled lines: a steadier score for comparing two prompts."},
    {"group": "Both", "name": "Full", "values": {"recipes": "all", "gold_sample": 200},
     "note": "Every recipe, then 200 labelled lines. The run to keep for a prompt you like."},
]


def builtin_prompt() -> str | None:
    """Mealie's built-in ingredient prompt, read from the installed package (no Mealie import)."""
    spec = importlib.util.find_spec("mealie")
    for base in (spec.submodule_search_locations or []) if spec else []:
        path = os.path.join(base, *PROMPT_FILE)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
    return None


class MealieParserSuite(webui.Suite):
    name = "mealie-parser"
    group = "mealie-parser.amer.dev"
    title = "Mealie ingredient parser"
    script = "/stress/mealie_parser.py"
    cli_name = "mealie_parser.py"
    default_namespace = "model-eval"
    hidden = {"out", "run_id"}
    presets = PRESETS
    preset_keep = ("model", "base_url", "timeout", "max_retries", "warmup_timeout", "prompt", "prompt_mode",
                   "global_instructions", "inject_units", "mealie_url", "mealie_group", "label", "store", "tag",
                   "lines")
    html_dir = HERE

    def build_parser(self):
        return mealie_parser.build_parser()

    def parse_args(self, argv, parser):
        return mealie_parser.parse_args(argv, parser=parser)

    def columns(self, form, kind):
        prompt = (form.get("prompt") or "").strip()
        lines = [ln for ln in str(form.get("lines") or "").splitlines() if ln.strip()]
        return {"model": form.get("model"), "recipes": form.get("recipes"), "tags": form.get("tag") or [],
                "lines": len(lines),
                "gold_sample": form.get("gold_sample"), "labels": form.get("label") or [],
                "prompt": f"custom ({form.get('prompt_mode') or 'override'})" if prompt else "built-in"}

    def result_summary(self, rep):
        return {"run_id": rep.get("run_id"), "status": rep.get("status"), "failed_stage": rep.get("failed_stage"),
                "error": rep.get("error"), "model": (rep.get("subject") or {}).get("target"),
                "metrics": rep.get("metrics") or {}, "stored": (rep.get("extra") or {}).get("stored"),
                "stopped_by_user": (rep.get("extra") or {}).get("stopped_by_user"),
                "duration_s": rep.get("duration_s")}

    def extra_options(self, cfg):
        return {"builtin_prompt": builtin_prompt()}


SUITE = MealieParserSuite()


def main() -> None:
    webui.serve(SUITE)


if __name__ == "__main__":
    main()
