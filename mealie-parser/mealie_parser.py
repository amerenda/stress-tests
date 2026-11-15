#!/usr/bin/env python3
"""mealie-parser: test Mealie's AI ingredient parser against a model, with an editable prompt.

Runs inside the Mealie image and drives Mealie's own code for the prompt (built-in text, group
overrides, unit list), the request (same messages and response_format), the reply parsing, the
confidence scores and the food/unit matching (see engine.py). Two parts, each optional:

  recipes  Mealie's own recipes: all, a random sample, or those with given tags, plus any lines you
           paste (--lines). One request per recipe, like an import. Scored on whether the recipe
           parses and on Mealie's confidence.
  gold     Labelled lines from strangetom/ingredient-parser (gold/strangetom.jsonl), sent in
           recipe-sized batches, scored on quantity, unit and food against the labels.

The result is an eval-result.v1 document (kind "ingredients"): printed as the report with
--jsonl, written with --out, and stored in the model-eval Postgres when PGHOST is set.

  ./mealie_parser.py --recipes 3 --gold-sample 50
  ./mealie_parser.py --prompt "$(cat my-prompt.txt)" --tag weeknight --gold-sample 0
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.dirname(HERE)):  # the suite, then the repo (stresskit)
    if path not in sys.path:
        sys.path.insert(1, path)

from stresskit.output import add_output_args  # noqa: E402

ROUTER = "http://llm-router-proxy.llm.svc.cluster.local:8080/v1"
MEALIE = "http://mealie.mealie.svc.cluster.local:9000"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mealie_parser.py", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("model")
    g.add_argument("--model", default="qwen38-27b", help="model name as Mealie's provider sends it")
    g.add_argument("--base-url", default=ROUTER, help="OpenAI-compatible endpoint (Mealie's provider base URL)")
    g.add_argument("--timeout", type=int, default=300, help="per-request timeout s (Mealie's provider timeout)")
    g.add_argument("--max-retries", type=int, default=0,
                   help="client retries per request (Mealie's client does 2; 0 shows every failure)")
    g.add_argument("--warmup-timeout", type=int, default=900,
                   help="s to wait for the model to answer before the first case (covers a model swap)")

    g = p.add_argument_group("prompt")
    g.add_argument("--prompt", metavar="TEXT", default="",
                   help="your version of Mealie's ingredient prompt; empty uses the built-in one")
    g.add_argument("--prompt-mode", choices=("override", "append"), default="override",
                   help="override replaces the built-in prompt, append adds to it (Mealie's two prompt modes)")
    g.add_argument("--global-instructions", metavar="TEXT", default="",
                   help="Mealie's group-wide AI instructions, appended to every prompt")
    g.add_argument("--inject-units", dest="inject_units", action="store_true", default=True,
                   help="append Mealie's unit list to the prompt, as Mealie does")
    g.add_argument("--no-inject-units", dest="inject_units", action="store_false", help=argparse.SUPPRESS)

    g = p.add_argument_group("recipes")
    g.add_argument("--recipes", default="all", help="all, a number to sample at random, or none")
    g.add_argument("--tag", action="append", default=[], metavar="SLUG",
                   help="only recipes with any of these tags (Mealie tag slugs)")
    g.add_argument("--seed", type=int, default=1, help="random seed for the recipe and gold samples")
    g.add_argument("--lines", metavar="TEXT", default="",
                   help="your own ingredient lines, one per line, parsed first as one extra recipe")

    g = p.add_argument_group("gold")
    g.add_argument("--gold-sample", type=int, default=100, help="labelled lines to score (0 skips this part)")
    g.add_argument("--gold-batch", type=int, default=10, help="lines per request, like a recipe's ingredient list")

    g = p.add_argument_group("mealie")
    g.add_argument("--mealie-url", default=MEALIE, help="Mealie base URL (recipes, foods; units with a token)")
    g.add_argument("--mealie-group", default="home",
                   help="group slug for the public API, used when MEALIE_TOKEN is not set")

    g = p.add_argument_group("results")
    g.add_argument("--label", action="append", default=[], help="tag for this run in the results database")
    g.add_argument("--store", dest="store", action="store_true", default=True,
                   help="store the result in the model-eval Postgres (needs PGHOST and friends)")
    g.add_argument("--no-store", dest="store", action="store_false", help=argparse.SUPPRESS)
    g.add_argument("--run-id", default="", help="result id (default: the pod name, or a random id)")
    g.add_argument("--out", help="write the result JSON here")

    add_output_args(p, status_interval=10)
    return p


def parse_args(argv=None, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    p = parser or build_parser()
    a = p.parse_args(argv)
    r = a.recipes.strip().lower()
    if r not in ("all", "none") and not (r.isdigit() and int(r) > 0):
        p.error("--recipes must be all, none or a positive number")
    a.recipes = r
    if a.gold_sample < 0 or a.gold_batch < 1:
        p.error("--gold-sample must be 0 or more and --gold-batch at least 1")
    a.lines = [ln.strip() for ln in a.lines.replace("\r\n", "\n").split("\n") if ln.strip()]
    if a.recipes == "none" and a.gold_sample == 0 and not a.lines:
        p.error("nothing to run: --recipes none, --gold-sample 0 and no --lines")
    if a.timeout < 10 or a.warmup_timeout < 0:
        p.error("--timeout must be at least 10 s")
    a.tag = [t.strip() for t in a.tag if t.strip()]
    a.prompt = a.prompt.replace("\r\n", "\n").strip()
    a.global_instructions = a.global_instructions.replace("\r\n", "\n").strip()
    return a


def main() -> int:
    args = parse_args()
    import engine  # Mealie's code: only importable inside the Mealie image

    return engine.run(args)


if __name__ == "__main__":
    sys.exit(main())
