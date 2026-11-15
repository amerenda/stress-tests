"""The mealie-parser run: Mealie's own parser code against a model, scored.

Mealie's code is used for everything but the database and the HTTP call: OpenAIService builds
the prompt (built-in text, overrides, unit list), OpenAIIngredients parses the reply, and
OpenAIParser computes confidence and matches foods and units, all fed from Mealie's API instead
of its database. The request mirrors OpenAIService._get_raw_response (Mealie v3.30.0) so the raw
reply, token usage and failure reason can be recorded.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import statistics
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from rapidfuzz import fuzz

from mealie.db.models.recipe.ingredient import IngredientUnitModel
from mealie.schema.group.ai_providers import AIPromptMode
from mealie.schema.openai.recipe_ingredient import OpenAIIngredients
from mealie.schema.recipe.recipe_ingredient import IngredientFood, IngredientUnit
from mealie.services.openai.openai import GLOBAL_PROMPT_NAME, OpenAIService
from mealie.services.parser_services._base import DataMatcher
from mealie.services.parser_services.openai.parser import OpenAIParser
from stresskit.output import emit, emit_report, watch_stop_file

PROMPT_NAME = "recipes.parse-recipe-ingredients"
HERE = Path(__file__).resolve().parent
GOLD_FILE = HERE / "gold" / "strangetom.jsonl"
SEED_UNITS = Path(OpenAIService.PROMPTS_DIR).parents[2] / "repos/seed/resources/units/locales/en-US.json"
LOW_CONFIDENCE = 0.8
STOPWORDS = {"a", "an", "and", "or", "of", "the", "to", "for", "about", "plus", "into", "in", "with", "x"}


def builtin_prompt() -> str:
    return OpenAIService.get_default_prompt(PROMPT_NAME)


# --- output ---------------------------------------------------------------------------------


class Out:
    """Events and progress: JSON lines with --jsonl, plain lines otherwise."""

    def __init__(self, jsonl: bool):
        self.jsonl = jsonl
        self.t0 = time.monotonic()
        self.progress = {"phase": "setup", "done": 0, "total": 0, "recipes_ok": 0, "recipes_failed": 0,
                         "gold_lines": 0, "gold_ok": 0, "current": None}

    def event(self, msg: str, **fields):
        if self.jsonl:
            emit({"type": "event", "t": round(time.monotonic() - self.t0, 1), "msg": msg, **fields})
        else:
            print(msg, flush=True)

    def detail(self, msg: str):
        """Human output only (the JSON report carries the same data)."""
        if not self.jsonl:
            print(msg, flush=True)

    def status(self):
        emit({"type": "status", "t": round(time.monotonic() - self.t0, 1), **self.progress})


# --- Mealie data ------------------------------------------------------------------------------


class Mealie:
    """Recipes, foods and units. With MEALIE_TOKEN: the full API; without: the public explore API."""

    def __init__(self, url: str, group: str, token: str):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.http = httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=60)
        self.authed = bool(token)
        self.recipes_path = "/api/recipes" if token else f"/api/explore/groups/{group}/recipes"
        self.foods_path = "/api/foods" if token else f"/api/explore/groups/{group}/foods"

    def _get(self, path, **params):
        r = self.http.get(path, params=params)
        r.raise_for_status()
        return r.json()

    def _all(self, path, **params):
        items, page = [], 1
        while True:
            d = self._get(path, page=page, perPage=100, **params)
            items += d["items"]
            if page >= d.get("total_pages", 1):
                return items
            page += 1

    def recipes(self, tags):
        params = {"tags": tags} if tags else {}
        return self._all(self.recipes_path, orderBy="created_at", orderDirection="asc", **params)

    def recipe(self, slug):
        return self._get(f"{self.recipes_path}/{slug}")

    def foods(self):
        return self._all(self.foods_path)

    def units(self):
        return self._all("/api/units") if self.authed else None


def seed_units():
    """Mealie's default units, for when the units API needs a token we do not have."""
    data = json.loads(SEED_UNITS.read_text())
    # Mealie's schema wants UUIDv4 ids; derive stable ones from the seed keys.
    return [{"id": str(uuid.UUID(bytes=hashlib.md5(k.encode()).digest(), version=4)), **v} for k, v in data.items()]


def ingredient_text(ing):
    """What Mealie would parse for a stored ingredient: the original line if it was kept."""
    if ing.get("originalText"):
        return ing["originalText"].strip()
    if not ing.get("food") and not ing.get("unit") and not ing.get("quantity"):
        return (ing.get("note") or "").strip()
    return (ing.get("display") or ing.get("note") or "").strip()


# --- Mealie's parser, without a database --------------------------------------------------------


def build_matcher(foods, units):
    dm = DataMatcher.__new__(DataMatcher)
    dm.repos = None
    dm._food_fuzzy_match_threshold = 85
    dm._unit_fuzzy_match_threshold = 70
    dm._foods_by_id = {f.id: f for f in foods}
    dm._units_by_id = {u.id: u for u in units}
    dm._foods_by_alias = None
    dm._units_by_alias = None
    return dm


def build_parser(dm):
    parser = OpenAIParser.__new__(OpenAIParser)
    parser.data_matcher = dm
    return parser


def build_prompt(args, dm):
    svc = OpenAIService.__new__(OpenAIService)
    overrides = {}
    if args.global_instructions:
        overrides[GLOBAL_PROMPT_NAME] = SimpleNamespace(mode=AIPromptMode.append, content=args.global_instructions)
    if args.prompt:
        overrides[PROMPT_NAME] = SimpleNamespace(mode=AIPromptMode(args.prompt_mode), content=args.prompt)
    svc.__dict__["_prompt_overrides"] = overrides
    matcher = dm if args.inject_units else build_matcher([], [])
    return build_parser(matcher)._get_prompt(svc)


# --- the request --------------------------------------------------------------------------------


@dataclass
class Reply:
    ok: bool
    category: str = ""
    error: str = ""
    latency_ms: float = 0.0
    content: str | None = None
    reasoning_chars: int = 0
    usage: dict = field(default_factory=dict)
    served_by: str | None = None
    parsed: OpenAIIngredients | None = None


async def ask(client, model, prompt, lines) -> Reply:
    message = json.dumps(lines, separators=(",", ":"))  # as OpenAIParser._parse
    t0 = time.monotonic()
    try:
        async with client.chat.completions.with_streaming_response.parse(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "text", "text": message}]},
            ],
            model=model,
            response_format=OpenAIIngredients,
        ) as response:
            data = json.loads(await response.text())
    except openai.APITimeoutError as e:
        return Reply(False, "timeout", str(e), (time.monotonic() - t0) * 1000)
    except openai.APIStatusError as e:
        return Reply(False, f"http_{e.status_code}", str(e)[:500], (time.monotonic() - t0) * 1000)
    except openai.APIConnectionError as e:
        return Reply(False, "connection", str(e)[:500], (time.monotonic() - t0) * 1000)
    latency = (time.monotonic() - t0) * 1000

    if isinstance(data, dict):
        data.pop("service_tier", None)
    message0 = ((data.get("choices") or [{}])[0] or {}).get("message") or {}
    reasoning = message0.get("reasoning_content") or message0.get("reasoning") or ""
    completion = ChatCompletion.model_validate(data)
    usage = completion.usage.model_dump() if completion.usage else {}
    r = Reply(False, latency_ms=latency, usage=usage, served_by=completion.model, reasoning_chars=len(reasoning))
    if not completion.choices:
        r.category, r.error = "empty", "no choices"
        return r
    choice = completion.choices[0]
    r.content = choice.message.content
    if choice.finish_reason == "length":
        r.category, r.error = "length", f"finish_reason=length after {usage.get('completion_tokens')} tokens"
        return r
    if choice.finish_reason == "content_filter":
        r.category, r.error = "content_filter", "finish_reason=content_filter"
        return r
    try:
        r.parsed = OpenAIIngredients.parse_openai_response(r.content)
    except Exception as e:  # noqa: BLE001 - Mealie fails the import on any parse error
        content = r.content or ""
        r.category = "think_leak" if ("<think>" in content or "</think>" in content) else "json_invalid"
        r.error = f"{type(e).__name__}: {str(e)[:300]}"
        return r
    if len(r.parsed.ingredients) != len(lines):
        r.category = "count_mismatch"
        r.error = f"expected {len(lines)} ingredients, got {len(r.parsed.ingredients)}"
        return r
    r.ok = True
    return r


async def warmup(client, args, out: Out):
    """Wait until the model answers, so a model swap is not counted against the first case."""
    t0 = time.monotonic()
    last = ""
    while time.monotonic() - t0 < args.warmup_timeout:
        try:
            await client.with_options(max_retries=0, timeout=args.timeout).chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": "ok"}], max_tokens=1,
            )
            return time.monotonic() - t0, None
        except openai.APIStatusError as e:
            last = f"HTTP {e.status_code}: {str(e)[:200]}"
            if e.status_code not in (429, 502, 503, 504):
                return time.monotonic() - t0, last
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            last = f"{type(e).__name__}: {e}"
        out.event(f"waiting for {args.model}: {last}")
        await asyncio.sleep(15)
    return time.monotonic() - t0, f"model not ready after {args.warmup_timeout}s: {last}"


# --- scoring --------------------------------------------------------------------------------


def singular(w):
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith(("oes", "ches", "shes", "sses", "xes")):
        return w[:-2]
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def words(s):
    return [singular(w) for w in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", (s or "").lower())]


def unit_key(dm, s):
    if not s:
        return None
    hit = dm.units_by_alias.get(IngredientUnitModel.normalize(s))
    return f"id:{hit.id}" if hit else " ".join(words(s))


def coverage(dm, text, ing):
    """Share of the line's words (not numbers, units or stopwords) that appear in any parsed field."""
    produced = set(words(" ".join(filter(None, [ing.food, ing.unit, ing.note, *ing.substitutes]))))
    wanted = [w for w in words(text)
              if w not in STOPWORDS and IngredientUnitModel.normalize(w) not in dm.units_by_alias]
    if not wanted:
        return 1.0
    return sum(1 for w in wanted if w in produced) / len(wanted)


def food_ok(pred, gold):
    """Same food: the label's last word (the head noun) is in the answer and the words mostly agree."""
    g, p = words(gold), words(pred)
    if not g or not p:
        return not g and not p
    return g[-1] in p and fuzz.token_set_ratio(" ".join(p), " ".join(g)) >= 90


def score_gold(dm, gold, ing):
    qty_ok = abs((ing.quantity or 0) - gold["qty"]) < 0.011
    unit_ok = unit_key(dm, ing.unit) == unit_key(dm, gold["unit"])
    f_ok = food_ok(ing.food, gold["food"])
    if gold["alternatives"]:
        alt_ok = all(any(food_ok(s, a) for s in ing.substitutes) for a in gold["alternatives"])
    else:
        alt_ok = not ing.substitutes
    return {"qty_ok": qty_ok, "unit_ok": unit_ok, "food_ok": f_ok, "alternatives_ok": alt_ok,
            "ok": qty_ok and unit_ok and f_ok}


def dump_ing(ing):
    return {"quantity": ing.quantity, "unit": ing.unit, "food": ing.food, "note": ing.note,
            "substitutes": ing.substitutes}


def pct(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


# --- the run ------------------------------------------------------------------------------------


class Run:
    def __init__(self, args, out: Out):
        self.args, self.out = args, out
        self.stop = asyncio.Event()
        self.stopped_by_user = False
        self.cases: list[dict] = []
        self.latencies: list[float] = []
        self.recipe_fail: dict[str, int] = {}
        self.ingredients: list[tuple[float, float]] = []
        self.food_hits = self.unit_hits = self.unit_total = 0
        self.gold: list = []
        self.gold_batches = 0
        self.gold_batch_fail: dict[str, int] = {}
        self.served_by = None

    def _record(self, r: Reply):
        self.latencies.append(r.latency_ms)
        self.served_by = self.served_by or r.served_by

    async def recipes(self, client, prompt, dm, parser, mealie):
        a, out = self.args, self.out
        summaries = []
        if a.recipes != "none":
            summaries = mealie.recipes(a.tag)
            if a.recipes != "all":
                summaries = random.Random(a.seed).sample(summaries, min(int(a.recipes), len(summaries)))
        out.progress.update(phase="recipes", done=0, total=len(summaries) + bool(a.lines))
        if a.lines:
            out.event(f"your lines: {len(a.lines)}")
            await self._recipe(client, prompt, dm, parser, "recipe:custom", "your lines", a.lines)
        if summaries:
            out.event(f"recipes: {len(summaries)} (tags: {', '.join(a.tag) or 'any'})")
        for s in summaries:
            if self.stop.is_set():
                return
            out.progress["current"] = s["slug"]
            recipe = mealie.recipe(s["slug"])
            lines = [t for t in (ingredient_text(i) for i in recipe.get("recipeIngredient") or []) if t]
            await self._recipe(client, prompt, dm, parser, f"recipe:{s['slug']}", recipe.get("name"), lines)

    async def _recipe(self, client, prompt, dm, parser, case_id, name, lines):
        """One request for one recipe's lines, like an import, then Mealie's own post-processing."""
        out = self.out
        slug = case_id.split(":", 1)[1]
        case = {"id": case_id, "group": "recipes", "status": "skip", "metrics": {},
                "detail": {"name": name, "lines": len(lines)}}
        self.cases.append(case)
        if not lines:
            out.event(f"{slug}: no ingredients, skipped")
            out.progress["done"] += 1
            return
        r = await ask(client, self.args.model, prompt, lines)
        self._record(r)
        out.progress["done"] += 1
        m = case["metrics"]
        m.update(ingredient_count=len(lines), latency_ms=round(r.latency_ms),
                 completion_tokens_count=r.usage.get("completion_tokens", 0) or 0, ok_count=int(r.ok))
        if not r.ok:
            case["status"] = "fail"
            case["detail"].update(category=r.category, error=r.error, content_head=(r.content or "")[:600],
                                  reasoning_chars=r.reasoning_chars)
            self.recipe_fail[r.category] = self.recipe_fail.get(r.category, 0) + 1
            out.progress["recipes_failed"] += 1
            out.event(f"FAIL {slug}: {r.category} ({r.error[:160]}), {r.latency_ms / 1000:.0f}s",
                      case=case_id, ok=False, category=r.category)
            return
        case["status"] = "pass"
        rows, confs, covs, food_hits, unit_hits, unit_total = [], [], [], 0, 0, 0
        for text, ing in zip(lines, r.parsed.ingredients):
            parsed = parser._convert_ingredient(text, ing)
            conf = parsed.confidence.average or 0
            cov = coverage(dm, text, ing)
            confs.append(conf)
            covs.append(cov)
            food_hits += int(bool(parsed.ingredient.food and getattr(parsed.ingredient.food, "id", None)))
            if ing.unit:
                unit_total += 1
                unit_hits += int(bool(parsed.ingredient.unit and getattr(parsed.ingredient.unit, "id", None)))
            rows.append({"input": text, **dump_ing(ing), "confidence": round(conf, 3), "coverage": round(cov, 3)})
        m.update(avg_confidence_rate=round(statistics.fmean(confs), 4),
                 low_confidence_count=sum(1 for c in confs if c < LOW_CONFIDENCE),
                 coverage_rate=round(statistics.fmean(covs), 4),
                 food_matched_count=food_hits, unit_matched_count=unit_hits, unit_parsed_count=unit_total)
        case["detail"]["ingredients"] = rows
        self.ingredients += list(zip(confs, covs))
        self.food_hits += food_hits
        self.unit_hits += unit_hits
        self.unit_total += unit_total
        out.progress["recipes_ok"] += 1
        out.event(f"ok {slug}: {len(lines)} lines, confidence {m['avg_confidence_rate']:.2f}, "
                  f"coverage {m['coverage_rate']:.2f}, {m['low_confidence_count']} low, {r.latency_ms / 1000:.0f}s",
                  case=case_id, ok=True)
        for row in rows:
            if row["confidence"] < LOW_CONFIDENCE or row["coverage"] < 1:
                out.detail(f"     ? {row['input']!r} -> qty={row['quantity']} unit={row['unit']!r} "
                           f"food={row['food']!r} note={row['note']!r} subs={row['substitutes']}")

    async def gold_part(self, client, prompt, dm):
        a, out = self.args, self.out
        if a.gold_sample <= 0:
            return
        pool = [json.loads(line) for line in GOLD_FILE.read_text().splitlines() if line.strip()]
        sample = random.Random(a.seed).sample(pool, min(a.gold_sample, len(pool)))
        batches = [sample[i:i + a.gold_batch] for i in range(0, len(sample), a.gold_batch)]
        out.progress.update(phase="gold", done=0, total=len(batches), current=None)
        out.event(f"gold: {len(sample)} lines in {len(batches)} batches of {a.gold_batch}")
        misses = []
        for bi, batch in enumerate(batches, 1):
            if self.stop.is_set():
                return
            out.progress["current"] = f"batch {bi}"
            r = await ask(client, a.model, prompt, [g["input"] for g in batch])
            self._record(r)
            self.gold_batches += 1
            out.progress["done"] += 1
            out.progress["gold_lines"] += len(batch)
            if not r.ok:
                self.gold_batch_fail[r.category] = self.gold_batch_fail.get(r.category, 0) + 1
                out.event(f"gold batch {bi}: FAIL {r.category} ({r.error[:160]})", ok=False, category=r.category)
                for g in batch:
                    self.cases.append({"id": f"gold:{g['id']}", "group": "gold", "status": "fail",
                                       "metrics": {"ok_count": 0},
                                       "detail": {"input": g["input"], "category": r.category}})
                    self.gold.append(None)
                continue
            good = 0
            for g, ing in zip(batch, r.parsed.ingredients):
                sc = score_gold(dm, g, ing)
                cov = coverage(dm, g["input"], ing)
                self.gold.append((sc, cov))
                good += sc["ok"]
                case = {"id": f"gold:{g['id']}", "group": "gold", "status": "pass" if sc["ok"] else "fail",
                        "metrics": {k + "_count": int(v) for k, v in sc.items()} | {"coverage_rate": round(cov, 3)}}
                if not sc["ok"] or not sc["alternatives_ok"]:
                    case["detail"] = {"input": g["input"], "expected": {k: g[k] for k in ("qty", "unit", "food", "alternatives")},
                                      "got": dump_ing(ing),
                                      "wrong": [k[:-3] for k in ("qty_ok", "unit_ok", "food_ok", "alternatives_ok") if not sc[k]]}
                    misses.append(case)
                self.cases.append(case)
            out.progress["gold_ok"] += good
            out.event(f"gold batch {bi}: {good}/{len(batch)} right, {r.latency_ms / 1000:.0f}s", ok=True)
        for c in misses[:40]:
            d = c["detail"]
            out.detail(f"  - {d['input']!r}\n      wrong: {', '.join(d['wrong'])}\n"
                       f"      want: {d['expected']}\n      got:  {d['got']}")

    def metrics(self) -> dict:
        m = {}
        recipes = [c for c in self.cases if c["group"] == "recipes" and c["status"] != "skip"]
        if recipes:
            ok = sum(1 for c in recipes if c["status"] == "pass")
            m.update(recipes_count=len(recipes), recipes_ok_count=ok, recipes_ok_rate=round(ok / len(recipes), 4))
            for cat, n in self.recipe_fail.items():
                m[f"recipes_fail_{cat}_count"] = n
            if self.ingredients:
                confs = [c for c, _ in self.ingredients]
                m.update(ingredients_count=len(confs),
                         ingredient_confidence_rate=round(statistics.fmean(confs), 4),
                         ingredient_low_confidence_count=sum(1 for c in confs if c < LOW_CONFIDENCE),
                         ingredient_coverage_rate=round(statistics.fmean(v for _, v in self.ingredients), 4),
                         food_matched_rate=round(self.food_hits / len(confs), 4))
            if self.unit_total:
                m["unit_matched_rate"] = round(self.unit_hits / self.unit_total, 4)
        if self.gold:
            n = len(self.gold)
            good = [s for s in self.gold if s]
            m.update(gold_lines_count=n, gold_batches_count=self.gold_batches,
                     gold_batch_fail_count=sum(self.gold_batch_fail.values()))
            for k in ("ok", "qty_ok", "unit_ok", "food_ok", "alternatives_ok"):
                m[f"gold_{k}_rate"] = round(sum(1 for s in good if s[0][k]) / n, 4)
            if good:
                m["gold_coverage_rate"] = round(statistics.fmean(c for _, c in good), 4)
            for cat, k in self.gold_batch_fail.items():
                m[f"gold_fail_{cat}_count"] = k
        if self.latencies:
            m.update(latency_p50_ms=round(pct(self.latencies, 0.5)), latency_max_ms=round(max(self.latencies)),
                     requests_count=len(self.latencies))
        return m


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class StageFailure(Exception):
    pass


def summary_lines(args, doc) -> list[str]:
    m = doc["metrics"]
    lines = [f"run {doc['run_id']}  model {args.model}  status {doc['status']}",
             f"prompt: {'custom (' + args.prompt_mode + ')' if args.prompt else 'Mealie built-in'}"
             f"{' + global instructions' if args.global_instructions else ''}, units injected: {args.inject_units}"]
    if "recipes_count" in m:
        lines.append(f"recipes: {m['recipes_ok_count']}/{m['recipes_count']} parsed"
                     + "".join(f", {k[13:-6]} {v}" for k, v in m.items() if k.startswith("recipes_fail_")))
        if "ingredients_count" in m:
            lines.append(f"ingredients: {m['ingredients_count']}, confidence {m['ingredient_confidence_rate']:.2f}, "
                         f"{m['ingredient_low_confidence_count']} below {LOW_CONFIDENCE}, coverage "
                         f"{m['ingredient_coverage_rate']:.2f}, food matched {m['food_matched_rate']:.2f}")
    if "gold_lines_count" in m:
        lines.append(f"gold: {m['gold_ok_rate']:.0%} fully right of {m['gold_lines_count']} "
                     f"(qty {m['gold_qty_ok_rate']:.0%}, unit {m['gold_unit_ok_rate']:.0%}, food {m['gold_food_ok_rate']:.0%}, "
                     f"alternatives {m['gold_alternatives_ok_rate']:.0%}), {m['gold_batch_fail_count']} failed batches")
    if "latency_p50_ms" in m:
        lines.append(f"latency p50 {m['latency_p50_ms'] / 1000:.0f}s, max {m['latency_max_ms'] / 1000:.0f}s, "
                     f"warmup {m.get('warmup_s', 0):.0f}s")
    if doc["error"]:
        lines.append(f"{doc['failed_stage']}: {doc['error']}")
    return lines


async def amain(args, out: Out) -> dict:
    started = datetime.now(timezone.utc)
    # In a Job the pod name (the Job name plus a suffix) identifies the run.
    run_id = args.run_id or os.environ.get("HOSTNAME") or f"mealie-parser-{uuid.uuid4().hex[:8]}"
    params = {k: getattr(args, k) for k in ("model", "base_url", "prompt", "prompt_mode", "global_instructions",
                                             "inject_units", "recipes", "tag", "lines", "seed", "gold_sample", "gold_batch",
                                             "timeout", "max_retries", "mealie_url")}
    params["mealie_source"] = "api" if os.environ.get("MEALIE_TOKEN") else "explore"
    doc = {"schema": "eval-result.v1", "run_id": run_id, "kind": "ingredients", "status": "pass",
           "failed_stage": None, "error": None, "started_at": iso(started), "finished_at": iso(started),
           "duration_s": 0.0, "tags": list(args.label),
           "workflow": {"template": "stress-tests/mealie-parser", "namespace": os.environ.get("POD_NAMESPACE", ""),
                        "image": os.environ.get("IMAGE", "")},
           "subject": {"host": "", "engine": "", "target": args.model, "served_by": None, "weights": None,
                       "load": {}, "switch_run_id": None},
           "params": params, "metrics": {}, "cases": [], "extra": {}}
    run = Run(args, out)
    doc["cases"] = run.cases
    stage = "fetch"
    watcher = status = None
    try:
        mealie = Mealie(args.mealie_url, args.mealie_group, os.environ.get("MEALIE_TOKEN", "").strip())
        foods = [IngredientFood.model_validate(f) for f in mealie.foods()]
        raw_units = mealie.units()
        units_source = "mealie" if raw_units is not None else "seed"
        units = [IngredientUnit.model_validate(u) for u in (raw_units if raw_units is not None else seed_units())]
        dm = build_matcher(foods, units)
        parser = build_parser(dm)
        prompt = build_prompt(args, dm)
        doc["extra"].update(system_prompt=prompt, system_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                            units_source=units_source, units_count=len(units), foods_count=len(foods))
        out.event(f"model {args.model}, {len(foods)} foods, {len(units)} units ({units_source})")
        out.detail("---- system prompt ----\n" + prompt + "\n-----------------------")

        watcher = asyncio.create_task(watch_stop_file(args.stop_file, run.stop, on_stop=lambda: (
            setattr(run, "stopped_by_user", True), out.event("stop requested"))))
        if args.jsonl:
            status = asyncio.create_task(_status_loop(out, args.status_interval))

        stage = "warmup"
        out.progress["phase"] = "warmup"
        client = AsyncOpenAI(base_url=args.base_url, api_key="unused", timeout=args.timeout,
                             max_retries=args.max_retries)
        warm_s, err = await warmup(client, args, out)
        doc["metrics"]["warmup_s"] = round(warm_s, 1)
        if err:
            raise StageFailure(err)
        out.event(f"{args.model} answered after {warm_s:.0f}s")

        stage = "recipes"
        await run.recipes(client, prompt, dm, parser, mealie)
        stage = "gold"
        await run.gold_part(client, prompt, dm)
        doc["metrics"].update(run.metrics())
    except StageFailure as e:
        doc["status"], doc["failed_stage"], doc["error"] = "fail", stage, str(e)
    except httpx.HTTPError as e:
        doc["status"], doc["failed_stage"], doc["error"] = "fail", stage, f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        doc["status"], doc["failed_stage"], doc["error"] = "error", stage, f"{type(e).__name__}: {e}"
        doc["metrics"].update(run.metrics())
    finally:
        for t in (watcher, status):
            if t:
                t.cancel()
    out.progress.update(phase="done", current=None)
    doc["subject"]["served_by"] = run.served_by
    doc["extra"]["stopped_by_user"] = run.stopped_by_user
    finished = datetime.now(timezone.utc)
    doc["finished_at"] = iso(finished)
    doc["duration_s"] = round((finished - started).total_seconds(), 1)
    return doc


async def _status_loop(out: Out, interval: float):
    while True:
        out.status()
        await asyncio.sleep(interval)


def result_of(doc: dict) -> dict:
    """The small summary kept on the Job: status and the run-level metrics."""
    return {"run_id": doc["run_id"], "status": doc["status"], "failed_stage": doc["failed_stage"],
            "error": doc["error"], "model": doc["subject"]["target"], "metrics": doc["metrics"],
            "stored": doc["extra"].get("stored"), "stopped_by_user": doc["extra"].get("stopped_by_user"),
            "duration_s": doc["duration_s"]}


def run(args) -> int:
    out = Out(args.jsonl)
    doc = asyncio.run(amain(args, out))
    if args.store and doc["status"] != "error":
        if os.environ.get("PGHOST"):
            import evalstore

            try:
                evalstore.store(doc)
                doc["extra"]["stored"] = True
                out.event(f"stored as {doc['run_id']} in {os.environ.get('PGDATABASE', 'model_eval')}")
            except Exception as e:  # noqa: BLE001 - the run itself is still worth reporting
                doc["extra"]["stored"] = False
                out.event(f"storing the result failed: {type(e).__name__}: {e}")
        else:
            out.event("not stored: PGHOST is not set")
    if args.out:
        Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    if args.jsonl:
        out.status()
        emit_report(doc, result_of(doc))
    else:
        print("\n== summary", flush=True)
        for line in summary_lines(args, doc):
            print("  " + line, flush=True)
    return 0 if doc["status"] != "error" else 1
