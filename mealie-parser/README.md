# mealie-parser

Tests Mealie's AI ingredient parser against a model, with a prompt you can edit per run, so a
prompt can be tuned before it goes into Mealie (Group settings → AI → prompts).

It runs inside the Mealie image and uses Mealie's own code for everything but the database and
the HTTP call: the prompt (built-in text, your override or appended text, group-wide
instructions, the unit list Mealie appends), the request (same messages and `response_format`),
the reply parsing, Mealie's confidence scores and its food/unit matching. A prompt that does well
here does the same in Mealie, as long as the image tag matches the deployed Mealie.

## What a run does

1. **Warm-up.** Waits (up to `--warmup-timeout`) until the model answers, so a model swap on the GPU
   host is not counted against the first case.
2. **Recipes** (`--recipes all|N|none`, `--tag`): Mealie's own recipes, read through its API. One
   request per recipe with all of its ingredient lines, like an import. A recipe passes when Mealie
   would accept the reply. Failures are sorted by cause:
   - `length`: the reply hit the server's output limit (often reasoning that ran on)
   - `think_leak`: the reasoning tag (`</think>`) ended up in the reply, so it is not JSON
   - `json_invalid`, `count_mismatch` (Mealie rejects a reply with a different number of ingredients),
     `timeout`, `http_<code>`, `connection`

   For recipes that pass: Mealie's confidence per ingredient, how many of each line's words ended up
   in some field (Mealie's prompt says all text must be accounted for), and how many foods and units
   matched ones that already exist in Mealie.
3. **Gold** (`--gold-sample`, `--gold-batch`): labelled lines from
   [strangetom/ingredient-parser](https://github.com/strangetom/ingredient-parser) (MIT), sent in batches
   like a recipe and scored on quantity (the lower number of a range), unit (same Mealie unit, or the
   same word), food (the label's head noun present, words mostly agree) and alternatives ("or" foods
   in `substitutes`). Same `--seed`, same lines, so runs are comparable.

The result is an `eval-result.v1` document (kind `ingredients`; metrics like `recipes_ok_rate`,
`gold_ok_rate`, `ingredient_confidence_rate`) with one case per recipe and per gold line, and the exact
system prompt sent. It is the run's report in the UI, and it is stored in the model-eval Postgres
(`results` and `metrics` tables) when `PGHOST` is set, so prompts can be compared over time:

```sql
SELECT run_id, started_at, tags, metrics->>'recipes_ok_rate' AS recipes, metrics->>'gold_ok_rate' AS gold,
       left(params->>'prompt', 60) AS prompt
FROM results WHERE kind = 'ingredients' ORDER BY started_at DESC;
```

## Mealie access

Without a token it reads the public explore API of `--mealie-group`: public recipes, foods and tags.
Units then come from Mealie's default unit list. With `MEALIE_TOKEN` (a Mealie API token) it reads
every recipe and your real units through the full API.

## Running

In the web UI: pick a preset (it keeps your prompt and model), edit the prompt (*Load Mealie's prompt
to edit* fills in the built-in one), start. *Use these settings* on a finished run copies its form back.

By hand, inside the image (it needs Mealie's package):

```sh
docker run --rm -e PGHOST= ... mealie-parser python /stress/mealie_parser.py --recipes 3 --gold-sample 50 \
  --prompt "$(cat my-prompt.txt)" --prompt-mode override
```

## Files

| File | |
|---|---|
| `mealie_parser.py` | CLI: options and checks (no Mealie import, so the UI can read the options) |
| `engine.py` | the run: Mealie's parser without a database, the request, scoring, output |
| `evalstore.py` | writes the result to the model-eval Postgres (same rows as `evallib.store`) |
| `gold/make_gold.py` | builds `gold/strangetom.jsonl` (1,537 labelled lines) from the pinned dataset; the image build runs it |
| `webui/` | the suite's `stresskit.webui` UI |
| `tests/` | `test_ui.py` runs anywhere; `test_engine.py` runs in the image build |
