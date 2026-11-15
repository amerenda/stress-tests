"""Engine tests: need Mealie's package, so they run in the image build (Dockerfile test stage)."""
import asyncio
import json
import os
import sys
import unittest

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

try:
    import engine
    from mealie.schema.openai.recipe_ingredient import OpenAIIngredient
    from mealie.schema.recipe.recipe_ingredient import IngredientFood, IngredientUnit
except ImportError:  # outside the Mealie image
    engine = None

import mealie_parser  # noqa: E402


def matcher():
    units = [IngredientUnit.model_validate(u) for u in engine.seed_units()]
    foods = [IngredientFood.model_validate({"id": "8d3c4a3e-2b1f-4c1a-9a7e-1f2e3d4c5b6a", "name": "garlic"})]
    return engine.build_matcher(foods, units)


def ing(**kw):
    return OpenAIIngredient(**kw)


def completion(content, finish="stop", reasoning=None):
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "served-model",
            "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def client_returning(body, status=200, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json=body)

    from openai import AsyncOpenAI
    return AsyncOpenAI(base_url="http://test/v1", api_key="x", max_retries=0,
                       http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@unittest.skipIf(engine is None, "needs the Mealie image")
class Scoring(unittest.TestCase):
    def setUp(self):
        self.dm = matcher()

    def test_seed_units_are_valid_and_aliased(self):
        self.assertIsNotNone(self.dm.units_by_alias.get("tbsp"))
        self.assertEqual(engine.unit_key(self.dm, "tablespoons"), engine.unit_key(self.dm, "tbsp"))
        self.assertEqual(engine.unit_key(self.dm, "cloves"), engine.unit_key(self.dm, "clove"))
        self.assertIsNone(engine.unit_key(self.dm, None))

    def test_food(self):
        self.assertTrue(engine.food_ok("chicken breasts", "boneless skinless chicken breasts"))
        self.assertTrue(engine.food_ok("soft light brown sugar", "sugar"))
        self.assertFalse(engine.food_ok("chicken", "chicken stock"))
        self.assertFalse(engine.food_ok(None, "onion"))

    def test_score_gold(self):
        gold = {"qty": 0.5, "unit": "cup", "food": "dry white wine", "alternatives": []}
        sc = engine.score_gold(self.dm, gold, ing(quantity=0.5, unit="cups", food="dry white wine"))
        self.assertTrue(sc["ok"] and sc["alternatives_ok"])
        sc = engine.score_gold(self.dm, gold, ing(quantity=1, unit="cup", food="white wine", substitutes=["sherry"]))
        self.assertEqual((sc["qty_ok"], sc["unit_ok"], sc["food_ok"], sc["alternatives_ok"]), (False, True, True, False))
        alt = {"qty": 1, "unit": None, "food": "plaice", "alternatives": ["lemon sole"]}
        self.assertTrue(engine.score_gold(self.dm, alt, ing(quantity=1, food="plaice", substitutes=["lemon sole"]))["alternatives_ok"])

    def test_coverage(self):
        line = "2 tbsp olive oil, plus extra for frying"
        self.assertEqual(engine.coverage(self.dm, line, ing(quantity=2, unit="tbsp", food="olive oil",
                                                            note="plus extra for frying")), 1.0)
        self.assertLess(engine.coverage(self.dm, line, ing(quantity=2, unit="tbsp", food="olive oil")), 1.0)

    def test_ingredient_text(self):
        self.assertEqual(engine.ingredient_text({"originalText": " 1 cup flour ", "note": "x"}), "1 cup flour")
        self.assertEqual(engine.ingredient_text({"note": "1 cup flour", "quantity": 0}), "1 cup flour")
        self.assertEqual(engine.ingredient_text({"food": {"name": "flour"}, "display": "1 cup flour", "quantity": 1}),
                         "1 cup flour")

    def test_gold_pool(self):
        rows = [json.loads(line) for line in engine.GOLD_FILE.read_text().splitlines()]
        self.assertGreater(len(rows), 1000)
        self.assertEqual(len({r["id"] for r in rows}), len(rows))
        for r in rows[:50]:
            self.assertTrue(r["input"] and r["food"])


@unittest.skipIf(engine is None, "needs the Mealie image")
class Prompt(unittest.TestCase):
    def test_builtin_with_units(self):
        p = engine.build_prompt(mealie_parser.parse_args([]), matcher())
        self.assertTrue(p.startswith(engine.builtin_prompt().strip()[:40]))
        self.assertIn("units database", p)
        self.assertIn("tbsp", p)

    def test_override_append_global_and_no_units(self):
        dm = matcher()
        p = engine.build_prompt(mealie_parser.parse_args(["--prompt", "MY PROMPT", "--no-inject-units"]), dm)
        self.assertTrue(p.startswith("MY PROMPT"))
        self.assertNotIn("units database", p)
        p = engine.build_prompt(mealie_parser.parse_args(["--prompt", "EXTRA", "--prompt-mode", "append",
                                                          "--global-instructions", "GLOBAL"]), dm)
        self.assertLess(p.index("Parse ingredient"), p.index("GLOBAL"))
        self.assertLess(p.index("GLOBAL"), p.index("EXTRA"))


@unittest.skipIf(engine is None, "needs the Mealie image")
class Ask(unittest.TestCase):
    def ask(self, body, lines=("1 cup flour",), status=200, seen=None):
        return asyncio.run(engine.ask(client_returning(body, status, seen), "m", "PROMPT", list(lines)))

    def test_ok_sends_what_mealie_sends(self):
        seen = []
        r = self.ask(completion(json.dumps({"ingredients": [{"quantity": 1, "unit": "cup", "food": "flour"}]}),
                                reasoning="hmm"), seen=seen)
        self.assertTrue(r.ok)
        self.assertEqual((r.served_by, r.reasoning_chars, r.usage["completion_tokens"]), ("served-model", 3, 20))
        req = seen[0]
        self.assertEqual(req["messages"][0], {"role": "system", "content": "PROMPT"})
        self.assertEqual(req["messages"][1]["content"], [{"type": "text", "text": '["1 cup flour"]'}])
        self.assertEqual(req["response_format"]["type"], "json_schema")

    def test_failures(self):
        good = json.dumps({"ingredients": [{"food": "flour"}]})
        cases = [(completion("</think>\n\n" + good), "think_leak"), (completion("not json"), "json_invalid"),
                 (completion(good, finish="length"), "length"),
                 (completion(json.dumps({"ingredients": []})), "count_mismatch")]
        for body, cat in cases:
            with self.subTest(cat=cat):
                r = self.ask(body)
                self.assertEqual((r.ok, r.category), (False, cat))
        r = self.ask({"error": {"message": "model pool incumbent busy"}}, status=503)
        self.assertEqual(r.category, "http_503")

    def test_markdown_fence_is_accepted_like_mealie(self):
        r = self.ask(completion("```json\n" + json.dumps({"ingredients": [{"food": "flour"}]}) + "\n```"))
        self.assertTrue(r.ok)


@unittest.skipIf(engine is None, "needs the Mealie image")
class Store(unittest.TestCase):
    def test_rows_match_evallib_columns(self):
        import evalstore

        doc = {"run_id": "r1", "kind": "ingredients", "status": "pass", "started_at": "2026-11-15T00:00:00Z",
               "finished_at": "2026-11-15T00:01:00Z", "tags": ["a"], "params": {"model": "m"},
               "subject": {"target": "m", "load": {}}, "metrics": {"recipes_ok_rate": 0.5, "gold_lines_count": 3},
               "cases": [{"id": "recipe:x", "metrics": {"ok_count": 1}}, {"id": "gold:y"}]}
        result, metrics = evalstore.rows(doc)
        self.assertEqual(len(result), 17)
        self.assertEqual(result[:3] + (result[5], result[11]), ("r1", "ingredients", "pass", "m", ["a"]))
        self.assertEqual(sorted(metrics), [("r1", "", "gold_lines_count", 3.0), ("r1", "", "recipes_ok_rate", 0.5),
                                           ("r1", "recipe:x", "ok_count", 1.0)])


if __name__ == "__main__":
    unittest.main()
