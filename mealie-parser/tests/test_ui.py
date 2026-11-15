"""CLI and web UI tests (no Mealie, no cluster): python -m unittest discover -s tests"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "webui"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(ROOT))

import app  # noqa: E402
import mealie_parser  # noqa: E402
from stresskit import webui  # noqa: E402

PROMPT = "Parse ingredient strings.\n\n- Use the lower number of a range\n- \"quoted\" and 'single' stay"


def defaults_form(spec):
    return {f["dest"]: f["default"] for g in spec for f in g["fields"]}


class Cli(unittest.TestCase):
    def test_defaults(self):
        a = mealie_parser.parse_args([])
        self.assertEqual((a.model, a.recipes, a.gold_sample, a.prompt, a.inject_units, a.store),
                         ("qwen38-27b", "all", 100, "", True, True))

    def test_negations(self):
        a = mealie_parser.parse_args(["--no-inject-units", "--no-store"])
        self.assertFalse(a.inject_units)
        self.assertFalse(a.store)

    def test_invalid(self):
        for argv in (["--recipes", "some"], ["--recipes", "0"], ["--recipes", "none", "--gold-sample", "0"],
                     ["--gold-batch", "0"], ["--prompt-mode", "replace"], ["--timeout", "1"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                mealie_parser.parse_args(argv)

    def test_lines_alone_are_a_run(self):
        a = mealie_parser.parse_args(["--recipes", "none", "--gold-sample", "0", "--lines", "1 cup flour\r\n\n 2 eggs "])
        self.assertEqual(a.lines, ["1 cup flour", "2 eggs"])

    def test_tags_and_prompt_are_cleaned(self):
        a = mealie_parser.parse_args(["--tag", " weeknight ", "--tag", " ", "--prompt", "a\r\nb\n"])
        self.assertEqual(a.tag, ["weeknight"])
        self.assertEqual(a.prompt, "a\nb")


class OptionSpec(unittest.TestCase):
    def setUp(self):
        self.spec = webui.option_spec(app.SUITE)
        self.fields = {f["dest"]: f for g in self.spec for f in g["fields"]}

    def test_every_cli_option_is_exposed_except_hidden(self):
        p = mealie_parser.build_parser()
        dests = {a.dest for a in p._actions if a.option_strings} - webui.ALWAYS_HIDDEN - app.SUITE.hidden
        self.assertEqual(dests, set(self.fields))

    def test_kinds(self):
        self.assertEqual(self.fields["prompt"]["kind"], "textarea")
        self.assertEqual(self.fields["global_instructions"]["kind"], "textarea")
        self.assertEqual(self.fields["lines"]["kind"], "textarea")
        self.assertEqual(self.fields["prompt_mode"]["kind"], "choice")
        self.assertEqual(self.fields["inject_units"]["off_flag"], "--no-inject-units")
        self.assertEqual(self.fields["store"]["off_flag"], "--no-store")
        self.assertEqual(self.fields["tag"]["kind"], "list")
        self.assertEqual(self.fields["gold_sample"]["kind"], "int")

    def test_defaults_give_empty_argv(self):
        self.assertEqual(webui.build_argv(defaults_form(self.spec), self.spec), [])

    def test_multiline_prompt_round_trips(self):
        form = defaults_form(self.spec) | {"prompt": PROMPT.replace("\n", "\r\n") + "\n", "prompt_mode": "append",
                                           "inject_units": False, "tag": ["a", "b"], "recipes": "5"}
        argv = webui.build_argv(form, self.spec)
        a = webui.validate_argv(app.SUITE, argv)
        self.assertEqual(a.prompt, PROMPT)
        self.assertEqual((a.prompt_mode, a.inject_units, a.tag, a.recipes), ("append", False, ["a", "b"], "5"))
        self.assertIn("--no-inject-units", argv)
        self.assertIn("--prompt-mode=append", webui.command_line(app.SUITE, argv))

    def test_invalid_form_raises_valueerror(self):
        for bad in ({"recipes": "lots"}, {"gold_sample": "x"}, {"prompt_mode": "bogus"},
                    {"recipes": "none", "gold_sample": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                webui.validate_argv(app.SUITE, webui.build_argv(defaults_form(self.spec) | bad, self.spec))


class Presets(unittest.TestCase):
    def test_every_preset_is_a_valid_run(self):
        spec = webui.option_spec(app.SUITE)
        dests = {f["dest"] for g in spec for f in g["fields"]}
        self.assertLessEqual(set(app.SUITE.preset_keep), dests)
        names = set()
        for p in app.PRESETS:
            with self.subTest(preset=p["name"]):
                self.assertTrue(p["note"] and p["group"])
                self.assertLessEqual(set(p["values"]), dests - set(app.SUITE.preset_keep))
                self.assertNotIn(p["name"], names)
                names.add(p["name"])
                a = webui.validate_argv(app.SUITE, webui.build_argv(defaults_form(spec) | p["values"], spec))
                for k, v in p["values"].items():
                    self.assertEqual(getattr(a, k), v)


class Job(unittest.TestCase):
    def setUp(self):
        os.environ.update({"RUNNER_IMAGE": "img@sha256:abc", "NAMESPACE": "model-eval",
                           "RUNNER_ENV": json.dumps([{"name": "PGHOST", "value": "pg-rw.postgres.svc"}])})
        self.cfg = webui.Config(app.SUITE)

    def tearDown(self):
        for k in ("RUNNER_IMAGE", "NAMESPACE", "RUNNER_ENV"):
            os.environ.pop(k, None)

    def test_job_manifest(self):
        pod = webui.parse_pod({}, self.cfg)
        job = webui.job_manifest(app.SUITE, "mealie-parser-x", ["--recipes=3"], pod, {"recipes": "3"}, self.cfg)
        c = job["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(c["name"], "mealie-parser")
        self.assertEqual(c["command"], ["python", "-u", "/stress/mealie_parser.py"])
        self.assertEqual(c["args"], ["--recipes=3", "--jsonl", "--stop-file=/podinfo/stop"])
        self.assertEqual(c["env"], [{"name": "PGHOST", "value": "pg-rw.postgres.svc"}])
        self.assertIn({"name": "tmp", "mountPath": "/tmp"}, c["volumeMounts"])
        self.assertTrue(webui.is_run(app.SUITE, job))
        self.assertEqual(job["metadata"]["labels"]["app.kubernetes.io/name"], "mealie-parser-run")

    def test_columns_and_result_summary(self):
        cols = app.SUITE.columns({"model": "m", "prompt": " x ", "prompt_mode": "append", "tag": ["t"]}, "stress")
        self.assertEqual((cols["prompt"], cols["tags"]), ("custom (append)", ["t"]))
        self.assertEqual(app.SUITE.columns({}, "stress")["prompt"], "built-in")
        rep = {"run_id": "r", "status": "pass", "metrics": {"recipes_ok_count": 2}, "subject": {"target": "m"},
               "extra": {"stored": True}, "cases": [{"id": "x"}]}
        s = app.SUITE.result_summary(rep)
        self.assertEqual((s["model"], s["stored"], s["metrics"]), ("m", True, {"recipes_ok_count": 2}))
        self.assertNotIn("cases", s)


if __name__ == "__main__":
    unittest.main()
