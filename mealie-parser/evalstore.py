"""Store an eval-result.v1 document in the model-eval Postgres.

Writes the same rows as evallib.store in dean-amer-dev/k3s-images (images/model-eval): one row
in `results` (the whole document in `doc`) and one row per metric in `metrics`, run-level
metrics with case_id ''. Uses psycopg2, which the Mealie image already has, and the standard
PG* environment variables. A run id that is already stored is replaced.

The tables belong to evallib's migrations; this only creates them (as evallib's 001_init.sql
would, and records that migration) on an empty database.
"""
from __future__ import annotations

import json

import psycopg2
from psycopg2.extras import Json

INIT = """
CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS results (
    run_id text PRIMARY KEY, kind text NOT NULL, status text NOT NULL, host text, engine text, target text,
    repo text, revision text, file text, context_size int, switch_run_id text,
    tags text[] NOT NULL DEFAULT '{}', started_at timestamptz, finished_at timestamptz,
    params jsonb, metrics jsonb, doc jsonb NOT NULL, inserted_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS metrics (
    run_id text NOT NULL REFERENCES results(run_id) ON DELETE CASCADE, case_id text NOT NULL DEFAULT '',
    name text NOT NULL, value double precision NOT NULL, PRIMARY KEY (run_id, case_id, name));
CREATE INDEX IF NOT EXISTS results_kind_host_started ON results (kind, host, started_at);
INSERT INTO schema_migrations (name) VALUES ('001_init.sql') ON CONFLICT DO NOTHING;
"""


def rows(doc: dict) -> tuple[tuple, list[tuple]]:
    subj = doc.get("subject") or {}
    w = subj.get("weights") or {}
    rid = doc["run_id"]
    result = (rid, doc["kind"], doc["status"], subj.get("host"), subj.get("engine"), subj.get("target"),
              w.get("repo"), w.get("revision"), w.get("file"), (subj.get("load") or {}).get("contextSize"),
              subj.get("switch_run_id"), doc.get("tags") or [], doc["started_at"], doc["finished_at"],
              Json(doc.get("params")) if doc.get("params") is not None else None, Json(doc["metrics"]),
              Json(json.loads(json.dumps(doc, default=str))))
    metrics = [(rid, "", n, float(v)) for n, v in doc["metrics"].items()]
    for c in doc.get("cases") or []:
        metrics += [(rid, c["id"], n, float(v)) for n, v in (c.get("metrics") or {}).items()]
    return result, metrics


def store(doc: dict) -> None:
    result, metrics = rows(doc)
    conn = psycopg2.connect("")
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.results')")
            if cur.fetchone()[0] is None:
                cur.execute(INIT)
            cur.execute("DELETE FROM results WHERE run_id = %s", (doc["run_id"],))
            cur.execute(
                "INSERT INTO results (run_id, kind, status, host, engine, target, repo, revision, file, "
                "context_size, switch_run_id, tags, started_at, finished_at, params, metrics, doc) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", result)
            cur.executemany("INSERT INTO metrics (run_id, case_id, name, value) VALUES (%s,%s,%s,%s)", metrics)
    finally:
        conn.close()
