"""Build gold/strangetom.jsonl from strangetom/ingredient-parser's training.sqlite3.

Keeps only lines whose labels map cleanly onto Mealie's fields (one quantity, at most a
one-word unit right after it, a food), then takes a fixed stratified sample per source.

    curl -LO https://raw.githubusercontent.com/strangetom/ingredient-parser/7c4c47a8cd8fbf3e13873fd1996b99985167b844/train/data/training.sqlite3
    python gold/make_gold.py training.sqlite3 gold/strangetom.jsonl

training.sqlite3 at that commit: sha256 f4bef553ddd9ae734fd8a0c8ee9b299606071cc5e9ad9fa9aa96fac5f3339a99
"""

import json
import random
import re
import sqlite3
import sys

DATASET = "strangetom/ingredient-parser@7c4c47a8cd8fbf3e13873fd1996b99985167b844:train/data/training.sqlite3"
PER_SOURCE = {"nyt": 300, "allrecipes": 300, "bbc": 300, "cookstr": 300, "tc": 200, "ba": 100, "manual": 50}
WORD_QTY = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1}
MULTI_WORD_UNITS = {"fluid ounce"}
NAME = {"B_NAME_TOK", "I_NAME_TOK"}


def parse_qty(tok):
    """'2', '#1$4', '1#1$2', '2-3', '#1$2-#3$4', '4x', 'One' -> lower bound as float, else None."""
    t = tok.strip().lower().rstrip("x")
    if t in WORD_QTY:
        return float(WORD_QTY[t])
    t = t.split("-")[0]
    m = re.fullmatch(r"(\d+)?#(\d+)\$(\d+)", t)
    if m:
        whole = float(m.group(1) or 0)
        return whole + float(m.group(2)) / float(m.group(3))
    if re.fullmatch(r"\d+(\.\d+)?", t):
        return float(t)
    return None


def spans(tokens, labels):
    """Name spans: a B_NAME_TOK and the I_NAME_TOKs after it."""
    out, cur = [], None
    for tok, lab in zip(tokens, labels):
        if lab == "B_NAME_TOK":
            if cur:
                out.append(cur)
            cur = [tok]
        elif lab == "I_NAME_TOK" and cur is not None:
            cur.append(tok)
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return [" ".join(s) for s in out]


def gold_for(sentence, tokens, labels):
    qty_idx = [i for i, lab in enumerate(labels) if lab == "QTY"]
    unit_idx = [i for i, lab in enumerate(labels) if lab == "UNIT"]
    names = spans(tokens, labels)
    if not names:
        return None
    if unit_idx and not qty_idx:
        return None  # "Pinch of salt": quantity 0 vs 1 is ambiguous
    qty, unit = 0.0, None
    if qty_idx:
        first = qty_idx[0]
        qty = parse_qty(tokens[first])
        if qty is None:
            return None
        run = []
        for tok, lab in zip(tokens[first + 1:], labels[first + 1:]):
            if lab != "UNIT":
                break
            run.append(tok)
        if run:
            unit = " ".join(run)
            if len(run) > 1 and unit.lower() not in MULTI_WORD_UNITS:
                return None
        elif unit_idx and unit_idx[0] < labels.index("B_NAME_TOK"):
            return None  # unit before the food but not right after the quantity
    # The food is every name token before the first "or" (NAME_SEP), so "boneless, skinless chicken
    # breasts" stays one food; the names after it are the alternatives.
    sep = labels.index("NAME_SEP") if "NAME_SEP" in labels else len(labels)
    food = " ".join(tok for tok, lab in zip(tokens[:sep], labels[:sep]) if lab in NAME)
    if not food:
        return None
    alternatives = spans(tokens[sep:], labels[sep:]) if sep < len(labels) else []
    return {"qty": qty, "unit": unit, "food": food, "alternatives": alternatives}


def main(db, out):
    conn = sqlite3.connect(db)
    rng = random.Random(20261115)
    pool = []
    for source, n in PER_SOURCE.items():
        rows = conn.execute(
            "SELECT id, sentence, tokens, labels FROM en WHERE source = ? AND sentence_split = '[]' ORDER BY id",
            (source,),
        ).fetchall()
        rng.shuffle(rows)
        kept = 0
        for rid, sentence, tokens, labels in rows:
            g = gold_for(sentence, json.loads(tokens), json.loads(labels))
            if not g:
                continue
            pool.append({"id": f"{source}-{rid}", "source": source, "input": sentence, **g})
            kept += 1
            if kept >= n:
                break
    with open(out, "w") as f:
        for rec in pool:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{len(pool)} lines from {DATASET}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
