"""Multi-pass blocking: candidate pairs = union of several cheap blocking keys,
so a typo that breaks one key is rescued by another."""
from collections import defaultdict
from itertools import combinations

import re

import pandas as pd


def soundex(s: str) -> str:
    s = re.sub(r"[^a-z]", "", str(s).lower())
    if not s:
        return ""
    codes = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
             **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}
    out, last = s[0].upper(), codes.get(s[0], "")
    for ch in s[1:]:
        c = codes.get(ch, "")
        if c and c != last:
            out += c
        if ch not in "hw":
            last = c
    return (out + "000")[:4]


def _clean(df, col):
    return df[col].fillna("").astype(str).str.strip().str.lower()


# each pass returns one key string per record ("" = no key -> record skipped)
PASSES = {
    "postcode": lambda df: _clean(df, "postcode").tolist(),
    "surname_soundex+initial": lambda df: [
        f"{soundex(s)}|{g[:1]}" if s else ""
        for s, g in zip(_clean(df, "surname"), _clean(df, "given_name"))],
    "date_of_birth": lambda df: _clean(df, "date_of_birth").tolist(),
    # --- added after error analysis (pairs lost to swapped names / DOB missing / postcode typos) ---
    "suburb": lambda df: _clean(df, "suburb").tolist(),
    "name_prefix_unordered": lambda df: [
        "|".join(sorted([g[:3], s[:3]])) if (g or s) else ""
        for s, g in zip(_clean(df, "surname"), _clean(df, "given_name"))],
}
ORIGINAL = ["postcode", "surname_soundex+initial", "date_of_birth"]


def block_pairs(keys, max_block=300):
    blocks = defaultdict(list)
    for pos, k in enumerate(keys):
        if k:
            blocks[k].append(pos)
    pairs = set()
    for idxs in blocks.values():
        if len(idxs) <= max_block:          # skip giant blocks (e.g. missing/default values)
            pairs.update(combinations(idxs, 2))
    return pairs


def candidate_pairs(df, passes=None, max_block=300):
    passes = passes or list(PASSES)
    per_pass = {name: block_pairs(PASSES[name](df), max_block) for name in passes}
    union = set().union(*per_pass.values())
    return union, per_pass


def blocking_report(df, union, per_pass, truth):
    n = len(df)
    total = n * (n - 1) // 2
    orig_union = set().union(*(per_pass[k] for k in ORIGINAL if k in per_pass))
    rows = []
    summary = {"UNION original 3 passes": orig_union, "UNION all passes": union}
    for name, pairs in {**per_pass, **summary}.items():
        rows.append({
            "blocking": name,
            "candidate_pairs": len(pairs),
            "reduction_ratio": 1 - len(pairs) / total,
            "pair_completeness": len(pairs & truth) / len(truth),
        })
    return pd.DataFrame(rows)
