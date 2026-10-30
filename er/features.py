"""Pairwise features for the matcher."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .blocking import soundex

FEATURES = [
    "given_jw", "given_ratio", "surname_jw", "surname_ratio", "name_swapped",
    "address_ratio", "suburb_ratio", "postcode_exact", "postcode_dist",
    "dob_exact", "dob_dist", "surname_soundex_eq", "n_missing",
]


def _col(df, c):
    return df[c].fillna("").astype(str).str.strip().str.lower().to_numpy()


def pair_features(df, pairs) -> pd.DataFrame:
    g, s, a, sub, pc, dob = (_col(df, c) for c in
                             ["given_name", "surname", "address", "suburb", "postcode", "date_of_birth"])
    sx = np.array([soundex(x) for x in s])
    rows = []
    for i, j in pairs:
        gi, gj, si, sj = g[i], g[j], s[i], s[j]
        miss = sum(1 for arr in (g, s, a, sub, pc, dob) if not arr[i] or not arr[j])
        rows.append((
            JaroWinkler.normalized_similarity(gi, gj) if gi and gj else 0.0,
            fuzz.ratio(gi, gj) / 100 if gi and gj else 0.0,
            JaroWinkler.normalized_similarity(si, sj) if si and sj else 0.0,
            fuzz.ratio(si, sj) / 100 if si and sj else 0.0,
            float(bool(gi and si and gi == sj and si == gj)),
            fuzz.token_sort_ratio(a[i], a[j]) / 100 if a[i] and a[j] else 0.0,
            fuzz.ratio(sub[i], sub[j]) / 100 if sub[i] and sub[j] else 0.0,
            float(bool(pc[i]) and pc[i] == pc[j]),
            Levenshtein.distance(pc[i], pc[j]) if pc[i] and pc[j] else 6,
            float(bool(dob[i]) and dob[i] == dob[j]),
            Levenshtein.distance(dob[i], dob[j]) if dob[i] and dob[j] else 8,
            float(bool(sx[i]) and sx[i] == sx[j]),
            miss,
        ))
    return pd.DataFrame(rows, columns=FEATURES)


def baseline_score(X: pd.DataFrame) -> np.ndarray:
    """Hand-weighted rule-based score (the 'v1' approach) used as the baseline."""
    return (0.35 * X["given_ratio"] + 0.35 * X["surname_ratio"] + 0.30 * X["address_ratio"]).to_numpy()
