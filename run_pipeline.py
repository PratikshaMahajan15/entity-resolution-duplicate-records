"""End-to-end entity resolution pipeline.

    python run_pipeline.py --data febrl3 --seeds 5     # real benchmark (needs recordlinkage)
    python run_pipeline.py --data synthetic            # Faker-generated data

Steps: load -> multi-pass blocking -> pair features -> rule baseline vs logistic regression vs
gradient boosting -> evaluation on held-out entities (repeated over several random splits) ->
clustering -> PR curve, feature importance and error analysis (saved to results/).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from er.blocking import blocking_report, candidate_pairs
from er.data import FIELDS, generate_synthetic, load_febrl3
from er.evaluate import best_threshold, cluster_pairs, pair_metrics, pr_curve, true_pairs
from er.features import baseline_score, pair_features

OUT = Path("results")
ML = ("logistic_regression", "gradient_boosting")


def make_models(seed):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, learning_rate=0.1, random_state=seed),
    }


def run_split(seed, ctx):
    """One entity-level train/test split: tune thresholds on train (CV), report on test."""
    X, y, I, J, pairs, entity, truth, union = (ctx[k] for k in
        ("X", "y", "I", "J", "pairs", "entity", "truth", "union"))

    rng = np.random.default_rng(seed)
    ents = np.unique(entity).copy()
    rng.shuffle(ents)
    train_ents = set(ents[: int(0.7 * len(ents))])
    is_tr = np.fromiter((e in train_ents for e in entity), bool, len(entity))
    tr = np.where(is_tr[I] & is_tr[J])[0]
    te = np.where(~is_tr[I] & ~is_tr[J])[0]
    truth_tr = {p for p in truth if is_tr[p[0]]}
    truth_te = truth - truth_tr
    ceiling = len(union & truth_te) / len(truth_te)

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    models = make_models(seed)
    scores_te, thr, cv_f1 = {}, {}, {}

    thr["rule_baseline"], cv_f1["rule_baseline"] = best_threshold(
        baseline_score(X.iloc[tr]), y[tr], len(truth_tr))
    scores_te["rule_baseline"] = baseline_score(X.iloc[te])
    for name, model in models.items():
        p_tr = cross_val_predict(model, X.iloc[tr], y[tr], cv=cv, method="predict_proba")[:, 1]
        thr[name], cv_f1[name] = best_threshold(p_tr, y[tr], len(truth_tr))
        model.fit(X.iloc[tr], y[tr])
        scores_te[name] = model.predict_proba(X.iloc[te])[:, 1]

    rows, preds = [], {}
    for name, s in scores_te.items():
        preds[name] = {pairs[k] for k, sc in zip(te, s) if sc >= thr[name]}
        rows.append({"seed": seed, "model": name, "threshold": thr[name], "cv_f1_train": cv_f1[name],
                     "blocking_ceiling": ceiling, **pair_metrics(preds[name], truth_te)})

    best = max(ML, key=lambda n: cv_f1[n])
    closed, n_clusters = cluster_pairs(preds[best])
    cm = pair_metrics(closed, truth_te)
    return dict(rows=rows, best=best, cluster=cm, n_clusters=n_clusters, te=te, tr=tr,
                truth_te=truth_te, scores_te=scores_te, preds=preds, models=models, ceiling=ceiling)


def main(args):
    OUT.mkdir(exist_ok=True)
    df = load_febrl3() if args.data == "febrl3" else generate_synthetic(seed=args.seed)
    entity = df["entity_id"].to_numpy()
    truth = true_pairs(entity)
    print(f"[data] {args.data}: {len(df)} records, {len(set(entity))} entities, "
          f"{len(truth)} true duplicate pairs")

    # ---- 1. blocking ---------------------------------------------------------
    union, per_pass = candidate_pairs(df)
    rep = blocking_report(df, union, per_pass, truth)
    rep.to_csv(OUT / "blocking.csv", index=False)
    print("\n[blocking]\n" + rep.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 2. features ---------------------------------------------------------
    pairs = sorted(union)
    X = pair_features(df, pairs)
    y = np.array([p in truth for p in pairs], dtype=int)
    I, J = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    print(f"\n[features] {len(pairs)} candidate pairs, {y.sum()} true matches among them")
    ctx = dict(X=X, y=y, I=I, J=J, pairs=pairs, entity=entity, truth=truth, union=union)

    # ---- 3. repeated entity-level splits --------------------------------------
    runs = [run_split(args.seed + k, ctx) for k in range(args.seeds)]
    res = pd.DataFrame([r for run in runs for r in run["rows"]])
    res.to_csv(OUT / "metrics_by_seed.csv", index=False)
    summ = res.groupby("model")[["precision", "recall", "f1"]].agg(["mean", "std"])
    summ.to_csv(OUT / "metrics_summary.csv")
    order = ["rule_baseline", "logistic_regression", "gradient_boosting"]
    print(f"\n[test results over {args.seeds} random entity splits, mean ± std]")
    for m in order:
        s = summ.loc[m]
        print(f"  {m:20s} P={s[('precision','mean')]:.4f}±{s[('precision','std')]:.4f}  "
              f"R={s[('recall','mean')]:.4f}±{s[('recall','std')]:.4f}  "
              f"F1={s[('f1','mean')]:.4f}±{s[('f1','std')]:.4f}")
    cf = [r["cluster"]["f1"] for r in runs]
    cr = [r["cluster"]["recall"] for r in runs]
    print(f"  mean blocking ceiling on test recall: {np.mean([r['ceiling'] for r in runs]):.4f}")
    print(f"  after clustering (best model per split): recall={np.mean(cr):.4f}±{np.std(cr, ddof=1) if len(cr)>1 else 0:.4f} "
          f"F1={np.mean(cf):.4f}±{np.std(cf, ddof=1) if len(cf)>1 else 0:.4f}")

    # ---- detailed outputs from the first split ---------------------------------
    r0 = runs[0]
    te, truth_te, scores_te, preds, best = r0["te"], r0["truth_te"], r0["scores_te"], r0["preds"], r0["best"]

    plt.figure(figsize=(6.5, 5))
    for name, s in scores_te.items():
        _, p, r, _ = pr_curve(s, y[te], len(truth_te))
        plt.plot(r, p, label=name)
    plt.axvline(r0["ceiling"], ls="--", c="grey", label=f"blocking ceiling ({r0['ceiling']:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall ({args.data}, held-out entities)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT / "pr_curve.png", dpi=150); plt.close()

    # feature importance: LR coefficients (standardised) + permutation importance for boosting
    lr = r0["models"]["logistic_regression"].named_steps["logisticregression"]
    gb = r0["models"]["gradient_boosting"]
    pi = permutation_importance(gb, X.iloc[te], y[te], scoring="average_precision",
                                n_repeats=5, random_state=args.seed)
    fi = pd.DataFrame({"feature": X.columns, "lr_coef_standardised": lr.coef_[0],
                       "gb_permutation_importance": pi.importances_mean}
                      ).sort_values("gb_permutation_importance", ascending=False)
    fi.to_csv(OUT / "feature_importance.csv", index=False)
    plt.figure(figsize=(6.5, 5))
    top = fi.head(10).iloc[::-1]
    plt.barh(top["feature"], top["gb_permutation_importance"])
    plt.xlabel("Drop in average precision when shuffled"); plt.title("Feature importance (gradient boosting)")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()
    print("\n[feature importance - top 5]\n" + fi.head(5).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # error analysis (first split, best model)
    rows = []
    cand_te = {pairs[k] for k in te}
    fp = sorted(preds[best] - truth_te)[:25]
    fn = sorted((truth_te & cand_te) - preds[best])[:25]
    lost = sorted(truth_te - union)[:25]
    for kind, plist in (("false_positive", fp), ("false_negative", fn), ("lost_in_blocking", lost)):
        for i, j in plist:
            row = {"kind": kind}
            for f in FIELDS:
                row[f"{f}_A"], row[f"{f}_B"] = df.at[i, f], df.at[j, f]
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "errors.csv", index=False)
    print(f"\n[errors, split 1, {best}] FP={len(preds[best] - truth_te)} "
          f"FN(in blocks)={len((truth_te & cand_te) - preds[best])} "
          f"lost-in-blocking={len(truth_te - union)} -> results/errors.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["febrl3", "synthetic"], default="febrl3")
    ap.add_argument("--seed", type=int, default=42, help="first split seed")
    ap.add_argument("--seeds", type=int, default=5, help="number of random entity splits")
    main(ap.parse_args())
