# Entity Resolution: Duplicate Customer Record Matching

Finds duplicate customer records using multi-pass blocking, fuzzy-string pair features and a
learned matcher, benchmarked against a rule-based baseline on held-out entities (Febrl3).

![PR curve](results/pr_curve.png)

## Pipeline
1. **Data**: Febrl3 benchmark (5,000 records, 2,000 entities, 6,538 true duplicate pairs, realistic typos and
   missing values) or a Faker-based synthetic generator with planted duplicates and hard negatives.
2. **Multi-pass blocking**: candidate pairs are the union of five cheap keys: postcode, Soundex(surname)+initial,
   date of birth, suburb, and a swap-invariant name-prefix key. A typo that breaks one key is rescued by another.
3. **Pair features** (rapidfuzz): Jaro-Winkler/ratio on names, token-sort on address, exact/edit-distance on
   postcode and DOB, name-swap flag, Soundex match, missing-value count.
4. **Models**: hand-weighted rule baseline vs logistic regression vs gradient boosting; thresholds tuned on
   cross-validated train predictions, never on test.
5. **Evaluation**: split by *entity* (no leakage), repeated over several random splits (mean ± std). Recall is
   measured against ALL true pairs, so pairs lost in blocking count as misses. Predicted pairs are then clustered
   (connected components).
6. **Analysis**: PR curve, feature importance, and error analysis (`results/errors.csv`).

## Run
```bash
pip install -r requirements.txt
python run_pipeline.py --data febrl3 --seeds 5
python run_pipeline.py --data synthetic
```
Outputs go to `results/`: `blocking.csv`, `metrics_summary.csv`, `metrics_by_seed.csv`, `pr_curve.png`,
`feature_importance.csv/.png`, `errors.csv`.

## Results
**Iteration 1: 3 blocking passes, single split (seed 42)**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Rule baseline | 0.941 | 0.793 | 0.861 |
| Logistic regression | 0.997 | 0.985 | 0.991 |
| Gradient boosting | 0.998 | 0.986 | 0.992 |

Blocking: postcode alone 0.763 pair completeness; 3-pass union 0.986 (99.8% fewer comparisons: ~12.5M -> 20,860).
After clustering: precision 0.997, recall 0.999, F1 0.998.

**Iteration 2: 5 blocking passes, 5 random splits (mean ± std)**: _fill in from `results/metrics_summary.csv`_

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Rule baseline | | | |
| Logistic regression | | | |
| Gradient boosting | | | |

Blocking pair completeness (5 passes): __ | reduction ratio: __

## Error analysis (iteration 1)
- **81% of errors were pairs lost in blocking** (25 of 31): all had mismatched postcodes, and the rest of the
  keys failed due to swapped given/surname fields, typos that change the Soundex code, or a missing date of birth.
  Most still agreed on suburb and address, which motivated the suburb and swap-invariant name passes in iteration 2.
- **4 false positives** were one cluster of near-identical different people (same surname, postcode and suburb,
  DOBs differing by one digit), arguably label ambiguity.
- **2 false negatives** had a different address, postcode and DOB on each record: essentially unrecoverable.
- The classifier is near-perfect on candidates it sees; the remaining gap is blocking recall.
