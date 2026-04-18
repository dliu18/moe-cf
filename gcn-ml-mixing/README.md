# gcn-ml-mixing

Utilities for MovieLens-1M split creation, LightGCN data-mixing cross-validation sweeps, test-set re-evaluation, and analysis.

## Files

- `create_ml1m_lightgcn_splits.py`
  - creates shared test split + `k` validation folds
- `run_ml1m_mix_sweep.py`
  - runs baseline + random data-mixing trials across all validation folds
- `evaluate_top_mixes_test.py`
  - evaluates test performance from validation-selected top mixes or proportional perturbation trials
- `analyze_mixing_validation.ipynb`
  - analysis figures for validation and test outputs
- `run_lightgcn_ml1m.py`
  - single-run utility for parsing one LightGCN run

## 1) Create K-Fold Splits

```bash
python gcn-ml-mixing/create_ml1m_lightgcn_splits.py --num-folds 5
```

Optional:

```bash
python gcn-ml-mixing/create_ml1m_lightgcn_splits.py \
  --data-dir data \
  --output-dir LightGCN/data/ml-1m \
  --seed 42 \
  --num-folds 5
```

Outputs in `LightGCN/data/ml-1m`:

- `train_full.txt`
- `test.txt`
- `train_fold_0.txt` ... `train_fold_4.txt`
- `val_fold_0.txt` ... `val_fold_4.txt`
- compatibility aliases: `train.txt` and `val.txt` (fold 0)
- `user_labels.pkl`

Assertions include:

- per-user non-empty train/val/test where required
- for each fold: `train_fold_i + val_fold_i + test == full interactions`
- max user index and max item index are identical across all split files

## 2) Run Validation Mix Sweep (Cross-Validation)

```bash
python gcn-ml-mixing/run_ml1m_mix_sweep.py \
  --feature-name Age \
  --source-label 56 \
  --trials 1000 \
  --epochs 40 \
  --layer 1 \
  --num-folds 5
```

Default CSV output:

- `gcn-ml-mixing/validation/ml1m_mixing_results__feature-<feature>__source-<source>.csv`

Behavior:

- reuses existing split files and labels when present
- verifies max user/item index alignment across all split files
- evaluates baselines:
  - `no_augmentation` (`alpha_aug=0`)
  - `proportional` (`alpha_aug=1`, `alpha_g=n_g/n_aug`)
  - `stratified` (`alpha_aug=1`, uniform over augmentation groups)
- random trials:
  - `alpha_aug ~ Uniform(0.5, 2.0)`
  - `alpha_mix ~ Dirichlet(0.7)`
- runs each alpha on every fold index (`cv_fold = 0..k-1`)
- appends one row per `(trial, fold)` with metrics and `training_time_seconds`

Validation CSV compatibility note:

- Validation CSV files ending with `single-val.csv` were generated before cross-validation was added.
- Those legacy files do not include `cv_fold` and `num_folds` columns in their headers.

Runtime note:

- In our current environment, an 8-hour run completes about 75 trials (with `num_folds=5`).

## 3) Evaluate Top Validation Mixes On Test

```bash
python gcn-ml-mixing/evaluate_top_mixes_test.py \
  --feature-name Age \
  --source-label 56 \
  --top-k 200
```

Default input/output:

- reads validation CSVs from `gcn-ml-mixing/validation`
- writes:
  - `gcn-ml-mixing/test/ml1m_test_eval__feature-<feature>__source-<source>.csv`

Selection behavior:

- baselines are always included
- top-`k` non-baseline alphas are ranked by validation metric averaged across CV folds
- default ranking metric is first `recall@k` column
- override with `--selection-metric` (example: `recall@20`)

Important distinction:

- `--top-k` means how many alpha settings to choose
- `--topks` means ranking cutoff(s) used by LightGCN (example: `"[20]"`)

### Proportional Perturbation Mode

`evaluate_top_mixes_test.py` also supports a dedicated test mode that runs only
`proportional_perturbation` trials (no baselines or top-k selections in that run):

```bash
python gcn-ml-mixing/evaluate_top_mixes_test.py \
  --feature-name Age \
  --source-label 1 \
  --mode proportional_perturbation \
  --proportional-perturbation-trials 375 \
  --perturbation-scale 0.25
```

Behavior:

- sets `alpha_aug=1`
- starts from proportional `alpha_mix = x`
- samples `delta ~ Uniform([0,1]^(g-1))`, centers by subtracting mean, scales by `--perturbation-scale`
- uses `alpha_mix = x + delta'`, resampling until all ratios are non-negative
- appends results to the same test CSV output
- resume-safe: if rerun with the same `(feature, source, perturbation_scale)`, it only runs remaining trials up to `--proportional-perturbation-trials`

## 4) LightGCN Fold Selection

LightGCN now supports selecting validation fold index when `eval_split=val`:

- `--eval_split=val --val_split_idx=<i>`

When `--eval_split=test`, training uses `train_full.txt` if present.

## 5) Analyze Outputs

Notebook:

- `gcn-ml-mixing/analyze_mixing_validation.ipynb`

Assumes:

- validation CSVs in `gcn-ml-mixing/validation`
- test CSVs in `gcn-ml-mixing/test`

Includes:

- real alpha-mix distributions
- simulated Dirichlet alpha-mix distributions (multiple concentration overlays)
- metric density panels
- top-vs-all distance distributions
- best alpha-mix by alpha-aug regime
- grouped-bar test summary (with % deltas vs proportional baseline)
