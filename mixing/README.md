# mixing

Utilities for MovieLens-1M split creation, LightGCN data-mixing cross-validation sweeps, test-set re-evaluation, and analysis.

## Files

- `data_preproc/movielens.py`
  - creates shared test split + `k` validation folds
- `mix_sweep.py`
  - runs baseline + random data-mixing trials across all validation folds
- `submit_mix_sweep_sbatch.sh`
  - submits one `sbatch` job per source label (`1, 18, 25, 35, 45, 56`) on `dean` with RTX 6000 Ada GPUs
- `evaluate_top_mixes_test.py`
  - evaluates test performance from validation-selected top mixes or proportional perturbation trials
- `analyze_mixing_validation.ipynb`
  - analysis figures for validation and test outputs
- `run_lgn.py`
  - runs one LightGCN training job for a selected dataset using full-train split (`eval_split=test`) and writes metrics JSON
- `hyperparam/grid_search.py`
  - grid search over `lr` and `decay` for both `lgn` and `mf` with cross-validation averaging by Recall@k
- `hyperparam/submit_grid_search_sbatch.sh`
  - submits one sbatch job to run hyperparameter grid search for a dataset

## 1) Create K-Fold Splits

```bash
python mixing/data_preproc/movielens.py --num-folds 5
```

Optional:

```bash
python mixing/data_preproc/movielens.py \
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
python mixing/mix_sweep.py \
  --feature-name Age \
  --source-label 56 \
  --trials 1000 \
  --epochs 40 \
  --layer 1 \
  --num-folds 5
```

Default CSV output:

- `mixing/validation/<dataset>/<model>/mixing_results__feature-<feature>__source-<source>.csv`

Behavior:

- reuses existing split files and labels when present
- verifies max user/item index alignment across all split files
- evaluates baselines:
  - `no_augmentation` (`alpha_aug=0`)
  - `proportional` (`alpha_aug=1`, `alpha_g=n_g/n_aug`)
  - `stratified` (`alpha_aug=1`, uniform over augmentation groups)
- random trials use three samplers:
  - `dirichlet`: `alpha_mix ~ Dirichlet(dirichlet_alpha)` (default `0.7`)
  - `proportional_perturbation`: `x + scale * (delta - mean(delta))` where `x` is proportional mix and `delta ~ Uniform([0,1])`
  - `lognormal`: sample `alpha' ~ LogNormal(mu, sigma)` (default `mu=0, sigma=0.5`) and transform with group sizes
- sampler allocation:
  - trials are split across samplers as evenly as possible (about one-third each)
  - total sampled random mixes is asserted to equal `--trials`
- `alpha_aug` is always sampled as `Uniform(0.5, 2.0)` for random trials
- every sampled `alpha_mix` is validated (`sum=1`, no negative entries) and re-sampled on failure
- runs each alpha on every fold index (`cv_fold = 0..k-1`)
- appends one row per `(trial, fold)` with metrics and `training_time_seconds`

Smoke-test mode:

- samples all random alpha mixes but skips LightGCN training/evaluation
- writes a distribution figure (component densities, disaggregated by sampler) using notebook-style formatting

```bash
python mixing/mix_sweep.py \
  --feature-name Age \
  --source-label 56 \
  --trials 300 \
  --smoke-test \
  --smoke-plot-path mixing/validation/<dataset>/<model>/mix_sweep_smoke.png
```

SBATCH helper:

```bash
bash mixing/submit_mix_sweep_sbatch.sh
```

This script submits six jobs with:

- `--job-name="mix-<source_label>"`
- `--partition=dean`
- `--gres="gpu:nvidia_rtx_6000_ada_generation:1"`
- mix sweep command:
  - `python mixing/mix_sweep.py --feature-name Age --source-label <label> --trials 300 --epochs 40 --layer 1 --num-folds 5`

Validation CSV compatibility note:

- Validation CSV files ending with `single-val.csv` were generated before cross-validation was added.
- Those legacy files do not include `cv_fold` and `num_folds` columns in their headers.

Runtime note:

- In our current environment, an 8-hour run completes about 75 trials (with `num_folds=5`).

## 3) Evaluate Top Validation Mixes On Test

```bash
python mixing/evaluate_top_mixes_test.py \
  --feature-name Age \
  --source-label 56 \
  --top-k 200
```

Default input/output:

- reads validation CSVs from `mixing/validation/<dataset>/<model>`
- writes:
  - `mixing/test/<dataset>/<model>/test_eval__feature-<feature>__source-<source>.csv`

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
python mixing/evaluate_top_mixes_test.py \
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

## 4.1) Run Standalone LightGCN (Single Full-Train Run)

```bash
python mixing/run_lgn.py \
  --dataset ml-1m \
  --epochs 40
```

Notes:

- `--dataset` is a CLI argument (examples: `ml-1m`, `lastfm-asia`)
- script runs a single training job with `eval_split=test` (uses `train_full.txt` when available)
- default output JSON path is `mixing/<dataset>_lgn_metrics.json`
- override output path with `--output-json <path>`

## 5) Analyze Outputs

Notebook:

- `mixing/analyze_mixing_validation.ipynb`

Assumes:

- validation CSVs in `mixing/validation/<dataset>/<model>`
- test CSVs in `mixing/test/<dataset>/<model>`

Includes:

- real alpha-mix distributions
- simulated Dirichlet alpha-mix distributions (multiple concentration overlays)
- metric density panels
- top-vs-all distance distributions
- best alpha-mix by alpha-aug regime
- grouped-bar test summary (with % deltas vs proportional baseline)

## 6) Hyperparameter Grid Search

Run a cross-validation grid search over:

- `lr ∈ {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5}`
- `decay ∈ {1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5}`
- models: `lgn` and `mf`

For each `(model, lr, decay)`, the script runs all CV folds (`eval_split=val`, `val_split_idx=0..k-1`) and writes averaged Recall@k to CSV.  
`bpr_batch` is set to `1,000,000` by default as requested.
The selection metric is `recall@k` where `k` is the first value in `--topks`.

```bash
python mixing/hyperparam/grid_search.py \
  --dataset ml-1m \
  --epochs 40 \
  --layer 1 \
  --recdim 64 \
  --topks "[20]"
```

Outputs:

- CSV: `mixing/hyperparam/results/<dataset>.csv`
- PDF heatmaps (two panels: `lgn`, `mf`): `mixing/hyperparm/plots/<dataset>.pdf`

SBATCH helper:

```bash
bash mixing/hyperparam/submit_grid_search_sbatch.sh ml-1m
```
