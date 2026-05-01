# proportional vs stratified

This experiment compares `proportional` vs `stratified` group-mixing baselines as embedding dimension (`recdim`) changes.

## What `train.py` does

For one `(dataset, model, source_label, lr, decay)` configuration:

- Uses baseline trials: `proportional` and `stratified`
- Evaluates `recdim` values: `2, 4, 8, 16, 32, 64, 128, 256`
- Runs `5` trials per `(baseline, recdim)` with different seeds
- Evaluates on the test split via LightGCN ranking metrics
- Writes rows to:
  - `mixing/experiments/prop_vs_stratified/results/{dataset}__{model}__source-{source_label}.csv`

Defaults for feature + topks are dataset-aware:

- `lastfm-asia` -> feature `Country`, `topks=[100]`
- `movielens` or `ml-1m` -> feature `Age`, `topks=[20]`

## Run training

Local:

```bash
python mixing/experiments/prop_vs_stratified/train.py \
  --dataset ml-1m \
  --model lgn \
  --source-label 1 \
  --lr 0.001 \
  --decay 1e-4
```

Cluster:

```bash
bash mixing/experiments/prop_vs_stratified/submit_sbatch.sh lgn 0.001 1e-4 ml-1m 1
# with explicit trial count:
bash mixing/experiments/prop_vs_stratified/submit_sbatch.sh lgn 0.001 1e-4 ml-1m 1 5
```

## Plotting

`plot.py` loads CSVs for a specific `(dataset, model)` pair and creates a grid:

- subplot rows: source groups
- subplot columns: metric names
- x-axis: embedding dimension (`log2` scale)
- y-axis: mean metric value across random seeds
- one line per baseline (`proportional`, `stratified`)

Output path:

- `mixing/experiments/prop_vs_stratified/plots/{dataset}__{model}__prop_vs_stratified.png`

Example:

```bash
python mixing/experiments/prop_vs_stratified/plot.py --dataset ml-1m --model lgn
```
