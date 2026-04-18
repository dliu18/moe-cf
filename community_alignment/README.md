# Community Alignment Scripts

This folder contains three Python scripts to support data cleaning and data-mixing experiments on the Community Alignment dataset.

## 1) Prepare dataset splits

This script creates the preprocessed dataset and the 70/10/20 prompt-group splits.

```bash
python community_alignment/prepare_community_alignment_splits.py \
  --input_csv data/community_alignment.csv \
  --output_dir data
```

Outputs:

- `data/community_alignment_en.csv`
- `data/community_alignment_en_train.csv`
- `data/community_alignment_en_val.csv`
- `data/community_alignment_en_test.csv`

Actual Output: 

Input rows: 90,256
English rows: 30,856
Pregenerated English rows: 22,168
Unique first_turn_prompt groups: 1,033
Split prompt counts: train=723, val=103, test=207
Saved filtered dataset to: data/community_alignment_en.csv
Saved train rows to: data/community_alignment_en_train.csv (15,196)
Saved val rows to: data/community_alignment_en_val.csv (2,435)
Saved test rows to: data/community_alignment_en_test.csv (4,537)

## 2) Run mixing experiment scaffold

```bash
python community_alignment/mixing_experiment.py \
  --protected_attribute annotator_country \
  --source_group india \
  --augmentation_group united\ states \
  --training_prompts 100 \
  --test_prompts 250 \
  --test_set both \
  --num_trials 1 \
  --alpha_step 0.1 \
  --random_seed 0 \
  --output_dir results/community_alignment_mixing \
  --simulate \
  --debug
```

`--alpha` runs a single alpha value and skips the sweep.
If `--alpha` is omitted, alpha is swept using `--alpha_step`.
When sweeping (when `--alpha` is omitted), the script also runs a no-example baseline for `num_trials` trials, where the prompt contains only the instruction and test prompts.
The plot includes the average baseline accuracy as a dashed horizontal line labeled `No-example baseline (mean)`.
If a live API call returns a `finish_reason` other than `"stop"` (for example, `"length"`), that trial is marked unsuccessful and its accuracy is excluded from accuracy reporting (stored as `NaN` in CSV) and from the plotted points.
In this mode, accuracy is computed at the **annotator level** (every source-group test row is one target), not collapsed per prompt.
`--test_prompts` is interpreted as a number of prompts, sampled uniformly at random from valid source-group prompts (each selected prompt is included **once** in the API call, and expanded to all source-group annotators for accuracy).
At the end of execution, the script prints success rate by alpha.

Example with a single alpha:

```bash
python community_alignment/mixing_experiment.py \
  --protected_attribute annotator_country \
  --source_group india \
  --augmentation_group "united states" \
  --training_prompts 120 \
  --test_prompts 40 \
  --test_set both \
  --num_trials 3 \
  --alpha 0.2 \
  --random_seed 0 \
  --simulate
```

By default, the script performs **real API calls** for scoring unless `--simulate` is set.
`--model_name` defaults to `gpt-5-mini` and can be changed at runtime.
If you omit `--output_csv`, `--plot_path`, and `--debug_log_path`, outputs default into `--output_dir`.
The API input estimation used for reporting uses a local token approximation:
- estimated input tokens are computed from the request prompt text,
- and API-reported input tokens are captured from completion usage.

### Simulation mode (recommended for smoke checks)

Use `--simulate` to run end-to-end with fake predictions so you still get real CSV/plot outputs and accuracy columns:

```bash
python community_alignment/mixing_experiment.py \
  --protected_attribute annotator_gender \
  --source_group female \
  --augmentation_group male \
  --training_prompts 120 \
  --test_prompts 40 \
  --test_set both \
  --num_trials 3 \
  --alpha_step 0.1 \
  --simulate \
  --simulate_prediction_mode random \
  --random_seed 0
```

If you want to inspect the generated prompts, use `--debug` and inspect the log file in `--output_dir` (or your explicit `--debug_log_path`).

The debug log stores each entry as:
- a compact JSON object with:
  - `api_call`: exact OpenAI request payload (`model`, `messages`, `temperature`, `max_completion_tokens` for GPT-5-nano; `max_tokens` for older models)
  - `meta`: call metadata (alpha/trial/budget, token counts, etc.)

Each trial also records sequence-length validity:
- `predicted_response_count`: number of parsed letters returned by the model,
- `expected_response_count`: number of source-group annotator rows in the call,
- `is_sequence_length_valid`: whether those counts match.
The script prints both success rate and sequence-length validity rate by alpha at the end.

### Per-call metrics logging

Use `--log_call_metrics` to print one block per call with:

- exact API response text
- finish_reason and refusal (if returned)
- completion token usage (including model-internal reasoning tokens reported by usage)
- accuracy for that call
- estimated input token count
- actual input token count from API usage

```bash
python community_alignment/mixing_experiment.py \
  --protected_attribute annotator_country \
  --source_group india \
  --augmentation_group "united states" \
  --training_prompts 120 \
  --test_prompts 40 \
  --test_set both \
  --num_trials 1 \
  --alpha_step 0.1 \
  --log_call_metrics
```

For real API runs (required for API response output), omit `--simulate`:

```bash
python community_alignment/mixing_experiment.py \
  --protected_attribute annotator_ethnicity \
  --source_group Indo-Aryan \
  --augmentation_group White \
  --training_prompts 25 \
  --test_prompts 20 \
  --test_set both \
  --num_trials 10 \
  --alpha_step 0.2 \
  --output_dir results/community_alignment_mixing \
  --log_call_metrics 
```

Notes:

- For real API runs, ensure `OPENAI_API_KEY` is set in your environment.
- For local end-to-end validation without API calls, use `--simulate`.

## 3) Run polarizing test-prompt mixing experiment

This variant uses a fixed shared set of test prompts selected by polarization between source and augmentation groups.

How it works:

- Build the test pool from `--test_set` (`val`, `test`, or `both`).
- For each prompt in that pool, compute normalized preference vectors for source and augmentation groups from their `first_turn_preferred_response`.
Compute cosine similarity between the two vectors.
Compute a weighted polarizing score:
  - `weighted_score = (1.0 - cosine_similarity) * (source_rows * augmentation_rows)`
- Sort prompts by this score in descending order and keep the top-`k` most polarizing (`--test_prompts`, or all if `-1`).
- Use that same fixed test set for all alphas and all trials.
- Exclude prompts from ranking that lack either group rows.

```bash
python community_alignment/mixing_experiment_polarizing.py \
  --protected_attribute annotator_ethnicity \
  --source_group Indo-Aryan \
  --augmentation_group White \
  --training_prompts 25 \
  --test_prompts 20 \
  --test_set both \
  --num_trials 10 \
  --alpha_step 0.2 \
  --output_dir results/community_alignment_mixing_polarizing \
  --random_seed 0 \
  --log_call_metrics
```

Single-alpha mode:

```bash
python community_alignment/mixing_experiment_polarizing.py \
  --protected_attribute annotator_country \
  --source_group india \
  --augmentation_group "united states" \
  --training_prompts 120 \
  --test_prompts 25 \
  --test_set both \
  --num_trials 3 \
  --alpha 0.2 \
  --simulate
```

Outputs:

- `community_alignment_polarizing_results.csv` (trial-level CSV)
- `community_alignment_polarizing_scatter.png` (accuracy scatter + alpha mean curve + optional baseline line)
- `community_alignment_polarizing_report.html` (per-prompt true/predicted preference-vector charts)

In this polarizing variant, trial accuracy is also computed at the **annotator level**:
denominator = total number of source-group annotator rows across the fixed top-k prompts,
numerator = how many of those labels are matched by the model's per-prompt prediction.

When sweeping alphas (`--alpha` omitted), this script also runs a no-example baseline for `num_trials`.
