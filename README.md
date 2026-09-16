# LPDG Innovation Hub Selection Challenge 2026 — Part 2: Machine Learning

## Overview

This project is my solution for Part 2 of the LPDG Innovation Hub Selection Challenge 2026.

I selected **Machine Learning (Area E)**.

The goal is to rank 15 gateways for each challenge week so that field engineers can prioritize gateways that are most likely to need attention. The model is evaluated on cost, not simply prediction accuracy.

The final solution uses a gateway-relative, temporally separated Isolation Forest model and is designed to run using challenge data supplied separately from the repository.

## Challenge Output

The final model produces:

`predictions.csv`

with exactly 120 rows:

* 15 gateways per week
* 8 weeks
* Challenge weeks: 2026-02-02 through 2026-03-23

The required columns are:

* `week_start`
* `rank`
* `gateway_id`
* `score`
* `reason`

The official submission validator confirms the final file is valid.

## Project Files

`run.bat`
Single-command entry point that runs the final ML pipeline and validates the generated `predictions.csv`.

`baseline_3sigma.py`
Official 3-Sigma baseline supplied for comparison.

`baseline_3sigma_v4.py`
Final Machine Learning solution.

`compare_baseline_vs_ml.py`
Compares the baseline and final ML model using the available engineer-review data as proxy evidence.

`validate_submission.py`
Checks that `predictions.csv` satisfies the challenge output requirements.

`validate_unseen_gateways.py`
Tests the model on gateways that were not included in the training gateway set.

`validate_future_weeks.py`
Tests the model on weeks occurring after the training period.

`validate_network_change.py`
Controlled stress test for behavior when network conditions change.

`predictions.csv`
Final ML predictions.

`predictions_baseline.csv`
Predictions produced by the official 3-Sigma baseline.

`unseen_gateway_validation.csv`
Output from unseen-gateway validation.

`future_week_validation.csv`
Output from future-week validation.

`network_change_validation.csv`
Output from the controlled network-change stress test.

Challenge data is intentionally not included in this repository.

## Approach

### 1. Input Data

The model uses telemetry data from the challenge data directory.

The final model focuses on three reliability-related telemetry metrics:

* `offline_duration_sec`
* `disconnection_cnt`
* `reboot_cnt`

These were selected because they directly represent gateway reliability problems and are also used by the supplied 3-Sigma baseline.

I intentionally avoided using field-visit outcomes or engineer-review labels as training labels because those are not the official hidden challenge ground truth.

### 2. Temporal Separation

For every prediction week, the model separates historical data from the period being scored.

The final model uses:

* 28-day lookback window
* the most recent 7 days reserved for scoring
* the preceding 21 days used for training
* the recent 7-day period is then scored and ranked

Conceptually:

```text
28-day lookback window
<------------------------------->

|       21 days       |  7 days  |
|       training      |  scoring |
|                     | (recent) |
```

This prevents the model from using the same recent observations to define its own baseline and score them.

### 3. Gateway-Relative Features

Different gateways can have different normal operating behavior.

Therefore, instead of relying only on raw telemetry values, the model calculates gateway-specific historical statistics from the training period.

For each of the three metrics, the model creates features based on:

* raw value
* deviation from historical mean
* z-score
* ratio to historical mean
* deviation from historical maximum
* deviation from historical median

This allows the model to identify behavior that is unusual for that gateway, rather than assuming every gateway has the same normal range.

### 4. Machine Learning Model

The final model uses:

**Isolation Forest**

Configuration:

```text
n_estimators = 300
contamination = "auto"
random_state = 42
n_jobs = -1
```

Isolation Forest is suitable here because official hidden labels are not available for training.

The model is trained on the transformed historical observations and then used to score the recent period.

A higher anomaly score indicates behavior that is more unusual compared with the gateway's historical operating pattern.

### 5. Ranking

For each gateway in the recent seven-day period, the model calculates:

* mean anomaly score
* maximum anomaly score

Gateways are ranked primarily by mean anomaly score, with maximum anomaly score used as a tie-breaker.

The top 15 gateways are selected for each week.

The output reason is:

`Gateway-relative Isolation Forest anomaly`

## Baseline vs Final ML

The supplied 3-Sigma baseline and the final ML model were compared using the available February engineer-review information.

**Important:** engineer review is only proxy evidence. It is not the official hidden challenge ground truth or official challenge score.

### Current proxy comparison

| Model            | Reviewed predictions | Known Schlecht | Known Normal | Proxy precision | Proxy wasted visits | Proxy visit cost |
| ---------------- | -------------------: | -------------: | -----------: | --------------: | ------------------: | ---------------: |
| 3-Sigma Baseline |                   48 |             22 |           26 |          45.83% |                  26 |            €9880 |
| Final ML V4      |                   89 |             85 |            4 |          95.51% |                   4 |            €1520 |

### Proxy cost improvement

```text
Baseline proxy cost : €9880
Final ML proxy cost : €1520
Cost reduction      : €8360
Percentage reduction: 84.62%
```

These numbers are included as supporting evidence only and must not be interpreted as the official hidden challenge score.

## Validation

### Unseen Gateways

The model was tested on gateways that were not included in the training gateway set.

Result:

```text
Training gateways    : 256
Unseen test gateways : 64

Reviewed selections : 59
Known Schlecht      : 46
Known Normal        : 13
Proxy precision     : 77.97%
Proxy wasted visits : 13
Proxy visit cost    : €4940
```

This is proxy evidence based on the available engineer-review data.

### Future Weeks

The model was also tested by training on an earlier period and evaluating on a later week.

Result:

```text
Reviewed selections : 93
Known Schlecht      : 91
Known Normal        : 2
Proxy precision     : 97.85%
Proxy wasted visits : 2
Proxy visit cost    : €760
```

This tests whether the model can score observations that occur after its training period.

Again, these are proxy results and not official hidden challenge scores.

### Network Change Stress Test

A controlled stress test was used to examine model behavior when network conditions change.

Result:

```text
Normal average anomaly score : 0.364645
Changed average anomaly score: 0.374168

Top-15 overlap: 13 / 15

Gateways affected                 : 61
Changed gateways in top 15        : 5
Percentage of top 15 affected     : 33.33%
Average rank improvement affected : 11.33
```

This is a controlled stress test, not official challenge scoring.

The model responds to changed network behavior by increasing anomaly scores for affected gateways, while retaining substantial overlap with the original top-15 ranking.

## Training and Prediction Separation

Training and prediction are separate operations inside the weekly process.

The model:

1. Builds historical gateway statistics.
2. Creates training features from historical data.
3. Fits Isolation Forest on historical observations.
4. Creates features for the recent period using the already-computed historical statistics.
5. Scores the recent observations.
6. Ranks gateways and selects the top 15.

The model does not retrain on the observations it is currently using for the final ranking.

## Running the Project

### Requirements

Python 3.x with the packages listed in:

`requirements.txt`

### Option 1 — Single-Command Run

The project provides `run.bat` as a single-command entry point for running the final ML pipeline and validating the generated submission.

From PowerShell, run:

```powershell
.\run.bat
```

The script:

1. Checks for challenge data in the project's `data\telemetry` directory.
2. If the data is not found there, asks for the full path to the challenge data directory.
3. Runs the final ML model.
4. Generates `predictions.csv`.
5. Runs the submission validator.
6. Reports whether the final predictions pass validation.

If the challenge data is available under:

```text
data\
└── telemetry\
```

the script uses it automatically.

The challenge data itself is not included in the repository.

### Option 2 — Manual Execution

The individual Python commands can also be used to run and validate each component separately.

#### Run the Final ML Model

From the project directory:

```powershell
python baseline_3sigma_v4.py --data "C:\path\to\challenge\data"
```

The model writes:

`predictions.csv`

to the project directory.

If the program is configured to use the project's default data directory, the `--data` argument can be omitted.

#### Run the Official Submission Check

```powershell
python validate_submission.py predictions.csv
```

Expected result:

```text
predictions.csv: OK
15 ranked gateways for each of 8 weeks
```

#### Run the Baseline

```powershell
python baseline_3sigma.py --data "C:\path\to\challenge\data" --out predictions_baseline.csv
```

#### Compare Baseline and ML

```powershell
python compare_baseline_vs_ml.py --data "C:\path\to\challenge\data"
```

### Run Validation Tests

#### Unseen gateways

```powershell
python validate_unseen_gateways.py --data "C:\path\to\challenge\data"
```

#### Future weeks

```powershell
python validate_future_weeks.py --data "C:\path\to\challenge\data"
```

#### Network change

```powershell
python validate_network_change.py --data "C:\path\to\challenge\data"
```

## Data and Privacy

The challenge dataset is not included in this repository.

The code expects the data to be supplied separately through the `--data` argument or through the project's `data\` directory.

No challenge data, private credentials, API keys, or secrets should be committed to the repository.

## Limitations / What It Cannot Do

This solution has several limitations:

* It is not trained on the official hidden challenge labels. The engineer-review information available during development is only proxy evidence.
* It uses only three telemetry metrics. Other telemetry signals may contain additional information about gateway failures.
* Isolation Forest identifies unusual behavior; it does not understand the physical cause of a failure. A high score means anomalous behavior, not a guaranteed diagnosis.
* The gateway-relative statistics depend on having enough historical observations. Very new gateways or gateways with sparse telemetry may be harder to score reliably.
* The network-change experiment is controlled. It does not reproduce every possible real-world network change.
* The solution does not automatically retrain while answering a live query. Training and prediction are intentionally separated so prediction can remain fast and deterministic.

## What Another Two Weeks of Work Could Improve

With two additional weeks, I would:

* test additional telemetry features without introducing leakage;
* perform stronger time-based model selection;
* evaluate multiple anomaly-detection models;
* investigate gateway groups with consistently different operating patterns;
* improve robustness to missing or sparse telemetry;
* test more realistic network-change scenarios;
* improve explanation of why individual gateways receive high scores.

## Reproducibility

The final solution uses fixed model settings and:

```text
random_state = 42
```

The dependency versions are recorded in `requirements.txt`.

The same challenge data and execution method should therefore produce reproducible results under the specified environment.

## AI Usage

AI assistance was used during development for:

* understanding the challenge requirements;
* brainstorming model and feature-engineering approaches;
* debugging Python and package/environment issues;
* interpreting validation results;
* improving project documentation.

All code was run and checked locally, and implementation decisions were reviewed before being used.

More details, including one concrete AI mistake that was identified and corrected, are documented in:

`AI-USAGE.md`

## Decision Log

The main design choices and alternatives considered are documented in:

`DECISIONS.md`

## Demo Video

A short screen-recording demonstrating the project execution, ML-based gateway ranking,
validation results, and repository contents is available here:

[Watch the LPDG Demo Video](https://drive.google.com/file/d/1ELN2wtmZ43Kp4gzF7ff0opvDkFq8f5yr/view?usp=sharing)