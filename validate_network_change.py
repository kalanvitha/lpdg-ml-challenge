import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

TEST_WEEK = pd.Timestamp(
    "2026-03-23",
    tz="UTC"
)

HISTORICAL_DAYS = 28
RECENT_DAYS = 7

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt"
]


# ============================================================
# LOAD TELEMETRY
# ============================================================

def load_telemetry(data_dir):

    telemetry_dir = (
        Path(data_dir)
        / "telemetry"
    )

    months = [
        "2025-08",
        "2025-09",
        "2025-10",
        "2025-11",
        "2025-12",
        "2026-01",
        "2026-02",
        "2026-03"
    ]

    frames = []

    for month in months:

        path = (
            telemetry_dir
            / f"month={month}"
            / "part-0.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Telemetry file not found: {path}"
            )

        frames.append(
            pd.read_parquet(path)
        )

    df = pd.concat(
        frames,
        ignore_index=True
    )

    df["ts_utc"] = pd.to_datetime(
        df["ts_utc"],
        utc=True
    )

    return df


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Validate ML model under network changes"
)

parser.add_argument(
    "--data",
    default=str(DEFAULT_DATA_DIR),
    help="Path to challenge data directory"
)

args = parser.parse_args()

DATA_DIR = Path(args.data)


# ============================================================
# LOAD TELEMETRY
# ============================================================

print("Loading telemetry...")

df = load_telemetry(
    DATA_DIR
)

print(
    "Loaded rows:",
    len(df)
)

print(
    "Gateways:",
    df["gateway_id"].nunique()
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(train_df, score_df):

    stats = (
        train_df
        .groupby("gateway_id")[METRICS]
        .agg([
            "mean",
            "std",
            "max",
            "median"
        ])
    )

    stats.columns = [
        f"{metric}_{stat}"
        for metric, stat in stats.columns
    ]

    stats = stats.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for metric in METRICS:

        std_col = f"{metric}_std"

        stats[std_col] = (
            stats[std_col]
            .fillna(0)
        )

        stats.loc[
            stats[std_col] == 0,
            std_col
        ] = 1e-6

    result = score_df[
        ["gateway_id"] + METRICS
    ].copy()

    result = result.merge(
        stats,
        left_on="gateway_id",
        right_index=True,
        how="inner"
    )

    feature_columns = []

    for metric in METRICS:

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        max_col = f"{metric}_max"
        median_col = f"{metric}_median"

        # Raw value
        raw_col = f"{metric}_raw"

        result[raw_col] = result[metric]

        feature_columns.append(
            raw_col
        )

        # Deviation from historical mean
        deviation_col = (
            f"{metric}_deviation"
        )

        result[deviation_col] = (
            result[metric]
            - result[mean_col]
        )

        feature_columns.append(
            deviation_col
        )

        # Z-score
        z_col = f"{metric}_zscore"

        result[z_col] = (
            result[metric]
            - result[mean_col]
        ) / result[std_col]

        feature_columns.append(
            z_col
        )

        # Ratio
        ratio_col = f"{metric}_ratio"

        denominator = (
            result[mean_col]
            .replace(0, 1e-6)
        )

        result[ratio_col] = (
            result[metric]
            / denominator
        )

        feature_columns.append(
            ratio_col
        )

        # Difference from historical maximum
        max_dev_col = (
            f"{metric}_max_deviation"
        )

        result[max_dev_col] = (
            result[metric]
            - result[max_col]
        )

        feature_columns.append(
            max_dev_col
        )

        # Difference from historical median
        median_dev_col = (
            f"{metric}_median_deviation"
        )

        result[median_dev_col] = (
            result[metric]
            - result[median_col]
        )

        feature_columns.append(
            median_dev_col
        )

    X = result[
        feature_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    return result, X


# ============================================================
# PERIODS
# ============================================================

train_start = (
    TEST_WEEK
    - pd.Timedelta(
        days=HISTORICAL_DAYS
    )
)

train_end = TEST_WEEK

test_start = TEST_WEEK

test_end = (
    TEST_WEEK
    + pd.Timedelta(
        days=RECENT_DAYS
    )
)


print()
print("=" * 80)
print("NETWORK CHANGE VALIDATION")
print("=" * 80)

print(
    "Training period:",
    train_start,
    "to",
    train_end
)

print(
    "Test period    :",
    test_start,
    "to",
    test_end
)


# ============================================================
# TRAIN / TEST DATA
# ============================================================

train_df = df[
    (df["ts_utc"] >= train_start) &
    (df["ts_utc"] < train_end)
].copy()

test_df = df[
    (df["ts_utc"] >= test_start) &
    (df["ts_utc"] < test_end)
].copy()

print()
print(
    "Training rows:",
    len(train_df)
)

print(
    "Test rows    :",
    len(test_df)
)


# ============================================================
# TRAIN MODEL
# ============================================================

print()
print(
    "Preparing historical training features..."
)

_, X_train = prepare_features(
    train_df,
    train_df
)

print(
    "Training Isolation Forest..."
)

model = IsolationForest(
    n_estimators=300,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train
)

print(
    "Model trained."
)


# ============================================================
# SCORE FUNCTION
# ============================================================

def score_data(
    score_df,
    scenario
):

    prepared, X = prepare_features(
        train_df,
        score_df
    )

    prepared["anomaly_score"] = (
        -model.score_samples(X)
    )

    scores = (
        prepared
        .groupby("gateway_id")
        .agg(
            mean_score=(
                "anomaly_score",
                "mean"
            ),
            max_score=(
                "anomaly_score",
                "max"
            )
        )
        .reset_index()
    )

    scores = scores.sort_values(
        [
            "mean_score",
            "max_score"
        ],
        ascending=False
    )

    scores["rank"] = range(
        1,
        len(scores) + 1
    )

    scores["scenario"] = scenario

    return scores


# ============================================================
# NORMAL NETWORK
# ============================================================

print()
print(
    "Scoring NORMAL network..."
)

normal_scores = score_data(
    test_df,
    "normal"
)

normal_top15 = (
    normal_scores
    .head(15)
    .copy()
)


# ============================================================
# GLOBAL NETWORK CHANGE
# ============================================================

print()
print(
    "Applying simulated GLOBAL network change..."
)

global_shift = test_df.copy()

# Convert BEFORE modifying
for metric in METRICS:

    global_shift[metric] = (
        global_shift[metric]
        .astype(float)
    )

global_shift["offline_duration_sec"] *= 1.50
global_shift["disconnection_cnt"] *= 1.50
global_shift["reboot_cnt"] *= 1.50

print(
    "Scoring GLOBAL network-change scenario..."
)

global_scores = score_data(
    global_shift,
    "global_change"
)

global_top15 = (
    global_scores
    .head(15)
    .copy()
)


# ============================================================
# LOCALIZED NETWORK CHANGE
# ============================================================

print()
print(
    "Applying simulated LOCALIZED network change..."
)

local_shift = test_df.copy()

# Convert the whole columns to float first.
for metric in METRICS:

    local_shift[metric] = (
        local_shift[metric]
        .astype(float)
    )


# Select gateways
all_gateways = sorted(
    local_shift["gateway_id"]
    .unique()
)

rng = np.random.RandomState(
    42
)

num_changed = max(
    1,
    int(
        len(all_gateways)
        * 0.20
    )
)

changed_gateways = set(
    rng.choice(
        all_gateways,
        size=num_changed,
        replace=False
    )
)

print(
    "Changed gateways:",
    len(changed_gateways)
)


mask = local_shift[
    "gateway_id"
].isin(
    changed_gateways
)


# Modify affected gateways
local_shift.loc[
    mask,
    "offline_duration_sec"
] = (
    local_shift.loc[
        mask,
        "offline_duration_sec"
    ] * 1.50
)

local_shift.loc[
    mask,
    "disconnection_cnt"
] = (
    local_shift.loc[
        mask,
        "disconnection_cnt"
    ] * 1.50
)

local_shift.loc[
    mask,
    "reboot_cnt"
] = (
    local_shift.loc[
        mask,
        "reboot_cnt"
    ] * 1.50
)


print(
    "Scoring LOCALIZED network-change scenario..."
)

local_scores = score_data(
    local_shift,
    "localized_change"
)

local_top15 = (
    local_scores
    .head(15)
    .copy()
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 80)
print("NETWORK CHANGE RESULTS")
print("=" * 80)


# ============================================================
# GLOBAL CHANGE
# ============================================================

normal_mean = (
    normal_scores["mean_score"]
    .mean()
)

global_mean = (
    global_scores["mean_score"]
    .mean()
)

normal_std = (
    normal_scores["mean_score"]
    .std()
)

global_std = (
    global_scores["mean_score"]
    .std()
)


print()
print(
    "GLOBAL NETWORK CHANGE"
)

print(
    "-" * 80
)

print(
    f"Normal average anomaly score : "
    f"{normal_mean:.6f}"
)

print(
    f"Changed average anomaly score: "
    f"{global_mean:.6f}"
)

print(
    f"Normal score std             : "
    f"{normal_std:.6f}"
)

print(
    f"Changed score std            : "
    f"{global_std:.6f}"
)


global_overlap = len(
    set(normal_top15["gateway_id"])
    &
    set(global_top15["gateway_id"])
)

print(
    "Top-15 overlap:",
    global_overlap,
    "/ 15"
)


# ============================================================
# LOCALIZED CHANGE
# ============================================================

local_top15_changed = local_top15[
    local_top15["gateway_id"]
    .isin(changed_gateways)
]


print()
print(
    "LOCALIZED NETWORK CHANGE"
)

print(
    "-" * 80
)

print(
    "Gateways affected:",
    len(changed_gateways)
)

print(
    "Changed gateways in top 15:",
    len(local_top15_changed)
)

print(
    "Percentage of top 15 affected:",
    round(
        len(local_top15_changed)
        / 15
        * 100,
        2
    ),
    "%"
)


# ============================================================
# RANK MOVEMENT
# ============================================================

normal_rank = normal_scores[
    [
        "gateway_id",
        "rank"
    ]
].rename(
    columns={
        "rank": "normal_rank"
    }
)

local_rank = local_scores[
    [
        "gateway_id",
        "rank"
    ]
].rename(
    columns={
        "rank": "changed_rank"
    }
)

rank_comparison = normal_rank.merge(
    local_rank,
    on="gateway_id",
    how="inner"
)

rank_comparison[
    "rank_improvement"
] = (
    rank_comparison["normal_rank"]
    -
    rank_comparison["changed_rank"]
)

affected_rank = rank_comparison[
    rank_comparison["gateway_id"]
    .isin(changed_gateways)
]


print()
print(
    "Average rank improvement for "
    "affected gateways:",
    round(
        affected_rank[
            "rank_improvement"
        ].mean(),
        2
    )
)


# ============================================================
# SAVE
# ============================================================

normal_top15[
    "scenario_rank"
] = range(
    1,
    16
)

global_top15[
    "scenario_rank"
] = range(
    1,
    16
)

local_top15[
    "scenario_rank"
] = range(
    1,
    16
)


results = pd.concat(
    [
        normal_top15,
        global_top15,
        local_top15
    ],
    ignore_index=True
)


# Save validation result in project folder,
# NOT inside the challenge data folder.
output_file = (
    Path(__file__).resolve().parent
    / "network_change_validation.csv"
)

results.to_csv(
    output_file,
    index=False
)


print()
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

print(
    f"Saved: {output_file.name}"
)

print()
print(
    "IMPORTANT: This is a controlled "
    "stress test."
)

print(
    "It simulates network changes; "
    "it is not official hidden challenge scoring."
)