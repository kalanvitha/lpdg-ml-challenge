import argparse
import pandas as pd
import numpy as np
import glob
import os
import datetime as dt
from sklearn.ensemble import IsolationForest


# =========================================================
# CONFIGURATION
# =========================================================

DATA_DIR = "data"

OUTPUT_FILE = "predictions.csv"

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]

WEEKS = pd.date_range(
    "2026-02-02",
    "2026-03-23",
    freq="7D"
)

VISITS_PER_WEEK = 15

BASELINE_DAYS = 28
RECENT_DAYS = 7

RANDOM_STATE = 42


# =========================================================
# COMMAND-LINE ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="LPDG V4 ML gateway ranking"
    )

    parser.add_argument(
        "--data",
        default=DATA_DIR,
        help="Path to challenge data directory"
    )

    return parser.parse_args()


# =========================================================
# LOAD TELEMETRY
# =========================================================

def load_telemetry(data_dir):

    telemetry_dir = os.path.join(
        data_dir,
        "telemetry"
    )

    files = glob.glob(
        os.path.join(
            telemetry_dir,
            "**",
            "*.parquet"
        ),
        recursive=True
    )

    frames = []

    for file in files:

        df = pd.read_parquet(file)

        required = [
            "gateway_id",
            "ts_utc",
            *METRICS
        ]

        df = df[required].copy()

        frames.append(df)

    frame = pd.concat(
        frames,
        ignore_index=True
    )

    frame["ts"] = pd.to_datetime(
        frame["ts_utc"],
        utc=True
    ).dt.tz_localize(None)

    frame["gateway_id"] = (
        frame["gateway_id"]
        .astype(str)
        .str.replace(
            ":",
            "",
            regex=False
        )
        .str.upper()
    )

    return frame


# =========================================================
# CALCULATE HISTORICAL GATEWAY STATISTICS
# =========================================================

def calculate_gateway_stats(baseline):

    stats = baseline.groupby(
        "gateway_id"
    )[METRICS].agg(
        [
            "mean",
            "std",
            "max",
            "median"
        ]
    )

    return stats


# =========================================================
# CREATE GATEWAY-RELATIVE FEATURES
# =========================================================

def create_relative_features(
    data,
    stats
):

    features = data[
        ["gateway_id"] + METRICS
    ].copy()

    for metric in METRICS:

        mean_col = (
            metric,
            "mean"
        )

        std_col = (
            metric,
            "std"
        )

        max_col = (
            metric,
            "max"
        )

        median_col = (
            metric,
            "median"
        )

        mean_values = (
            features["gateway_id"]
            .map(stats[mean_col])
        )

        std_values = (
            features["gateway_id"]
            .map(stats[std_col])
        )

        max_values = (
            features["gateway_id"]
            .map(stats[max_col])
        )

        median_values = (
            features["gateway_id"]
            .map(stats[median_col])
        )

        # -------------------------------------------------
        # Deviation from historical mean
        # -------------------------------------------------

        features[
            f"{metric}_deviation"
        ] = (
            features[metric]
            - mean_values
        )

        # -------------------------------------------------
        # Z-score
        # -------------------------------------------------

        features[
            f"{metric}_zscore"
        ] = (
            features[metric]
            - mean_values
        ) / std_values.replace(
            0,
            np.nan
        )

        # -------------------------------------------------
        # Ratio to historical mean
        # -------------------------------------------------

        features[
            f"{metric}_ratio"
        ] = (
            features[metric]
            / mean_values.replace(
                0,
                np.nan
            )
        )

        # -------------------------------------------------
        # Deviation from historical maximum
        # -------------------------------------------------

        features[
            f"{metric}_max_deviation"
        ] = (
            features[metric]
            - max_values
        )

        # -------------------------------------------------
        # Deviation from historical median
        # -------------------------------------------------

        features[
            f"{metric}_median_deviation"
        ] = (
            features[metric]
            - median_values
        )

    # -----------------------------------------------------
    # Remove gateway ID before ML
    # -----------------------------------------------------

    feature_columns = [
        column
        for column in features.columns
        if column != "gateway_id"
    ]

    X = features[
        feature_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return features, X


# =========================================================
# MAIN
# =========================================================

args = parse_args()

print("Loading telemetry...")

frame = load_telemetry(
    args.data
)

print(
    f"Loaded {len(frame)} telemetry rows"
)

print(
    f"Number of gateways: "
    f"{frame['gateway_id'].nunique()}"
)


predictions = []


# =========================================================
# PROCESS EACH WEEK
# =========================================================

for week in WEEKS:

    print(
        f"Processing week: "
        f"{week.date()}"
    )

    end = week.to_pydatetime()

    # -----------------------------------------------------
    # HISTORICAL TRAINING PERIOD
    #
    # 28 days before prediction week,
    # excluding the most recent 7 days.
    #
    # Example:
    #
    # Prediction week = Feb 2
    #
    # Training:
    # Jan 5 -> Jan 26
    #
    # Testing:
    # Jan 26 -> Feb 2
    # -----------------------------------------------------

    baseline_start = (
        end
        - dt.timedelta(
            days=BASELINE_DAYS
        )
    )

    baseline_end = (
        end
        - dt.timedelta(
            days=RECENT_DAYS
        )
    )

    baseline = frame[
        (frame["ts"] >= baseline_start)
        &
        (frame["ts"] < baseline_end)
    ].copy()

    # -----------------------------------------------------
    # FUTURE / RECENT 7-DAY PERIOD
    # -----------------------------------------------------

    recent = frame[
        (frame["ts"] >= baseline_end)
        &
        (frame["ts"] < end)
    ].copy()

    if baseline.empty or recent.empty:

        print(
            "Skipping week because "
            "training or recent data is empty."
        )

        continue

    # -----------------------------------------------------
    # LEARN NORMAL BEHAVIOUR FROM HISTORICAL DATA ONLY
    # -----------------------------------------------------

    stats = calculate_gateway_stats(
        baseline
    )

    # -----------------------------------------------------
    # CREATE FEATURES
    #
    # IMPORTANT:
    # Recent data uses the SAME historical
    # statistics. It does NOT calculate
    # new statistics from itself.
    # -----------------------------------------------------

    baseline_features, X_train = (
        create_relative_features(
            baseline,
            stats
        )
    )

    recent_features, X_recent = (
        create_relative_features(
            recent,
            stats
        )
    )

    # -----------------------------------------------------
    # TRAIN ISOLATION FOREST
    # -----------------------------------------------------

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(
        X_train
    )

    # -----------------------------------------------------
    # SCORE FUTURE 7-DAY DATA
    # -----------------------------------------------------

    recent_features[
        "anomaly_score"
    ] = -model.score_samples(
        X_recent
    )

    # -----------------------------------------------------
    # GATEWAY-LEVEL AGGREGATION
    # -----------------------------------------------------

    ranked = (
        recent_features
        .groupby("gateway_id")
        .agg(
            anomaly_score=(
                "anomaly_score",
                "mean"
            ),
            max_anomaly_score=(
                "anomaly_score",
                "max"
            )
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # RANK GATEWAYS
    # -----------------------------------------------------

    ranked = ranked.sort_values(
        [
            "anomaly_score",
            "max_anomaly_score"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )

    top = ranked.head(
        VISITS_PER_WEEK
    )

    # -----------------------------------------------------
    # CREATE SUBMISSION ROWS
    # -----------------------------------------------------

    for rank, (_, row) in enumerate(
        top.iterrows(),
        start=1
    ):

        predictions.append(
            {
                "week_start":
                    week.strftime(
                        "%Y-%m-%d"
                    ),

                "rank":
                    rank,

                "gateway_id":
                    row["gateway_id"],

                "score":
                    row["anomaly_score"],

                "reason":
                    "Gateway-relative "
                    "Isolation Forest anomaly"
            }
        )


# =========================================================
# SAVE RESULTS
# =========================================================

predictions_df = pd.DataFrame(
    predictions
)

predictions_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()

print(
    f"Wrote {OUTPUT_FILE}"
)

print(
    f"Rows: {len(predictions_df)}"
)

print(
    f"Weeks: "
    f"{predictions_df['week_start'].nunique()}"
)

print()

print(
    predictions_df.head(15).to_string(
        index=False
    )
)