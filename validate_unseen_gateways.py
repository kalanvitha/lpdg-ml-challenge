import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt"
]

HISTORICAL_DAYS = 28
RECENT_DAYS = 7

# 20% of gateways will be treated as unseen gateways
HOLDOUT_FRACTION = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD TELEMETRY
# ============================================================

def load_telemetry(data_dir):

    telemetry_dir = Path(data_dir) / "telemetry"

    files = sorted(
        telemetry_dir.glob("month=*/part-0.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No telemetry parquet files found in: "
            f"{telemetry_dir}"
        )

    print("Loading telemetry...")

    frames = []

    for file in files:

        df = pd.read_parquet(file)

        required = [
            "gateway_id",
            "ts_utc"
        ] + METRICS

        missing = [
            c for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {file}: {missing}"
            )

        frames.append(
            df[required]
        )

    telemetry = pd.concat(
        frames,
        ignore_index=True
    )

    # IMPORTANT:
    # Keep timestamps timezone-aware in UTC.
    telemetry["ts_utc"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True
    )

    telemetry["gateway_id"] = (
        telemetry["gateway_id"]
        .astype(str)
        .str.strip()
    )

    print(
        f"Loaded {len(telemetry)} telemetry rows"
    )

    print(
        f"Number of gateways: "
        f"{telemetry['gateway_id'].nunique()}"
    )

    return telemetry


# ============================================================
# NORMALIZE GATEWAY IDS
# ============================================================

def normalize_gateway_id(value):

    return (
        str(value)
        .strip()
        .replace(":", "")
        .upper()
    )


# ============================================================
# CREATE UNSEEN GATEWAY HOLDOUT
# ============================================================

def create_holdout_gateways(gateways):

    gateways = sorted(gateways)

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    shuffled = np.array(
        gateways,
        dtype=object
    )

    rng.shuffle(shuffled)

    n_holdout = max(
        1,
        int(
            len(shuffled)
            * HOLDOUT_FRACTION
        )
    )

    unseen_gateways = set(
        shuffled[:n_holdout]
    )

    training_gateways = set(
        shuffled[n_holdout:]
    )

    return (
        training_gateways,
        unseen_gateways
    )


# ============================================================
# BUILD GATEWAY-RELATIVE FEATURES
# ============================================================

def build_features(
    historical,
    scoring
):

    # --------------------------------------------------------
    # Calculate historical statistics
    # --------------------------------------------------------

    stats = (
        historical
        .groupby("gateway_id")[METRICS]
        .agg([
            "mean",
            "std",
            "max",
            "median"
        ])
    )

    # Flatten multi-level columns
    stats.columns = [
        f"{metric}_{stat}"
        for metric, stat in stats.columns
    ]

    stats = stats.reset_index()

    # --------------------------------------------------------
    # Protect against zero std / zero mean
    # --------------------------------------------------------

    for metric in METRICS:

        std_col = f"{metric}_std"
        mean_col = f"{metric}_mean"

        stats[std_col] = (
            stats[std_col]
            .replace(0, np.nan)
            .fillna(1.0)
        )

        stats[mean_col] = (
            stats[mean_col]
            .replace(0, np.nan)
            .fillna(1e-6)
        )

    # --------------------------------------------------------
    # Merge historical gateway statistics
    # --------------------------------------------------------

    data = scoring.merge(
        stats,
        on="gateway_id",
        how="left"
    )

    feature_columns = []

    # --------------------------------------------------------
    # Create gateway-relative features
    # --------------------------------------------------------

    for metric in METRICS:

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        max_col = f"{metric}_max"
        median_col = f"{metric}_median"

        # Raw metric
        feature_columns.append(
            metric
        )

        # Difference from historical mean
        deviation_col = (
            f"{metric}_deviation"
        )

        data[deviation_col] = (
            data[metric]
            - data[mean_col]
        )

        feature_columns.append(
            deviation_col
        )

        # Z-score
        zscore_col = (
            f"{metric}_zscore"
        )

        data[zscore_col] = (
            (
                data[metric]
                - data[mean_col]
            )
            / data[std_col]
        )

        feature_columns.append(
            zscore_col
        )

        # Ratio to historical mean
        ratio_col = (
            f"{metric}_ratio"
        )

        data[ratio_col] = (
            data[metric]
            / data[mean_col]
        )

        feature_columns.append(
            ratio_col
        )

        # Difference from historical maximum
        max_deviation_col = (
            f"{metric}_max_deviation"
        )

        data[max_deviation_col] = (
            data[metric]
            - data[max_col]
        )

        feature_columns.append(
            max_deviation_col
        )

        # Difference from historical median
        median_deviation_col = (
            f"{metric}_median_deviation"
        )

        data[median_deviation_col] = (
            data[metric]
            - data[median_col]
        )

        feature_columns.append(
            median_deviation_col
        )

    # --------------------------------------------------------
    # Clean features
    # --------------------------------------------------------

    X = data[
        feature_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    return X, data


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # COMMAND-LINE ARGUMENTS
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Validate ML model on unseen gateways"
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_DIR),
        help="Path to challenge data directory"
    )

    args = parser.parse_args()

    data_dir = Path(args.data)

    review_file = (
        data_dir
        / "engineer_review_2026-02.xlsx"
    )

    # --------------------------------------------------------
    # Load telemetry
    # --------------------------------------------------------

    telemetry = load_telemetry(
        data_dir
    )

    telemetry["gateway_id"] = (
        telemetry["gateway_id"]
        .map(normalize_gateway_id)
    )

    # --------------------------------------------------------
    # Create gateway holdout
    # --------------------------------------------------------

    all_gateways = set(
        telemetry["gateway_id"].unique()
    )

    (
        training_gateways,
        unseen_gateways
    ) = create_holdout_gateways(
        all_gateways
    )

    print()
    print("=" * 80)
    print("UNSEEN GATEWAY VALIDATION")
    print("=" * 80)

    print(
        f"Total gateways       : "
        f"{len(all_gateways)}"
    )

    print(
        f"Training gateways    : "
        f"{len(training_gateways)}"
    )

    print(
        f"Unseen test gateways : "
        f"{len(unseen_gateways)}"
    )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # Use UTC-aware dates because telemetry is UTC-aware.
    # --------------------------------------------------------

    scored_weeks = pd.date_range(
        "2026-02-02",
        "2026-03-23",
        freq="7D",
        tz="UTC"
    )

    weekly_results = []

    # ========================================================
    # TEST EACH WEEK
    # ========================================================

    for week_start in scored_weeks:

        print()
        print(
            f"Testing week: "
            f"{week_start.date()}"
        )

        historical_start = (
            week_start
            - pd.Timedelta(
                days=HISTORICAL_DAYS
            )
        )

        recent_start = (
            week_start
            - pd.Timedelta(
                days=RECENT_DAYS
            )
        )

        # ----------------------------------------------------
        # Historical data
        # ----------------------------------------------------

        historical = telemetry[
            (
                telemetry["ts_utc"]
                >= historical_start
            )
            &
            (
                telemetry["ts_utc"]
                < recent_start
            )
        ].copy()

        # ----------------------------------------------------
        # Recent/future data
        # ----------------------------------------------------

        recent = telemetry[
            (
                telemetry["ts_utc"]
                >= recent_start
            )
            &
            (
                telemetry["ts_utc"]
                < week_start
            )
        ].copy()

        # ----------------------------------------------------
        # ONLY training gateways are used for model training
        # ----------------------------------------------------

        historical_train = historical[
            historical["gateway_id"]
            .isin(training_gateways)
        ].copy()

        # ----------------------------------------------------
        # Historical data for unseen gateways
        #
        # These are NOT used to train Isolation Forest.
        # They are used only to calculate their own baseline.
        # ----------------------------------------------------

        historical_unseen = historical[
            historical["gateway_id"]
            .isin(unseen_gateways)
        ].copy()

        # ----------------------------------------------------
        # Future/recent data for unseen gateways
        # ----------------------------------------------------

        recent_unseen = recent[
            recent["gateway_id"]
            .isin(unseen_gateways)
        ].copy()

        if (
            historical_train.empty
            or historical_unseen.empty
            or recent_unseen.empty
        ):

            print(
                "Not enough data for this week."
            )

            continue

        # ----------------------------------------------------
        # Build training features
        # ----------------------------------------------------

        X_train, _ = build_features(
            historical_train,
            historical_train
        )

        # ----------------------------------------------------
        # Train Isolation Forest
        # ----------------------------------------------------

        model = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        print(
            "Training Isolation Forest..."
        )

        model.fit(
            X_train
        )

        # ----------------------------------------------------
        # Build features for unseen gateways
        #
        # Their historical data is NOT used to train the
        # Isolation Forest.
        # ----------------------------------------------------

        X_unseen, unseen_data = (
            build_features(
                historical_unseen,
                recent_unseen
            )
        )

        # ----------------------------------------------------
        # Calculate anomaly scores
        # ----------------------------------------------------

        anomaly_score = (
            -model.score_samples(
                X_unseen
            )
        )

        unseen_data = unseen_data.copy()

        unseen_data[
            "anomaly_score"
        ] = anomaly_score

        # ----------------------------------------------------
        # Aggregate scores by gateway
        # ----------------------------------------------------

        gateway_scores = (
            unseen_data
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

        gateway_scores = (
            gateway_scores
            .sort_values(
                [
                    "mean_score",
                    "max_score"
                ],
                ascending=False
            )
        )

        # ----------------------------------------------------
        # Select top 15 unseen gateways
        # ----------------------------------------------------

        top15 = (
            gateway_scores
            .head(15)
            .copy()
        )

        top15[
            "week_start"
        ] = week_start.date()

        weekly_results.append(
            top15
        )

        print(
            f"Unseen gateways scored: "
            f"{len(gateway_scores)}"
        )

        print(
            f"Top 15 selected: "
            f"{len(top15)}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    if not weekly_results:

        print(
            "No validation results generated."
        )

        return

    results = pd.concat(
        weekly_results,
        ignore_index=True
    )

    output_file = (
        Path(__file__).resolve().parent
        / "unseen_gateway_validation.csv"
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
        f"Saved: "
        f"{output_file.name}"
    )

    print(
        f"Rows: "
        f"{len(results)}"
    )

    print(
        f"Weeks tested: "
        f"{results['week_start'].nunique()}"
    )

    # ========================================================
    # ENGINEER REVIEW PROXY CHECK
    # ========================================================

    if review_file.exists():

        print()
        print("=" * 80)
        print("UNSEEN GATEWAY PROXY CHECK")
        print("=" * 80)

        review = pd.read_excel(
            review_file
        )

        review["gateway_id"] = (
            review["gateway_id"]
            .map(normalize_gateway_id)
        )

        review["is_bad"] = (
            review["Kategorie"]
            .astype(str)
            .str.strip()
            .eq("Schlecht")
        )

        results["gateway_id"] = (
            results["gateway_id"]
            .map(normalize_gateway_id)
        )

        evaluated = results.merge(
            review[
                [
                    "gateway_id",
                    "is_bad"
                ]
            ],
            on="gateway_id",
            how="inner"
        )

        if len(evaluated) > 0:

            known_bad = (
                evaluated["is_bad"]
                == True
            ).sum()

            known_normal = (
                evaluated["is_bad"]
                == False
            ).sum()

            precision = (
                known_bad
                / len(evaluated)
                * 100
            )

            proxy_cost = (
                known_normal
                * 380
            )

            print(
                f"Reviewed unseen-gateway selections : "
                f"{len(evaluated)}"
            )

            print(
                f"Known Schlecht                     : "
                f"{known_bad}"
            )

            print(
                f"Known Normal                       : "
                f"{known_normal}"
            )

            print(
                f"Proxy precision                    : "
                f"{precision:.2f}%"
            )

            print(
                f"Proxy wasted visits                : "
                f"{known_normal}"
            )

            print(
                f"Proxy visit cost                   : "
                f"€{proxy_cost}"
            )

            print()
            print(
                "IMPORTANT: This is proxy evidence only."
            )

            print(
                "Engineer review is NOT the official "
                "hidden challenge ground truth."
            )

        else:

            print(
                "No engineer-reviewed gateways "
                "appeared in the unseen selections."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()