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

HISTORY_DAYS = 28
TEST_WEEK_DAYS = 7

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

        df = df[required].copy()

        frames.append(df)

    telemetry = pd.concat(
        frames,
        ignore_index=True
    )

    telemetry["ts_utc"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True
    )

    telemetry["gateway_id"] = (
        telemetry["gateway_id"]
        .astype(str)
        .str.strip()
        .str.upper()
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
# BUILD FEATURES
# ============================================================

def create_features(
    historical,
    scoring
):

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

    stats.columns = [
        f"{metric}_{stat}"
        for metric, stat in stats.columns
    ]

    stats = stats.reset_index()

    for metric in METRICS:

        stats[f"{metric}_std"] = (
            stats[f"{metric}_std"]
            .replace(0, np.nan)
            .fillna(1.0)
        )

        stats[f"{metric}_mean"] = (
            stats[f"{metric}_mean"]
            .replace(0, np.nan)
            .fillna(1e-6)
        )

    data = scoring.merge(
        stats,
        on="gateway_id",
        how="left"
    )

    feature_columns = []

    for metric in METRICS:

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        max_col = f"{metric}_max"
        median_col = f"{metric}_median"

        # Raw value
        feature_columns.append(metric)

        # Deviation
        col = f"{metric}_deviation"

        data[col] = (
            data[metric]
            - data[mean_col]
        )

        feature_columns.append(col)

        # Z-score
        col = f"{metric}_zscore"

        data[col] = (
            data[metric]
            - data[mean_col]
        ) / data[std_col]

        feature_columns.append(col)

        # Ratio
        col = f"{metric}_ratio"

        data[col] = (
            data[metric]
            / data[mean_col]
        )

        feature_columns.append(col)

        # Difference from maximum
        col = f"{metric}_max_deviation"

        data[col] = (
            data[metric]
            - data[max_col]
        )

        feature_columns.append(col)

        # Difference from median
        col = f"{metric}_median_deviation"

        data[col] = (
            data[metric]
            - data[median_col]
        )

        feature_columns.append(col)

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
        description="Validate ML model on future weeks"
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

    # --------------------------------------------------------
    # Challenge weeks
    # --------------------------------------------------------

    test_weeks = pd.date_range(
        "2026-02-02",
        "2026-03-23",
        freq="7D",
        tz="UTC"
    )

    results = []

    print()
    print("=" * 80)
    print("FUTURE-WEEK VALIDATION")
    print("=" * 80)

    for test_week in test_weeks:

        # ----------------------------------------------------
        # Training period
        #
        # ONLY data before the test week is used.
        # ----------------------------------------------------

        train_end = test_week

        train_start = (
            test_week
            - pd.Timedelta(days=HISTORY_DAYS)
        )

        # ----------------------------------------------------
        # Test period
        # ----------------------------------------------------

        test_start = test_week

        test_end = (
            test_week
            + pd.Timedelta(days=TEST_WEEK_DAYS)
        )

        training_data = telemetry[
            (
                telemetry["ts_utc"]
                >= train_start
            )
            &
            (
                telemetry["ts_utc"]
                < train_end
            )
        ].copy()

        test_data = telemetry[
            (
                telemetry["ts_utc"]
                >= test_start
            )
            &
            (
                telemetry["ts_utc"]
                < test_end
            )
        ].copy()

        print()
        print(
            f"Test week: {test_week.date()}"
        )

        print(
            f"Training period: "
            f"{train_start.date()} "
            f"to "
            f"{train_end.date()}"
        )

        print(
            f"Test period: "
            f"{test_start.date()} "
            f"to "
            f"{test_end.date()}"
        )

        # ----------------------------------------------------
        # Build training features
        # ----------------------------------------------------

        X_train, _ = create_features(
            training_data,
            training_data
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(X_train)

        # ----------------------------------------------------
        # Build test features
        #
        # IMPORTANT:
        # Historical statistics come ONLY from training period.
        # ----------------------------------------------------

        X_test, test_features = create_features(
            training_data,
            test_data
        )

        # ----------------------------------------------------
        # Score future week
        # ----------------------------------------------------

        test_features = test_features.copy()

        test_features["anomaly_score"] = (
            -model.score_samples(X_test)
        )

        gateway_scores = (
            test_features
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

        top15 = gateway_scores.head(15).copy()

        top15["week_start"] = (
            test_week.date()
        )

        results.append(top15)

        print(
            f"Gateways scored: "
            f"{len(gateway_scores)}"
        )

        print(
            f"Top 15 selected: "
            f"{len(top15)}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    results = pd.concat(
        results,
        ignore_index=True
    )

    # Save validation result in project folder,
    # NOT inside the challenge data folder.
    output_file = (
        Path(__file__).resolve().parent
        / "future_week_validation.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 80)
    print("FUTURE-WEEK VALIDATION COMPLETE")
    print("=" * 80)

    print(
        f"Saved: {output_file.name}"
    )

    print(
        f"Rows: {len(results)}"
    )

    print(
        f"Weeks tested: "
        f"{results['week_start'].nunique()}"
    )

    # ========================================================
    # PROXY CHECK
    # ========================================================

    if review_file.exists():

        review = pd.read_excel(
            review_file
        )

        review["gateway_id"] = (
            review["gateway_id"]
            .astype(str)
            .str.strip()
            .str.replace(":", "", regex=False)
            .str.upper()
        )

        review["is_bad"] = (
            review["Kategorie"]
            .astype(str)
            .str.strip()
            .eq("Schlecht")
        )

        results["gateway_id"] = (
            results["gateway_id"]
            .astype(str)
            .str.strip()
            .str.upper()
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

        print()
        print("=" * 80)
        print("FUTURE-WEEK PROXY CHECK")
        print("=" * 80)

        if len(evaluated) > 0:

            known_bad = (
                evaluated["is_bad"] == True
            ).sum()

            known_normal = (
                evaluated["is_bad"] == False
            ).sum()

            precision = (
                known_bad
                / len(evaluated)
                * 100
            )

            proxy_cost = (
                known_normal * 380
            )

            print(
                f"Reviewed selections : "
                f"{len(evaluated)}"
            )

            print(
                f"Known Schlecht      : "
                f"{known_bad}"
            )

            print(
                f"Known Normal        : "
                f"{known_normal}"
            )

            print(
                f"Proxy precision     : "
                f"{precision:.2f}%"
            )

            print(
                f"Proxy wasted visits : "
                f"{known_normal}"
            )

            print(
                f"Proxy visit cost    : "
                f"€{proxy_cost}"
            )

        else:

            print(
                "No engineer-reviewed "
                "selections available."
            )

        print()
        print(
            "IMPORTANT: Engineer review is "
            "proxy evidence only, not official "
            "hidden ground truth."
        )


if __name__ == "__main__":
    main()