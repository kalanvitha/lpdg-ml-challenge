import argparse
import pandas as pd
from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA_DIR = PROJECT_DIR / "data"

BASELINE_FILE = "predictions_baseline.csv"
FINAL_FILE = "predictions.csv"

WASTED_VISIT_COST = 380


# =========================================================
# COMMAND-LINE ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(
    description="Compare the supplied 3-sigma baseline with the final ML model"
)

parser.add_argument(
    "--data",
    type=Path,
    default=DEFAULT_DATA_DIR,
    help="Path to challenge data directory"
)

args = parser.parse_args()

DATA_DIR = args.data


# =========================================================
# LOAD ENGINEER REVIEW
# =========================================================

review_file = DATA_DIR / "engineer_review_2026-02.xlsx"

if not review_file.exists():
    raise FileNotFoundError(
        f"Engineer review file not found: {review_file}"
    )

print(
    "Loading engineer review from:",
    review_file
)

review = pd.read_excel(review_file)

review["gateway_id"] = (
    review["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.upper()
)

review["is_bad"] = (
    review["Kategorie"]
    .astype(str)
    .str.strip()
    == "Schlecht"
)


# =========================================================
# EVALUATION FUNCTION
# =========================================================

def evaluate_predictions(version, filename):

    prediction_file = PROJECT_DIR / filename

    if not prediction_file.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {prediction_file}"
        )

    print(
        f"Reading {version}: {prediction_file}"
    )

    predictions = pd.read_csv(
        prediction_file
    )

    predictions["gateway_id"] = (
        predictions["gateway_id"]
        .astype(str)
        .str.replace(":", "", regex=False)
        .str.upper()
    )

    merged = predictions.merge(
        review[
            [
                "gateway_id",
                "Kategorie",
                "is_bad"
            ]
        ],
        on="gateway_id",
        how="left"
    )

    evaluated = merged[
        merged["Kategorie"].notna()
    ].copy()

    known_bad = int(
        (
            evaluated["is_bad"]
            == True
        ).sum()
    )

    known_normal = int(
        (
            evaluated["is_bad"]
            == False
        ).sum()
    )

    reviewed_predictions = (
        known_bad
        + known_normal
    )

    if reviewed_predictions > 0:
        precision = (
            known_bad
            / reviewed_predictions
            * 100
        )
    else:
        precision = 0.0

    proxy_wasted_visits = known_normal

    proxy_cost = (
        proxy_wasted_visits
        * WASTED_VISIT_COST
    )

    return {
        "Version": version,
        "Prediction rows": len(predictions),
        "Reviewed predictions": reviewed_predictions,
        "Known Schlecht": known_bad,
        "Known Normal": known_normal,
        "Precision %": round(
            precision,
            2
        ),
        "Proxy wasted visits": proxy_wasted_visits,
        "Proxy visit cost (€)": proxy_cost
    }


# =========================================================
# EVALUATE BASELINE AND FINAL MODEL
# =========================================================

baseline_result = evaluate_predictions(
    "3-Sigma Baseline",
    BASELINE_FILE
)

final_result = evaluate_predictions(
    "Final ML V4",
    FINAL_FILE
)

result_df = pd.DataFrame(
    [
        baseline_result,
        final_result
    ]
)


# =========================================================
# DISPLAY COMPARISON
# =========================================================

print()
print("=" * 90)
print("3-SIGMA BASELINE vs FINAL ML MODEL")
print("=" * 90)

print(
    result_df.to_string(
        index=False
    )
)


# =========================================================
# COST IMPROVEMENT
# =========================================================

baseline_cost = baseline_result[
    "Proxy visit cost (€)"
]

final_cost = final_result[
    "Proxy visit cost (€)"
]

cost_reduction = (
    baseline_cost
    - final_cost
)

if baseline_cost > 0:

    percentage_reduction = (
        cost_reduction
        / baseline_cost
        * 100
    )

else:

    percentage_reduction = 0.0


# =========================================================
# FINAL RESULT
# =========================================================

print()
print("=" * 90)
print("FINAL ML IMPROVEMENT")
print("=" * 90)

print(
    f"Baseline proxy cost : €{baseline_cost}"
)

print(
    f"Final ML proxy cost : €{final_cost}"
)

print(
    f"Cost reduction      : €{cost_reduction}"
)

print(
    f"Percentage reduction: "
    f"{percentage_reduction:.2f}%"
)


# =========================================================
# SANITY CHECK
# =========================================================

print()
print("=" * 90)
print("SANITY CHECK")
print("=" * 90)

for _, row in result_df.iterrows():

    expected_normal = (
        row["Reviewed predictions"]
        - row["Known Schlecht"]
    )

    if expected_normal != row["Known Normal"]:

        print(
            f"WARNING: {row['Version']} "
            f"has inconsistent counts!"
        )

    else:

        print(
            f"{row['Version']}: OK"
        )


# =========================================================
# PROXY DISCLAIMER
# =========================================================

print()
print("=" * 90)
print("IMPORTANT")
print("=" * 90)

print(
    "Engineer review is proxy evidence only."
)

print(
    "It is NOT the official hidden challenge "
    "ground truth or official challenge score."
)