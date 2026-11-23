"""Export the local SciCoQA candidates for manual review."""
from pathlib import Path

import pandas as pd

SOURCE = Path("project_sample/data/prepared/scicoqa_real_screened.parquet")
OUTPUT = Path("data_review/review_queue.tsv")
SOURCE_COLUMNS = [
    "discrepancy_id",
    "paper_url",
    "paper_url_versioned",
    "code_url",
    "code_url_versioned",
    "origin_url",
    "discrepancy_category",
    "discrepancy_type",
    "validator_disagreement",
    "origin_discrepancy_text",
]
REVIEW_COLUMNS = [
    "study_id",
    "review_category",
    "review_status",
    "field",
    "paper_value",
    "code_value",
    "paper_location",
    "paper_excerpt",
    "code_location",
    "code_excerpt",
    "review_notes",
]


def main():
    screened = pd.read_parquet(SOURCE, columns=[*SOURCE_COLUMNS, "scope_status"])
    candidates = screened.loc[
        screened["scope_status"].eq("candidate_in_scope"), SOURCE_COLUMNS
    ].copy()
    ids = candidates["discrepancy_id"]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Review candidates must have unique discrepancy IDs")

    candidates = candidates.sort_values("discrepancy_id").assign(
        **dict.fromkeys(REVIEW_COLUMNS, "")
    )
    # Exclusive creation protects annotations already entered in the queue.
    candidates.to_csv(OUTPUT, sep="\t", index=False, encoding="utf-8-sig", mode="x")


if __name__ == "__main__":
    main()
