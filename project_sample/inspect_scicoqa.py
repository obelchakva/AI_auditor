"""Download and run a first quality audit of the SciCoQA real split."""
from pathlib import Path
import json
import pandas as pd
from datasets import load_dataset

BASE = Path("data")
RAW = BASE / "raw"
PREPARED = BASE / "prepared"
QUARANTINE = BASE / "quarantine"
REPORTS = BASE / "reports"
for folder in (RAW, PREPARED, QUARANTINE, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

# Preserve the source snapshot before any filtering or transformation.
ds = load_dataset("UKPLab/scicoqa", split="real")
ds.to_parquet(RAW / "scicoqa_real.parquet")
df = ds.to_pandas()

required = ["discrepancy_id", "paper_url", "code_url_versioned", "origin_url", "discrepancy_date"]
missing = df.isna().sum().rename("missing_count").to_frame()
missing["missing_share"] = (missing["missing_count"] / len(df)).round(4)
missing.to_csv(REPORTS / "missing_by_field.csv")

# Duplicate the discrepancy ID only; several discrepancies may legitimately
# belong to the same paper/repository pair, so do not deduplicate by that pair.
duplicate_id = df["discrepancy_id"].duplicated(keep="first")
missing_required = df[required].isna().any(axis=1)
missing_label = df[["discrepancy_type", "discrepancy_category"]].isna().any(axis=1)

# Model-validator disagreement is retained, but explicitly flagged for expert review.
validator_disagreement = (
    df["is_valid_discrepancy_gpt"].notna()
    & df["is_valid_discrepancy_gemini"].notna()
    & (df["is_valid_discrepancy_gpt"] != df["is_valid_discrepancy_gemini"])
)

quarantine_mask = duplicate_id | missing_required | missing_label
quarantine = df.loc[quarantine_mask].copy()
quarantine.to_parquet(QUARANTINE / "scicoqa_real_quarantine.parquet", index=False)

screened = df.loc[~quarantine_mask].copy()
screened["validator_disagreement"] = validator_disagreement.loc[screened.index].values
screened["scope_status"] = screened["discrepancy_category"].map(
    lambda x: "candidate_in_scope" if x in {"Training", "Data"} else "out_of_scope_review"
)
screened.to_parquet(PREPARED / "scicoqa_real_screened.parquet", index=False)

summary = {
    "source": "UKPLab/scicoqa",
    "split": "real",
    "rows_raw": int(len(df)),
    "duplicate_discrepancy_id": int(duplicate_id.sum()),
    "rows_missing_required_fields": int(missing_required.sum()),
    "rows_missing_scope_labels": int(missing_label.sum()),
    "rows_with_gpt_gemini_disagreement": int(validator_disagreement.sum()),
    "rows_quarantined": int(quarantine_mask.sum()),
    "rows_screened": int(len(screened)),
    "category_counts": {str(k): int(v) for k, v in df["discrepancy_category"].value_counts(dropna=False).items()},
    "type_counts": {str(k): int(v) for k, v in df["discrepancy_type"].value_counts(dropna=False).items()},
    "note": "No imputation is performed. Missing labels and source fields are not guessed.",
}
(REPORTS / "quality_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"Saved raw source to: {RAW / 'scicoqa_real.parquet'}")
