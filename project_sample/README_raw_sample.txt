SciCoQA real split — 6-row raw excerpt

Source: https://huggingface.co/datasets/UKPLab/scicoqa
License listed by publisher: CC-BY-4.0
Dataset authors/curators: Tim Baumgärtner and Iryna Gurevych, UKP Lab, Technical University of Darmstadt.

This file is a six-record excerpt reconstructed from the public Hugging Face dataset viewer on 2026-11-02. It preserves the displayed values for the selected metadata and label fields. It is NOT a copy of the full parquet/jsonl dataset and does not include long issue/description/evidence text fields. The six rows shown are the first six records visible in the real split preview.

For the complete original dataset, download the `real-00000-of-00001.parquet` file from the dataset Files tab or run:

from datasets import load_dataset
raw = load_dataset("UKPLab/scicoqa", split="real")
raw.to_parquet("raw/scicoqa_real.parquet")

Do not overwrite this excerpt; use it only as a sample for schema and initial quality review.
