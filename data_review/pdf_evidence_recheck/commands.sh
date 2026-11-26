#!/usr/bin/env bash
set -euo pipefail
root=$(pwd -P)
baseline="$root/data_review/independent_evaluation"
updated="$root/data_review/pdf_evidence_recheck"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp "$baseline/run.py" "$work/run.py"
for file in cases.tsv cases.sha256 runtime.txt runtime.lock.txt; do
  ln -s "$baseline/$file" "$work/$file"
done
for file in cli_snapshot.tar.gz cli_snapshot.sha256; do
  ln -s "$updated/$file" "$work/$file"
done
"$root/.venv/bin/python" "$work/run.py" \
  --inputs "${1:-/tmp/ind-eval}" --output "${2:-$updated/runs}" "${@:3}"
