#!/usr/bin/env bash
set -euo pipefail
root=$(pwd -P)
dir="$root/data_review/defense_mvp"
output=${1:-$dir/demo_runs}
mkdir "$output"
output=$(realpath "$output")
sha256sum -c "$dir/cli_snapshot.sha256"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tar -xzf "$dir/cli_snapshot.tar.gz" -C "$work"
cd "$work"
audit=("$root/.venv/bin/python" -E -s -m ai_auditor)
run() {
  local name=$1
  shift
  local status=0
  "${audit[@]}" "$@" >"$output/$name.json" 2>"$output/$name.stderr.txt" || status=$?
  printf '%s\n' "$status" >"$output/$name.exit"
  return "$status"
}

run dpo --check initial-lr --pdf /tmp/defense-inputs/dpo/paper.pdf --repo /tmp/defense-inputs/dpo/git \
  --commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f --stage dialogue \
  --paper-section 'B DPO Implementation Details and Hyperparameters' --paper-context RMSprop \
  --paper-experiment 'single-turn dialogue' --paper-experiment 'Anthropic HH' --paper-experiment 'Pythia-2.8B' \
  --paper-scope DPO --recipe-arg=model=pythia28 '--recipe-arg=datasets=[hh]' \
  --recipe-arg=batch_size=64 --recipe-arg=loss=dpo --recipe-arg=loss.beta=0.1

run csd --check test-rotation --pdf /tmp/ind-eval/csd/paper.pdf --repo /tmp/ind-eval/csd/git \
  --commit 9902cdc8ea54f2650cd1396f904a06598a864a76 --stage 'Rotation tasks' --dataset 'Rotated MNIST' --domain 90

run lora --check initial-lr --pdf /tmp/mvp-inputs/lora/paper.pdf --repo /tmp/mvp-inputs/lora/git \
  --commit c4593f060e6a368d7bb5af5273b8e42810cdef90 --stage 'RoBERTa large LoRA CoLA' \
  --paper-lr-table 9 'RoBERTa-large LoRA' CoLA --table-shell-recipe examples/NLU/roberta_large_cola.sh
