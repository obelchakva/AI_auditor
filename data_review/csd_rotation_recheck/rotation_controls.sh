#!/usr/bin/env bash
set -euo pipefail

base=paper_experiments/rotated_mnist/additional_unlabeled_domain/add_30
.venv/bin/python -m ai_auditor --check test-rotation \
  --pdf /tmp/ai-final-eval/diva/midl.pdf --repo /tmp/ai-final-eval/diva/git \
  --commit 4c5282a8e54feee01626f5e8a54595ea570ac169 --stage 'Semi-supervised domain generalization' \
  --dataset 'Rotated MNIST' --domain 75 --rotation-variant 30 --rotation-table 2 \
  --rotation-producer "$base/experiment.py" \
  --rotation-evaluator "$base/trained_models/sup_unsup_1_to_1/calculate_test.py" \
  --rotation-config-field list_test_domain > data_review/csd_rotation_recheck/diva.json

.venv/bin/python -m ai_auditor --check test-rotation \
  --pdf /tmp/rotation-pilot/paper.pdf --repo /tmp/rotation-pilot/git \
  --commit 3eee1730ae9efaae2bd763f9835ac540b9bde02a \
  --stage MatchDG --dataset 'Rotated MNIST' --domain 90 > data_review/csd_rotation_recheck/90c14330.json
