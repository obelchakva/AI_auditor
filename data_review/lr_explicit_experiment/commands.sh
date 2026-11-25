#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m ai_auditor --check initial-lr --pdf /tmp/ai-final-eval/crd/paper.pdf --repo /tmp/ai-final-eval/crd/git --commit b84f547c5db6a35318d4671d7d5c4de74c822403 --stage 'Implementation details' --paper-section '6.4 Implementation details' --paper-context CIFAR-100 --paper-model-pair resnet32x4 resnet8x4 --paper-table 1 --recipe-arg=--model_s=resnet8x4 --recipe-arg=--path_t=./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --recipe-arg=--distill=crd --recipe-arg=-a=0 --recipe-context 'command for running CRD' > data_review/lr_explicit_experiment/crd-selected.json

.venv/bin/python -m ai_auditor --check initial-lr --pdf /tmp/ai-final-eval/crd/paper.pdf --repo /tmp/ai-final-eval/crd/git --commit b84f547c5db6a35318d4671d7d5c4de74c822403 --stage 'Implementation details' --paper-section '6.4 Implementation details' --paper-context CIFAR-100 --paper-model-pair resnet32x4 resnet8x4 --paper-table 1 --recipe-arg=--model_s=resnet8x4 --recipe-arg=--path_t=./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --recipe-arg=--distill=crd > data_review/lr_explicit_experiment/crd-ambiguous.json

.venv/bin/python -m ai_auditor --check initial-lr --pdf /tmp/ai-final-eval/sam/iccv.pdf --repo /tmp/ai-final-eval/sam/git --commit dca509fe793f601edb92606367a655c15ac00fdf --stage training > data_review/lr_explicit_experiment/sam-original.json
