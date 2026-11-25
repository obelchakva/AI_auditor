#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python -m ai_auditor --check initial-lr --pdf /tmp/ai-final-eval/crd/paper.pdf --repo /tmp/ai-final-eval/crd/git --commit b84f547c5db6a35318d4671d7d5c4de74c822403 --stage 'Implementation details' --paper-section '6.4 Implementation details' --paper-context CIFAR-100 --paper-model-pair resnet32x4 resnet8x4 --paper-table 1 --recipe-arg=--model_s=resnet8x4 --recipe-arg=--path_t=./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --recipe-arg=--distill=crd --recipe-arg=-a=0 --recipe-context 'command for running CRD' > data_review/lr_generated_input/crd-selected.json

.venv/bin/python -m ai_auditor --check initial-lr --pdf /tmp/ai-final-eval/crd/paper.pdf --repo /tmp/ai-final-eval/crd/git --commit b84f547c5db6a35318d4671d7d5c4de74c822403 --stage 'Implementation details' --paper-section '6.4 Implementation details' --paper-context CIFAR-100 --paper-model-pair resnet32x4 resnet8x4 --paper-table 1 --recipe-arg=--model_s=resnet8x4 --recipe-arg=--path_t=./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth --recipe-arg=--distill=crd > data_review/lr_generated_input/crd-ambiguous.json

.venv/bin/python -m ai_auditor --check initial-lr --pdf /tmp/ai-final-eval/sam/iccv.pdf --repo /tmp/ai-final-eval/sam/git --commit dca509fe793f601edb92606367a655c15ac00fdf --stage training > data_review/lr_generated_input/sam-original.json

dpo=(
  .venv/bin/python -m ai_auditor --check initial-lr
  --pdf /tmp/ai-final-eval/dpo/paper.pdf --repo /tmp/ai-final-eval/dpo/git
  --commit f8b8c0f49dc92a430bae41585f9d467d3618fe2f --stage dialogue
  --paper-section 'B DPO Implementation Details and Hyperparameters'
  --paper-context RMSprop --paper-experiment 'single-turn dialogue'
  --paper-experiment 'Anthropic HH' --paper-experiment 'Pythia-2.8B' --paper-scope DPO
  --recipe-arg=model=pythia28 '--recipe-arg=datasets=[hh]' --recipe-arg=batch_size=64
)
"${dpo[@]}" --recipe-arg=loss=dpo --recipe-arg=loss.beta=0.1 > data_review/lr_generated_input/dpo-selected.json
"${dpo[@]}" > data_review/lr_generated_input/dpo-ambiguous.json

geq=(
  .venv/bin/python -m ai_auditor --check initial-lr
  --pdf /tmp/ai-final-eval/geq/arxiv.pdf --repo /tmp/ai-final-eval/geq/git
  --commit f96ff93ba0a2cc7509a5463e0897d9a4a556a798 --stage 'multiple conformations'
  --paper-section 'Implementation details' --paper-context 'multiple conformations' --paper-input 13C
  --recipe-script main.py --recipe-input=--train_path --recipe-producer data/create_data.py
  --recipe-transform add_conformations_to_train_dataset
)
"${geq[@]}" --recipe-arg=--train_path=datasets/train_data_13C.pickle > data_review/lr_generated_input/geqshift-selected.json
"${geq[@]}" --recipe-arg=--train_path=datasets/train_data_1H.pickle > data_review/lr_generated_input/geqshift-other-input.json
