#!/usr/bin/env bash
# Run from the root of AI_auditor.
set -euo pipefail

report_dir=data_review/lr_portability
cache_dir=${1:-$(mktemp -d)}
output_dir=${2:-$report_dir/replay}
mkdir -p "$cache_dir" "$output_dir"
cache_dir=$(realpath "$cache_dir")
sha256sum -c "$report_dir/expectations.sha256"

run_case() {
    local name=$1 id=$2 stage=$3 url=$4 digest=$5 upstream=$6 commit=$7
    local input="$cache_dir/$name"
    mkdir -p "$input"
    if [[ ! -f "$input/paper.pdf" ]]; then
        curl --fail --location --silent --show-error --retry 2 "$url" -o "$input/paper.download"
        mv "$input/paper.download" "$input/paper.pdf"
    fi
    printf '%s  %s\n' "$digest" "$input/paper.pdf" | sha256sum -c -
    if [[ ! -f "$input/git/HEAD" ]]; then
        git init --bare --quiet "$input/git"
    fi
    if ! git -C "$input/git" cat-file -e "$commit^{commit}" 2>/dev/null; then
        git -C "$input/git" fetch --depth=1 --no-tags "$upstream" "$commit"
    fi
    (
        set -x
        .venv/bin/python -m ai_auditor --pdf "$input/paper.pdf" --repo "$input/git" \
            --commit "$commit" --stage "$stage"
    ) > "$output_dir/$id.json" 2> "$output_dir/$id.command.log"
}

run_case xai 3bdc7c6f 'MNIST autoencoder' \
    'https://proceedings.mlr.press/v162/crabbe22a/crabbe22a.pdf' \
    fadbb94a9cf8e8bf27046f0847793ab0f8036f88347af24beb88c78bdec02ac9 \
    'https://github.com/JonathanCrabbe/Label-Free-XAI.git' 3f6fd5e98ab2fe431d6c15abeb5896c3c7747c2b

run_case bana b019c5d8 'Classification network' \
    'https://openaccess.thecvf.com/content/CVPR2021/papers/Oh_Background-Aware_Pooling_and_Noise-Aware_Loss_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2021_paper.pdf' \
    d21dc74b249f0b2c11db581967f47edf7341e4f9a93605d51a6112ada72bffd3 \
    'https://github.com/cvlab-yonsei/BANA.git' 407e92e521dc2db90055470f7ff2ccb5aa474093

run_case gcn 2c296ba3 'Experiment Settings' \
    'https://aclanthology.org/D18-1032.pdf' \
    89a88c497c757eb5cedc7645f0e0616ffb269094e0866825f62fe36c0dafb8ed \
    'https://github.com/1049451037/GCN-Align.git' 4fc90c438e5a609b96df03daff170fbcf03fde94
