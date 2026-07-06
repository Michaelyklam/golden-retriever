#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/minicpm5-1b-bench}"
mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d .venv ]; then
  uv venv .venv --python python3
  # vLLM currently carries the most mature high-throughput serving stack for this model on NVIDIA.
  . .venv/bin/activate
  uv pip install 'vllm>=0.21' 'transformers>=5.6' 'huggingface_hub'
else
  . .venv/bin/activate
fi

# Required on RTX 5060 Ti / SM 12.0 with current vLLM+FlashInfer wheels.
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm bench throughput \
  --model openbmb/MiniCPM5-1B \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 256 \
  --random-range-ratio 0 \
  --num-prompts 256 \
  --num-warmups 16 \
  --max-model-len 2048 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
