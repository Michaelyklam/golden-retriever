# MiniCPM5-1B RTX 5060 Ti benchmark

Folder created: `/home/michael/minicpm5-1b-bench`

## Environment

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM
- Driver: 580.159.03
- CUDA reported by driver: 13.0
- Python venv: `.venv` created with `uv`
- Framework used: vLLM 0.24.0 with PyTorch 2.11.0+cu130
- Model: `openbmb/MiniCPM5-1B`
- Precision: BF16
- Context cap for benchmark: `--max-model-len 2048`

Note: `nvidia-smi` initially failed with a loaded-kernel/userland driver mismatch (`580.142` kernel module vs `580.159` userland). I reloaded the NVIDIA modules with sudo, after which the GPU was visible and usable.

## Required workaround

On this RTX 5060 Ti / SM 12.0 setup, vLLM's default FlashInfer sampler path errored during startup with:

```text
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

The GPU is obviously newer than SM75; this appears to be an SM 12.x detection / FlashInfer path issue. The working command disables FlashInfer sampling:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

vLLM still used CUDA, BF16, FlashAttention, CUDA graphs, and torch.compile.

## Re-run max-throughput benchmark

```bash
cd /home/michael/minicpm5-1b-bench
source .venv/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm bench throughput \
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
```

Observed result:

```text
Throughput: 25.06 requests/s, 19243.52 total tokens/s, 6414.51 output tokens/s
Total num prompt tokens:  131072
Total num output tokens:  65536
```

## Re-run single-request latency benchmark

```bash
cd /home/michael/minicpm5-1b-bench
source .venv/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm bench latency \
  --model openbmb/MiniCPM5-1B \
  --input-len 512 \
  --output-len 256 \
  --batch-size 1 \
  --num-iters 10 \
  --num-iters-warmup 3 \
  --max-model-len 2048 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

Observed result:

```text
Avg latency: 1.3022316282615065 seconds
```

That is approximately `256 / 1.3022316282615065 = 196.6` output tokens/second for one active request.
