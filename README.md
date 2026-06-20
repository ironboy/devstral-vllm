# devstral-vllm

A self-contained, **vision-capable** inference image for **Devstral Small 2 (24B)**
served with **vLLM**. Runs on any Docker-based GPU host — serverless GPU platforms,
Kubernetes, or bare metal.

It is the official vLLM image plus a single source guard so the Pixtral (vision)
multimodal init doesn't crash on this transformers build — see the comments in the
[`Dockerfile`](./Dockerfile) for the exact reason.

## Requirements
- One NVIDIA GPU with enough VRAM for a 24B model in bf16 (~48 GB) — e.g. H100/H200/B200/A100-80G.
- CUDA-capable Docker runtime (`--gpus all` / nvidia-container-toolkit).
- ~48 GB free disk for the weights (downloaded from Hugging Face on first start).

## Build
```bash
docker build -t devstral-vllm:0.22.1 .
```

## Run
```bash
docker run --gpus all -p 8000:8000 \
  -e VLLM_API_KEY=<your-api-key> \
  devstral-vllm:0.22.1 \
  --host 0.0.0.0 --port 8000 \
  --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
  --revision f2ca762c466d28ab948b7205492ceb3914e73f8a \
  --max-model-len 131072 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser mistral \
  --spec-method ngram --spec-tokens 8
```

Notes:
- `--revision` pins an immutable model commit so a fresh download always gets the
  same weights/tokenizer.
- `--spec-method ngram --spec-tokens 8` enables n-gram speculative decoding (faster
  single-stream decode). Drop it to compare.
- `VLLM_API_KEY` (env) is the bearer token the server requires; clients must match it.
- Exposes an OpenAI-compatible API at `/v1/chat/completions`; health at `/health`
  (returns 200 once the model is loaded — allow a generous startup/grace window for
  the first weight download).

## Notes on serving platforms
- The model loads the API server only **after** weights are loaded, so set the
  platform's health-check **grace period** high enough for the first download
  (~10–15 min) or it will be killed mid-download.
- No persistent volume is required; with `--revision` pinned, each cold start pulls
  the same weights.
