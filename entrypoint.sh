#!/usr/bin/env bash
# Build the `vllm serve` command from ENV tunables, then exec it.
#
# WHY a script instead of a baked CMD: every knob below is an ENV var, so changing
# config means `docker run -e VAR=...` (or a host-template env field) with NO image
# rebuild. The Dockerfile only carries DEFAULTS in its last, cheap layer.
#
# Two kinds of flag:
#   - STRUCTURAL (hardcoded here): they define what this image IS for SM120 + Devstral.
#     Don't turn them into knobs — changing them changes the recipe.
#   - TUNABLE (ENV): perf/capacity dials you sweep per host.
set -euo pipefail

# --- Reverse proxy -----------------------------------------------------------
# Caddy äger den publika porten (PORT, default 8000 — den Vast mappar). vLLM
# binder bara till 127.0.0.1:VLLM_PORT och nås aldrig direkt utifrån. Proxyn är
# default-deny: bara /v1/* (vLLM:s egen nyckel) + token:at /metrics,/health.
PUBLIC_PORT="${PORT:-8000}"
VLLM_HOST="127.0.0.1"
VLLM_PORT="${VLLM_PORT:-8001}"

# Metrics/health-token: separat METRICS_TOKEN, annars återanvänd VLLM_API_KEY så
# klienten bara behöver en nyckel. Tomt → /metrics och /health nekas (403).
METRICS_TOKEN="${METRICS_TOKEN:-${VLLM_API_KEY:-}}"
export CADDY_PORT="$PUBLIC_PORT" METRICS_TOKEN
if [ -z "$METRICS_TOKEN" ]; then
  echo "WARN: varken METRICS_TOKEN eller VLLM_API_KEY satt → /metrics och /health returnerar 403 (låst)."
fi

echo "+ caddy: publik :${PUBLIC_PORT} → vLLM 127.0.0.1:${VLLM_PORT} (default-deny; /v1/* öppet, /metrics,/health token:at)"
caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &

args=(
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --model "${MODEL:?MODEL env required}"

  # --- STRUCTURAL: the Blackwell + Devstral recipe ---
  # Mistral-native path end-to-end → uses vLLM's MistralTokenizer (has is_fast),
  # sidestepping the transformers-v5 `CachedMistralCommonBackend.is_fast` crash.
  --tokenizer-mode mistral --config-format mistral --load-format mistral
  --tool-call-parser mistral --enable-auto-tool-choice
  --enable-prefix-caching --enable-prompt-tokens-details

  # --- VISION: keep the Pixtral tower live (SAM's view_image feeds images IN) ---
  # FlashAttention has no SM120 build and FlashInfer rejects the ViT head size, so the
  # encoder must run on a kernel-free path. TORCH_SDPA is the portable default; if the
  # encoder still crashes at startup, try XFORMERS (see README fallback ladder).
  --mm-encoder-attn-backend "${MM_ENCODER_BACKEND:-TORCH_SDPA}"
  --limit-mm-per-prompt "{\"image\":${IMAGE_LIMIT:-4}}"

  # --- TUNABLE: perf/capacity ---
  --max-model-len "${MAX_MODEL_LEN:-131072}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED:-16384}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}"
)

# Pin an immutable model commit so every cold start pulls identical weights/tokenizer.
[ -n "${REVISION:-}" ] && args+=(--revision "$REVISION")

# Speculative decoding — vLLM 0.24.0 takes ONE --speculative-config JSON arg (the old
# --spec-method/--spec-tokens flags are GONE). Empty by default: ngram INVERTS under high
# concurrency → use only for low-concurrency/dev. Passed quoted, so spaces in the JSON are fine.
SPEC_CONFIG='{"method":"ngram","num_speculative_tokens":10,"prompt_lookup_max":4,"prompt_lookup_min":2}'
[ -n "${SPEC_CONFIG:-}" ] && args+=(--speculative-config "$SPEC_CONFIG")

# Escape hatch for anything else without a rebuild. Word-split intentionally.
[ -n "${EXTRA_ARGS:-}" ] && args+=($EXTRA_ARGS)

echo "+ vllm serve ${args[*]}"
exec vllm serve "${args[@]}"
