# Devstral Small 2 (24B) on vLLM — vision-capable inference image.
# ░░ BLACKWELL (RTX PRO 6000 / SM 120) variant — branch `blackwell` ░░
#
# Differs from `main` (which targets Ada/H100/H200 on vLLM 0.22.1). Three things change
# for an SM 120 card, all verified against upstream:
#
#  1) BASE = vLLM 0.24.0. 0.22.1 prints "SM 12.x requires CUDA >= 12.9" (its cu128 wheel
#     is too old for Blackwell). 0.24.0 is the first release that lists SM120 support AND
#     contains the fetch_images fix (PR #45180) — so NO sed-guard is needed here, unlike
#     main. (Image is cu129; that already satisfies the >=12.9 gate. For a true CUDA-13
#     userspace use the cu130 nightly — see README.)
#  2) Tokenizer: we serve in mistral mode (entrypoint), using vLLM's native
#     MistralTokenizer, which dodges the transformers-v5 `is_fast` crash. No transformers
#     pin needed — the stock image's bundled versions are fine.
#  3) Vision encoder: FlashAttention has no SM120 build, so the Pixtral ViT is forced onto
#     TORCH_SDPA via --mm-encoder-attn-backend (entrypoint). << the one UNPROVEN bit:
#     smoke-test a real image right after launch, see README.
#
# No CUDA forward-compat shim (unlike main): SM120 needs a real >=580 host driver; the
# compat libs can't back-fill Blackwell support, so we keep them off the library path.

FROM vllm/vllm-openai:v0.24.0

# Blackwell: FlashAttention-3 has no SM120 build → pin the LLM attention to the FA2 path.
ENV VLLM_FLASH_ATTN_VERSION=2 \
    PYTHONUNBUFFERED=1

# Reverse proxy (static binary, ~40 MB): fronts vLLM so only /v1/* + token'd
# /metrics,/health are reachable; everything else (server_info, tokenize, docs…) is
# default-denied. vLLM itself binds 127.0.0.1 only (entrypoint) — never exposed direct.
COPY --from=caddy:2 /usr/bin/caddy /usr/bin/caddy
COPY Caddyfile /etc/caddy/Caddyfile

# Entrypoint builds `vllm serve` from the ENV tunables below. Logic layer — changes rarely,
# so editing a config default (next layer) never invalidates it, and neither busts the base.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Tunable defaults ── LAST layer on purpose: changing config rebuilds only this cheap
# ENV layer. Better still, override at runtime with `-e VAR=...` (host template env) — then
# there's no rebuild at all. Structural recipe flags live in entrypoint.sh, not here.
ENV MODEL=mistralai/Devstral-Small-2-24B-Instruct-2512 \
    REVISION=f2ca762c466d28ab948b7205492ceb3914e73f8a \
    MAX_MODEL_LEN=131072 \
    GPU_MEM_UTIL=0.90 \
    MAX_NUM_BATCHED=16384 \
    IMAGE_LIMIT=4 \
    KV_CACHE_DTYPE=fp8 \
    MM_ENCODER_BACKEND=TORCH_SDPA \
    SPEC_CONFIG= \
    EXTRA_ARGS= \
    METRICS_TOKEN=

# Caddy owns the public port (8000, what Vast maps); vLLM lives on 127.0.0.1:8001.
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
