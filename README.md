# devstral-vllm — `blackwell` branch

Vision-capable **Devstral Small 2 (24B)** on **vLLM**, tuned for **RTX PRO 6000 Blackwell
(SM 120)**. Sibling of `main` (which targets Ada/H100/H200 on vLLM 0.22.1) — see the
[`Dockerfile`](./Dockerfile) header for exactly what changes and why.

Config is driven entirely by **environment variables** (see [`entrypoint.sh`](./entrypoint.sh)),
so tuning a knob is a runtime `-e VAR=...` — **no image rebuild**.

## ⚠️ The one unproven bit — smoke-test vision first
The two hard blockers (the SM120 CUDA gate, the `is_fast` tokenizer crash) and the
`fetch_images` fix are all verified in 0.24.0. **What is NOT field-proven is the Pixtral
vision encoder running on SM 120** — FlashAttention has no SM120 build, so the ViT is
forced onto `TORCH_SDPA`. The expected failure mode is a CUDA error *in the encoder at
startup*. **Right after launch, POST a real image and confirm a sensible description** —
that single test is the only thing that closes this risk. If it crashes, walk the
[fallback ladder](#fallback-ladder).

## Requirements
- An **RTX PRO 6000 Blackwell** (or other SM 120) GPU, **host driver ≥ 580, CUDA 13**.
- **Bare-metal Linux, not WSL2** (WSL2 emulates FP8 ~3× slower + extra vision-kernel breakage).
- ~40 GB free disk for the FP8 weights (downloaded from Hugging Face on first start).
- A single 96 GB card fits ~24 GB FP8 weights + bf16 vision tower + 131K context + several
  images at `GPU_MEM_UTIL=0.90` — no tensor-parallel needed.

## Build & run
```bash
docker build -t devstral-vllm:blackwell .

docker run --gpus all -p 8000:8000 \
  -e VLLM_API_KEY=<your-api-key> \
  devstral-vllm:blackwell
```
Tune any knob without rebuilding by overriding its env var:
```bash
docker run --gpus all -p 8000:8000 \
  -e VLLM_API_KEY=<key> -e GPU_MEM_UTIL=0.85 -e IMAGE_LIMIT=8 -e MM_ENCODER_BACKEND=XFORMERS \
  devstral-vllm:blackwell
```

## Tunable env vars
| Var | Default | Notes |
|---|---|---|
| `VLLM_API_KEY` | — | Bearer token the server requires for `/v1/*`; clients must match. |
| `METRICS_TOKEN` | = `VLLM_API_KEY` | Bearer token for `/metrics` + `/health` (see [Exposed routes](#exposed-routes)). Defaults to `VLLM_API_KEY`; set separately to split. |
| `PORT` | `8000` | **Public** port Caddy listens on (what Vast maps). vLLM runs internally on `127.0.0.1:8001`. |
| `MODEL` | `mistralai/Devstral-Small-2-24B-Instruct-2512` | |
| `REVISION` | pinned commit | Immutable → every cold start pulls identical weights. |
| `MAX_MODEL_LEN` | `131072` | Size as `text + Σ image_tokens`. A full image ≈ ~3,000 tokens. |
| `GPU_MEM_UTIL` | `0.90` | |
| `MAX_NUM_BATCHED` | `16384` | Must be ≥ one image's token count or images can't be scheduled. |
| `IMAGE_LIMIT` | `4` | Max images per prompt (`view_image`/attachments). Raise to 8–10 if needed. |
| `KV_CACHE_DTYPE` | `fp8` | |
| `MM_ENCODER_BACKEND` | `TORCH_SDPA` | The load-bearing vision flag — see fallback ladder. |
| `SPEC` | (empty) | e.g. `--spec-method ngram --spec-tokens 8`. Off by default (inverts under concurrency). |
| `EXTRA_ARGS` | (empty) | Any extra `vllm serve` flags, no rebuild. |

The structural recipe flags (`--tokenizer-mode/--config-format/--load-format mistral`,
tool-calling, prefix-caching) are fixed in `entrypoint.sh` — they define the image, not a dial.

## Fallback ladder
If startup crashes in the **vision encoder**:
1. `-e MM_ENCODER_BACKEND=XFORMERS`
2. Rebuild flash-attn for Blackwell: `TORCH_CUDA_ARCH_LIST="12.0+PTX"`.
3. Last resort: text-only (`-e EXTRA_ARGS='--limit-mm-per-prompt {"image":0}'`) — **disables
   `view_image`**, so only for isolating whether the encoder is the problem.

If `0.24.0` stable misbehaves on this card, swap the base to the **cu130 nightly**
(`https://wheels.vllm.ai/nightly/cu130`) or NGC `nvcr.io/nvidia/vllm:25.09`.
If anything knocks serving off mistral mode (e.g. an HF-format mirror), the `is_fast` crash
returns — pin `transformers<5` as the backstop.

## Exposed routes
vLLM's `--api-key` only guards `/v1/*` — `/metrics`, `/health`, `/server_info`, `/tokenize`,
`/docs`, … are otherwise **wide open** (and `/server_info` leaks the full launch config). So a
tiny [Caddy](./Caddyfile) reverse proxy fronts vLLM with a **default-deny** policy:

| Route | Reachable | Auth |
|---|---|---|
| `/v1/*` | yes | `VLLM_API_KEY` (vLLM's own) |
| `/metrics`, `/health` | yes | `METRICS_TOKEN` (Bearer; defaults to `VLLM_API_KEY`) |
| everything else | **403** | — |

vLLM binds `127.0.0.1:8001` only; Caddy owns the public `PORT` (8000). **One port** — metrics
and serving share it, split by path — so Vast's single 8000 mapping is all you need. Scrape with:
```bash
curl -H "Authorization: Bearer $METRICS_TOKEN" http://<host>:8000/metrics
```

## Serving-platform notes
- The API server comes up **only after** weights load — set the platform's health-check
  **grace period** high (first download ~10–15 min) or it gets killed mid-download.
- Health at `/health` (200 once loaded, **token required**); OpenAI API at `/v1/chat/completions`.
- No persistent volume required; with `REVISION` pinned, each cold start pulls the same weights.

## Sources
vLLM [#44911](https://github.com/vllm-project/vllm/issues/44911) /
[PR #45180](https://github.com/vllm-project/vllm/pull/45180) (fetch_images fix in 0.24.0),
[#38411](https://github.com/vllm-project/vllm/issues/38411) (MMEncoderAttention on Blackwell),
[#27124](https://github.com/vllm-project/vllm/releases/tag/v0.11.1) (`--mm-encoder-attn-backend`).
