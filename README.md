# orion-bge-m3

BGE-M3 **sparse embedding** microservice for Orion. It's the *sparse* half of
Orion's hybrid retrieval — dense vectors come from Qwen3-Embedding, sparse
(lexical) vectors come from here — and both are stored in the single
`orion_ai_ops` Milvus collection for hybrid search.

A thin FastAPI wrapper around [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
via `FlagEmbedding`. One job: turn text into `{token_id: weight}` sparse maps.

- **Port:** `9301`
- **Endpoints:** `POST /sparse`, `GET /healthz`
- **Consumed by:** the Knowledge Base (query-time sparse vectors) and the
  Airflow ingest DAG (ingest-time sparse vectors), both via the
  `BGE_M3_URL` env var.

It runs **standalone on a host** (e.g. the LLM VM), not as part of the main
`docker-compose` stack.

---

## API

### `POST /sparse`
```jsonc
// request
{ "inputs": "hello world" }          // or: { "inputs": ["a", "b"] }

// response — one map per input, token_id (string) → weight
{ "results": [ { "33600": 0.21, "9": 0.18, ... } ] }
```

### `GET /healthz`
`200 OK` once the model is loaded.

---

## Setup

### 1. Fetch the model weights (on the host)
The weights are baked into the image at build time — fetch them first so the
build does no network I/O. `~2.2 GB`, resumable.

```bash
cd orion-bge-m3
./scripts/fetch-bge-m3.sh            # downloads BAAI/bge-m3 → ./bge-m3/
# ./scripts/fetch-bge-m3.sh /custom/dir   # optional custom location
```

### 2. Build the image

```bash
# CPU (default — runs anywhere, ~2.5 GB, torch+cpu so no CUDA wheels):
docker build -t orion-bge-m3:cpu .

# NVIDIA GPU (~7 GB; needs nvidia-container-toolkit on the host):
docker build -f Dockerfile.gpu -t orion-bge-m3:gpu .
```

### 3. Run

```bash
# CPU:
docker run -d --name orion-bge-m3 --restart unless-stopped \
  -p 9301:9301 orion-bge-m3:cpu

# NVIDIA GPU (USE_FP16 cuts VRAM + speeds it up):
docker run -d --name orion-bge-m3 --restart unless-stopped \
  --gpus all -e USE_FP16=true -p 9301:9301 orion-bge-m3:gpu
```

### 4. Verify

```bash
curl -fsS localhost:9301/healthz
curl -sS -X POST localhost:9301/sparse \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"hello world"}'
```

---

## Run natively on Apple Silicon (Metal / MPS)

For Mac dev you get **Metal GPU acceleration**, but **only as a native process** —
**not** in Docker. Two independent reasons: (1) the PyTorch MPS backend is
macOS-only and isn't compiled into Linux wheels, and (2) Docker Desktop /
OrbStack run containers in a **Linux VM that has no Metal device** (no GPU
passthrough on macOS). So run it on the host and point the stack at it.

With `DEVICE=auto` the encoder picks `mps` and force-enables `fp16`
(~1.4 GB RAM vs ~2.3 GB fp32, negligible quality loss — the lightest config
that keeps full sparse quality). MLX is **not** an option here: the mature MLX
embedding libs are dense-only, and this service needs BGE-M3's sparse head.

### One-time setup
```bash
brew install python@3.12                        # native arm64 Python
cd orion-bge-m3
./scripts/fetch-bge-m3.sh                        # weights → ./bge-m3/ (~2.2 GB)
python3.12 -m venv .venv
.venv/bin/pip install .                          # torch (arm64, w/ MPS) + FlagEmbedding
```

### Manage it as a service (`bge-m3ctl`)
Installed as a launchd **LaunchAgent** (`~/Library/LaunchAgents/com.orion.bge-m3.plist`)
— a LaunchAgent, not a Daemon, so it runs in your GUI session where Metal is
reachable. The `scripts/bge-m3ctl` wrapper (symlink it onto `PATH`, e.g.
`ln -sf "$PWD/scripts/bge-m3ctl" /opt/homebrew/bin/bge-m3ctl`) gives:

```bash
bge-m3ctl start      # load + start, wait until healthy
bge-m3ctl stop       # fully unload (won't auto-resurrect)
bge-m3ctl restart    # reload the model
bge-m3ctl status     # loaded? pid? health?
bge-m3ctl logs       # tail -f  (~/Library/Logs/orion-bge-m3.log)
```

`RunAtLoad=true` (auto-start at login) and `KeepAlive: Crashed` (restart on
crash). Confirm it's on Metal — the log shows `BGE-M3 loaded on mps`.

### Run it by hand instead (no service)
```bash
cd orion-bge-m3
MODEL_ID="$PWD/bge-m3" DEVICE=auto PYTORCH_ENABLE_MPS_FALLBACK=1 \
  .venv/bin/python -m uvicorn orion_bge_m3.asgi:app --host 0.0.0.0 --port 9301
```

### Wire the Docker stack to the native service
Containers reach the Mac host via `host.docker.internal`, so in
`orion-platform-infra/.env`:
```
BGE_M3_URL=http://host.docker.internal:9301
```

---

## Configuration

Env vars (see `src/orion_bge_m3/config.py`; also reads a `.env` file):

| Var | Default | Notes |
|---|---|---|
| `SERVICE_PORT` | `9301` | HTTP port |
| `MODEL_ID` | `/opt/orion/bge-m3` | Local weights path (baked into the image; a local dir for native runs) |
| `DEVICE` | `auto` | `auto` picks `mps` > `cuda` > `cpu`. Force with `cpu`/`cuda`/`mps`. `mps` only works in a **native macOS process** (containers on Mac have no Metal). |
| `USE_FP16` | `false` | Less memory + faster encode. **Auto-forced `true` on `mps`.** Set `true` yourself on CUDA. |
| `LOG_LEVEL` | `INFO` | |

---

## Wiring into Orion

Point `BGE_M3_URL` in `orion-platform-infra/.env` at this service. Both the
Knowledge Base and the Airflow ingest DAG read it; if it isn't forwarded into
the `airflow-env` block, ingest silently produces dense-only vectors.

```
BGE_M3_URL=http://<host>:9301
```

---

## Architecture

Clean architecture (matches the rest of Orion):

```
src/orion_bge_m3/
  domain/ports/sparse_encoder.py     # SparseEncoder Protocol
  application/encode_sparse.py        # EncodeSparse use case
  infrastructure/bgem3_encoder.py     # FlagEmbedding (BGEM3FlagModel) adapter
  interfaces/http/                    # FastAPI app + routers (embed, health)
  asgi.py                             # uvicorn entrypoint (orion_bge_m3.asgi:app)
```

---

## Hardware notes

- **CPU**: works anywhere. BGE-M3 is small (~560M params), so CPU sparse
  encoding is perfectly usable for normal ingest/query loads.
- **NVIDIA GPU**: use `Dockerfile.gpu` + `--gpus all` + `USE_FP16=true`.
- **Apple Silicon (Metal / MPS)**: `DEVICE=auto` uses the M-series GPU, but
  **only as a native macOS process** — Docker on Mac has no Metal. See
  [Run natively on Apple Silicon](#run-natively-on-apple-silicon-metal--mps).
- **Intel Arc / XPU**: not supported by these images — `FlagEmbedding`/PyTorch
  here have no XPU path. Run the **CPU image** (no Arc acceleration). True Arc
  support would mean adding `intel-extension-for-pytorch` and `device="xpu"` in
  `bgem3_encoder.py`; llama.cpp can't substitute, because BGE-M3's sparse /
  ColBERT heads need `FlagEmbedding`, not GGUF.
