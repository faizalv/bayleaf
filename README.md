# bayleaf

Go library that manages a Python subprocess for local embedding and document conversion. Import it, call `Start()`, get a client back. The Python server boots on first request, shuts down after idle timeout, restarts on next request.

Ships a self-contained Python runtime in a platform-specific tarball. No Python installation required on the host.

## What it does

Two services over a Unix domain socket:

- **Embed** -- `intfloat/e5-base` via ONNX Runtime. 768-dim L2-normalized vectors. Lazy model load on first call (~2s), subsequent calls <50ms.
- **Convert** -- File to markdown via Microsoft's `markitdown`. Reads directly from disk. Supports PDF, DOCX, PPTX, images, etc.

## Why it exists

[Lemongrass](https://github.com/faizalv/lemongrass) ran these services in a 2GB Docker container (`lg-embed`) that sat in memory permanently. Docker is being removed from lemongrass. Bayleaf replaces the container with a ~260MB tarball and a process that only runs when needed.

## Install

```go
import "github.com/faizalv/bayleaf"
```

Pre-download the runtime during install:

```go
bayleaf.EnsureRuntime(ctx, "") // downloads to ~/.bayleaf/cache/
```

## Usage

```go
client, err := bayleaf.Start(ctx, bayleaf.Config{})
defer client.Stop()

// Embedding
vec, err := client.Embed(ctx, "query: what is machine learning")
// vec is []float32, len 768, L2-normalized

// Document conversion
markdown, err := client.Convert(ctx, "/path/to/file.pdf")

// Pre-warm (boot server without blocking)
client.Warm(ctx)
```

## Config

```go
bayleaf.Config{
    CacheDir:    "~/.bayleaf/cache/",  // where runtime tarball is cached
    SocketPath:  "",                    // default: <CacheDir>/bayleaf.sock
    IdleTimeout: 300,                   // seconds, 0 = default (300), -1 = never shut down
    Logger:      nil,                   // default: discard
}
```

## Lifecycle

```
[no server] → first request → [starting] → poll /health → [running] → idle timeout → [stopping] → [no server]
```

Cold-start is two-phase:

1. **Server boot** (<500ms) — Python + uvicorn + FastAPI start, socket accepts connections, `/health` returns 200, `/convert` works. No model loaded.
2. **Model load** (~2s, lazy) — ONNX model loads on first `/embed` call. Subsequent embeds are instant.

Convert requests never pay for model load.

## Crash recovery

- **Stale socket** — Detected via `/health` probe, deleted, server restarts.
- **Process dies mid-request** — Connection error triggers one automatic restart + retry.
- **Concurrent cold-starts** — Mutex ensures only one goroutine starts the server; others wait.

## Development

Requires a local ONNX model for integration tests:

```bash
make export-model     # pip installs build deps (torch, transformers), exports e5-base to models/
make test             # unit tests (no runtime needed)
make test-integration # full integration tests (needs BAYLEAF_TEST_CACHE)
make vet
make build            # build platform tarball (caches downloads in ~/.bayleaf/build-cache/)
make build-clean      # wipe build cache and rebuild from scratch
```

Run integration tests against a local build:

```bash
make build
export BAYLEAF_TEST_CACHE=./dist/extracted
make test-integration
```

## Tarball contents

```
python/          — portable CPython (python-build-standalone)
venv/            — pre-built virtualenv (onnxruntime, tokenizers, numpy, markitdown, fastapi, uvicorn)
models/          — e5-base ONNX model + tokenizer.json
server/main.py   — FastAPI application
```

~260MB compressed. Downloaded once from GitHub Releases on first run.

## Platform support

| Platform | Status |
|----------|--------|
| linux-amd64 | supported |
| linux-arm64 | supported |
| darwin-arm64 | supported |
| windows | not yet |

## Python server endpoints

All communication is over Unix domain socket. These are internal to the library.

```
GET  /health                  → {"status": "ok"}
GET  /model                   → {"model": "intfloat/e5-base"}
POST /embed   {"text": "..."}  → {"embedding": [float...]}
POST /convert {"path": "..."}  → {"markdown": "..."}
```

## License

AGPL-3.0
