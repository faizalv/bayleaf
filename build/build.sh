#!/usr/bin/env bash
set -euo pipefail

PYTHON_MINOR="${PYTHON_MINOR:-3.12}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/dist}"

# Persistent cache -- survives between runs so we don't re-download gigabytes.
BUILD_CACHE="${BUILD_CACHE:-${HOME}/.bayleaf/build-cache}"
PBS_CACHE="${BUILD_CACHE}/pbs"
BUILD_VENV_DIR="${BUILD_CACHE}/build-venv"

WORKDIR="$(mktemp -d)"
STAGING="${WORKDIR}/staging"
trap 'rm -rf "$WORKDIR"' EXIT

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64)        ARCH="amd64"; PBS_ARCH="x86_64" ;;
    aarch64|arm64) ARCH="arm64"; PBS_ARCH="aarch64" ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac
PLATFORM="${OS}-${ARCH}"

case "$OS" in
    linux)  PBS_OS="unknown-linux-gnu" ;;
    darwin) PBS_OS="apple-darwin" ;;
    *) echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Building bayleaf runtime for ${PLATFORM}"
echo "    Build cache: ${BUILD_CACHE}"

# --- Portable Python (cached) ---

mkdir -p "${PBS_CACHE}"
PBS_TARBALL="${PBS_CACHE}/python-${PYTHON_MINOR}-${PLATFORM}.tar.gz"

if [ ! -f "${PBS_TARBALL}" ]; then
    echo "==> Resolving python-build-standalone release"
    PBS_API="https://api.github.com/repos/indygreg/python-build-standalone/releases/latest"
    PBS_AUTH_HEADER=""
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        PBS_AUTH_HEADER="Authorization: token ${GITHUB_TOKEN}"
    fi
    PBS_URL="$(curl -fsSL ${PBS_AUTH_HEADER:+-H "${PBS_AUTH_HEADER}"} "${PBS_API}" | python3 -c "
import sys, json, re
assets = json.load(sys.stdin)['assets']
pattern = re.compile(r'cpython-${PYTHON_MINOR}\.\d+\+\d+-${PBS_ARCH}-${PBS_OS}-install_only\.tar\.gz$')
matches = [a['browser_download_url'] for a in assets if pattern.match(a['name'])]
if not matches:
    print('ERROR: no matching asset found', file=sys.stderr)
    sys.exit(1)
print(matches[0])
")"
    echo "==> Downloading $(basename "${PBS_URL}")"
    curl -fSL -o "${PBS_TARBALL}" "${PBS_URL}"
else
    echo "==> Using cached python-build-standalone (${PBS_TARBALL})"
fi

mkdir -p "${STAGING}"
tar xf "${PBS_TARBALL}" -C "${STAGING}"

PYTHON="${STAGING}/python/bin/python3"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: python3 not found at ${PYTHON}" >&2
    exit 1
fi

echo "==> Python: $("${PYTHON}" --version)"

# --- Runtime virtualenv (goes into the tarball) ---

echo "==> Creating runtime virtualenv"
"${PYTHON}" -m venv "${STAGING}/venv"
VENV_PIP="${STAGING}/venv/bin/pip"

echo "==> Installing runtime dependencies"
"${VENV_PIP}" install --no-cache-dir --only-binary :all: \
    onnxruntime \
    tokenizers \
    numpy \
    "markitdown[all]" \
    fastapi \
    "uvicorn[standard]"

# --- Build virtualenv (cached, not shipped in tarball) ---

if [ ! -f "${BUILD_VENV_DIR}/bin/python" ]; then
    echo "==> Creating build virtualenv (cached at ${BUILD_VENV_DIR})"
    python3 -m venv "${BUILD_VENV_DIR}"
    "${BUILD_VENV_DIR}/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch \
        transformers \
        onnxscript \
        onnxruntime \
        tokenizers \
        numpy
else
    echo "==> Using cached build virtualenv"
fi

# --- ONNX model export ---

echo "==> Exporting E5-base to ONNX"
mkdir -p "${STAGING}/models"
"${BUILD_VENV_DIR}/bin/python" "${REPO_DIR}/server/export_onnx.py" "${STAGING}/models"

# --- Server code ---

echo "==> Copying server code"
mkdir -p "${STAGING}/server"
cp "${REPO_DIR}/server/main.py" "${STAGING}/server/"

# --- Strip unnecessary files ---

echo "==> Stripping unnecessary files"
find "${STAGING}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${STAGING}" -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find "${STAGING}" -type d -name "tests" -path "*/site-packages/*" -exec rm -rf {} + 2>/dev/null || true
find "${STAGING}" -type d -name "test" -path "*/site-packages/*" -exec rm -rf {} + 2>/dev/null || true
find "${STAGING}" -type d -name "docs" -path "*/site-packages/*" -exec rm -rf {} + 2>/dev/null || true

# --- Package ---

echo "==> Creating tarball"
mkdir -p "${OUTPUT_DIR}"
TARBALL="${OUTPUT_DIR}/${PLATFORM}.tar.gz"

cd "${STAGING}"
tar czf "${TARBALL}" .

if command -v sha256sum &>/dev/null; then
    sha256sum "${TARBALL}" | awk '{print $1}' > "${TARBALL}.sha256"
else
    shasum -a 256 "${TARBALL}" | awk '{print $1}' > "${TARBALL}.sha256"
fi

echo "==> Done: ${TARBALL}"
echo "    Size: $(du -h "${TARBALL}" | awk '{print $1}')"
echo "    SHA256: $(cat "${TARBALL}.sha256")"
