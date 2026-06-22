#!/usr/bin/env bash
set -euo pipefail

PYTHON_BUILD_STANDALONE_TAG="${PYTHON_BUILD_STANDALONE_TAG:-20250612}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/dist}"

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

# --- Portable Python ---

PBS_URL="https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_BUILD_STANDALONE_TAG}/cpython-${PYTHON_VERSION}+${PYTHON_BUILD_STANDALONE_TAG}-${PBS_ARCH}-${PBS_OS}-install_only.tar.gz"

echo "==> Downloading python-build-standalone"
curl -fSL -o "${WORKDIR}/python.tar.gz" "${PBS_URL}"

mkdir -p "${STAGING}"
tar xf "${WORKDIR}/python.tar.gz" -C "${STAGING}"

PYTHON="${STAGING}/python/bin/python3"
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: python3 not found at ${PYTHON}" >&2
    exit 1
fi

echo "==> Python: $("${PYTHON}" --version)"

# --- Virtualenv + dependencies ---

echo "==> Creating virtualenv"
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

# --- ONNX model export ---

echo "==> Exporting E5-base to ONNX"
BUILD_VENV="${WORKDIR}/build-venv"
"${PYTHON}" -m venv "${BUILD_VENV}"
"${BUILD_VENV}/bin/pip" install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    "optimum[exporters]" \
    sentence-transformers

mkdir -p "${STAGING}/models"
"${BUILD_VENV}/bin/python" "${REPO_DIR}/server/export_onnx.py" "${STAGING}/models"

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
