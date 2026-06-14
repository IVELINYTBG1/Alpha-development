#!/usr/bin/env bash
# setup_fedora.sh — Alpha v0.5 — Fedora Workstation 44 Setup
# ====================================================================
# Run once after cloning/unzipping the project.
# This script is idempotent — safe to run multiple times.
#
# Usage:
#   chmod +x setup_fedora.sh
#   ./setup_fedora.sh
#
# What it does:
#   1. Install system packages (dnf)
#   2. Install Rust (if not present)
#   3. Install Python deps (pip into user space)
#   4. Download STT model (faster-whisper tiny — ~75MB)
#   5. Set up project env file (.env)
#   6. Create voices/ placeholder README
#   7. Build the Rust binary in release mode
#   8. Print the run command

set -euo pipefail

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[1;33m'
CYN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYN}[INFO]${NC} $*"; }
ok()    { echo -e "${GRN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YLW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR ]${NC} $*"; exit 1; }

echo ""
echo -e "${CYN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYN}║   Alpha v0.5 — Fedora 44 Setup      ║${NC}"
echo -e "${CYN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing system packages via dnf..."
sudo dnf install -y \
    gcc gcc-c++ make \
    python3 python3-devel python3-pip \
    rust cargo \
    alsa-lib-devel \
    portaudio-devel \
    openssl-devel \
    pkg-config \
    libffi-devel \
    cmake \
    git \
    pipewire pipewire-alsa pipewire-pulseaudio \
    v4l-utils \
    2>/dev/null && ok "System packages installed" || warn "Some packages may have failed — continuing"

# ── 2. Rust toolchain ─────────────────────────────────────────────────────────
if ! command -v cargo &>/dev/null; then
    info "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
    ok "Rust installed: $(rustc --version)"
else
    ok "Rust already installed: $(rustc --version)"
fi

# ── 3. Python deps ────────────────────────────────────────────────────────────
info "Installing Python dependencies..."
python3 -m pip install --user --upgrade pip wheel setuptools

# Install in stages to catch errors early
info "  Installing PyTorch (CPU)..."
python3 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

info "  Installing SNN + vision deps..."
python3 -m pip install --user \
    snntorch \
    mediapipe \
    opencv-python \
    numpy \
    psutil

info "  Installing STT deps (faster-whisper)..."
python3 -m pip install --user faster-whisper soundfile

info "  Installing TTS (Coqui XTTS v2)..."
python3 -m pip install --user TTS sounddevice || \
    warn "TTS install failed — voice cloning will be disabled. Try: pip install --user TTS"

ok "Python dependencies installed"

# ── 4. STT model ──────────────────────────────────────────────────────────────
info "Checking faster-whisper tiny model..."
MODEL_DIR="$HOME/.cache/huggingface/hub/models--Systran--faster-whisper-tiny"
if [ -d "$MODEL_DIR" ]; then
    ok "faster-whisper tiny model already cached"
else
    info "Downloading faster-whisper tiny model (~75MB)..."
    python3 -c "
from faster_whisper import WhisperModel
print('Downloading...')
WhisperModel('tiny', device='cpu', compute_type='int8')
print('Done.')
" && ok "faster-whisper tiny model downloaded" || \
    warn "Model download failed — STT will fall back to silent mode"
fi

# ── 5. TTS model ──────────────────────────────────────────────────────────────
info "Checking XTTS v2 model (1.8GB — skipping auto-download)..."
info "  To download manually: python tts_engine.py --download"
warn "  Place voice references in voices/ before running TTS"

# ── 6. Project environment ────────────────────────────────────────────────────
info "Creating .env file..."
PYTHON_PATH=$(which python3)
cat > .env << ENVEOF
# Alpha v0.5 — Environment
# Source this before running: source .env

export PYO3_PYTHON=${PYTHON_PATH}
export PYTHONPATH="$(pwd):$PYTHONPATH"
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)

# CUDA/XPU disabled — CPU-native mode
export CUDA_VISIBLE_DEVICES=""
export XPU_VISIBLE_DEVICES=""
ENVEOF
ok ".env created — run 'source .env' before building"

# ── 7. Voices directory ───────────────────────────────────────────────────────
mkdir -p voices
if [ ! -f voices/alpha_reference.wav ]; then
    info "voices/ created — add voice reference files:"
    echo "    voices/alpha_reference.wav   (10-30s clean speech)"
    echo "    voices/alpha_reference.wav (10-30s clean speech)"
fi

# ── 8. Persistent data directories ───────────────────────────────────────────
mkdir -p logs
touch training_trace.jsonl semantic_memory.json story_log.jsonl brain_log.txt 2>/dev/null || true
ok "Data files initialized"

# ── 9. Microphone permission check ───────────────────────────────────────────
info "Checking microphone access..."
if python3 -c "import sounddevice; sounddevice.query_devices()" &>/dev/null; then
    ok "Microphone accessible"
else
    warn "Microphone check failed."
    warn "On Fedora 44: Settings > Privacy > Microphone > Allow"
    warn "Or run: flatpak permission-set devices-grant microphone"
fi

# ── 10. Camera check ──────────────────────────────────────────────────────────
info "Checking camera..."
if ls /dev/video* &>/dev/null; then
    ok "Camera device found: $(ls /dev/video* | head -1)"
else
    warn "No /dev/video* found — camera features will be disabled"
    warn "Connect a webcam or grant access in Settings > Privacy > Camera"
fi

# ── 11. Build Rust binary ─────────────────────────────────────────────────────
info "Building Alpha (release mode)..."
source .env
cargo build --release 2>&1 | tail -5
ok "Build complete: ./target/release/alpha_core"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GRN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GRN}║              Setup Complete!                 ║${NC}"
echo -e "${GRN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Run Alpha:"
echo -e "  ${CYN}source .env && cargo run --release${NC}"
echo -e "  or"
echo -e "  ${CYN}source .env && ./target/release/alpha_core${NC}"
echo ""
echo -e "First-time voice setup:"
echo -e "  ${CYN}python tts_engine.py --download${NC}   # downloads XTTS v2 (~1.8GB)"
echo -e "  ${CYN}python stt_engine.py --test${NC}       # tests STT backend"
echo ""
echo -e "TUI Controls:"
echo -e "  TAB    = switch TEXT / STT mode"
echo -e "  i      = open text input"
echo -e "  Enter  = send"
echo -e "  Esc    = cancel"
echo -e "  q      = quit"
echo ""
echo -e "In STT mode — say '${CYN}Alpha${NC}' or '${CYN}Alpha${NC}' to wake them."
echo -e "They will recognize you over time. No hardcoding."
echo ""
