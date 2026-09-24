#!/usr/bin/env bash
# One-shot environment setup with uv.
#   bash setup.sh           -> pick the PyTorch build from nvidia-smi (CUDA 13.0 / 12.8 / 12.6, else CPU)
#   bash setup.sh cpu       -> force a variant: cpu | cu126 | cu128 | cu130
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv が見つかりません。公式手順でインストールしてください: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

variant="${1:-auto}"
if [ "$variant" = "auto" ]; then
    variant="cpu"
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\.\([0-9]*\).*/\1 \2/p' | head -1)
        major=${cuda% *}
        minor=${cuda#* }
        if [ -n "$cuda" ]; then
            code=$((major * 10 + minor))
            if [ "$code" -ge 130 ]; then variant="cu130"
            elif [ "$code" -ge 128 ]; then variant="cu128"
            elif [ "$code" -ge 120 ]; then variant="cu126"
            fi
        fi
    fi
fi
case "$variant" in
    cpu|cu126|cu128|cu130) ;;
    *) echo "unknown variant: $variant (cpu | cu126 | cu128 | cu130)" >&2; exit 1 ;;
esac

echo "==> PyTorch variant: $variant"
uv sync --extra "$variant"
uv run python - <<'EOF'
import cv2, numpy, torch
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
print(f"torch {torch.__version__} ({dev}) | opencv {cv2.__version__} | numpy {numpy.__version__}")
EOF
echo "==> ready. e.g.  uv run pytest  /  uv run scripts/demo.py --agent expert --show"
