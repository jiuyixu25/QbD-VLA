#!/usr/bin/env bash
#
#   bash scripts/env/setup_openvla_oft.sh [env_name]
#
# Pins that matter: torch 2.2.0 / torchvision 0.17.0 / triton 2.2.0, the
# moojink transformers fork (bidirectional SDPA needed by OFT's parallel
# decoding), timm 0.9.10 (the remote code hard-fails on other versions),
# tensorflow 2.15 (image resize/crop ops used by the official preprocessing).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ENV="${1:-openvla-oft}"

conda_env "$ENV" 3.10
clone_pinned https://github.com/moojink/openvla-oft.git       openvla-oft e4287e94541f459edc4feabc4e181f537cd569a8
clone_pinned https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO  8f1084e3132a39270c3a13ebe37270a43ece2a01

run "$ENV" python -m pip install -e "$THIRD_PARTY/openvla-oft"
run "$ENV" python -m pip install -r "$THIRD_PARTY/openvla-oft/experiments/robot/libero/libero_requirements.txt"
run "$ENV" python -m pip install --no-deps -e "$THIRD_PARTY/LIBERO"
run "$ENV" python -m pip install "flash-attn==2.5.5" --no-build-isolation || \
  echo "  !! flash-attn not installed: OpenVLA falls back to eager attention (slower, same numerics)"


write_libero_config "$THIRD_PARTY/LIBERO/libero/libero"
install_qbdvla "$ENV"
run "$ENV" python - <<'PY'
import torch, transformers, timm, inspect
from transformers.models.llama import modeling_llama as m
assert torch.__version__.startswith("2.2"), f"torch must stay at 2.2.x, got {torch.__version__}"
assert "is_causal=False" in inspect.getsource(m.LlamaSdpaAttention.forward), "the moojink transformers fork is not installed"
print(f"OK torch {torch.__version__} | transformers {transformers.__version__} (fork) | timm {timm.__version__}")
PY
echo "done: MUJOCO_GL=egl qbdvla run --model openvla_oft --benchmark libero --suite libero_spatial --preset W4"
