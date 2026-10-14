#!/bin/sh
set -eu

# Fail clearly instead of spending hours on CPU when a GPU run was requested.
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("NVIDIA GPU unavailable. Run with --gpus all and install the "
             "NVIDIA Container Toolkit and a compatible host driver.")

# Also exercise a kernel: device detection alone does not prove the GPU's
# architecture is supported by this wheel or the driver works with CUDA 12.8.
torch.ones(1, device="cuda").sum().item()
print(f"NVIDIA GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})",
      flush=True)
PY

exec python /app/main.py "$@"
