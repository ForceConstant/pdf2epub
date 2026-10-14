"""Verify CUDA and Pascal kernels on CPU-only build runners before publishing."""
import torch

assert torch.version.cuda == "12.6", torch.version.cuda
# get_arch_list() returns [] without a GPU, so inspect the compiled-in list.
# NVIDIA supports cubins from a lower minor revision on the same major version:
# sm_60 kernels therefore cover the GTX 1060's sm_61.
architectures = torch._C._cuda_getArchFlags().split()
assert "sm_60" in architectures, f"Pascal kernels missing: {architectures}"
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}: {architectures}")
