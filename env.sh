# Environment for the MiMo-V2.6-Pro EXL3 fast path. Override any of these before sourcing; the defaults are our box.
_d=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH=${MIMO_VENV:-/data/Jarrel/nqenv}/bin:$PATH
export CUDA_HOME=${CUDA_HOME:-/home/jarrelscy/cuda128} TORCH_CUDA_ARCH_LIST=12.0a PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$_d/.triton} HF_HOME=${HF_HOME:-/data/huggingface}
export MIMO_EXL3_DIR=${MIMO_EXL3_DIR:-/data/models/jarrelscy/MiMo-V2.6-Pro-EXL3}
