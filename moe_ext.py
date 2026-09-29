"""JIT build/load of moe_ext.cu (heterogeneous-bitrate EXL3 MoE decode)."""
import os
from torch.utils.cpp_extension import load
_d = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(_d, ".torch_ext"))
import exllamav3
_inc = os.path.join(os.path.dirname(exllamav3.__file__), "exllamav3_ext")
mod = load(name="mimo_moe_ext", sources=[os.path.join(_d, "moe_ext.cu")], extra_include_paths=[_inc],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", "-Xcudafe", "--diag_suppress=177",
                              "-Xcudafe", "--diag_suppress=20012", "-std=c++17"], verbose=False)
