"""Local CUDA 13.0 declaration compatibility for glibc >=2.41; no kernel math changes."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    prefix = Path(os.environ.get("CUDA_HOME", ROOT/".venv-sglang/lib/python3.12/site-packages/nvidia/cu13")).resolve()
    expected = (ROOT/".venv-sglang/lib/python3.12/site-packages/nvidia/cu13").resolve()
    assert prefix == expected, "Only modify this project's isolated toolkit."
    header = prefix/"include/crt/math_functions.h"
    before = header.read_text()
    after = before
    marker = "// profile_spec: align CUDA rsqrt declarations with glibc IEC60559"
    if marker not in after:
        for name, type_ in [("rsqrt", "double"), ("rsqrtf", "float")]:
            declaration = next(s for s in after.splitlines() if s.startswith("extern __DEVICE_FUNCTIONS_DECL__") and f"{name}({type_} x);" in s)
            replacement = marker+"\n#if __NV_GLIBC_PROVIDES_IEC_60559_FUNCS\n"+declaration[:-1]+" __THROW;\n#else\n"+declaration+"\n#endif"
            after = after.replace(declaration, replacement, 1)
        # uv may hard-link the installed header to its package cache: replace the
        # directory entry, never mutate the shared cache's inode in place.
        temporary = header.with_suffix(".profile_spec.tmp")
        temporary.write_text(after); temporary.replace(header)
    evidence = ROOT/"logs/cuda_glibc_compatibility.json"
    if not evidence.exists():
        evidence.write_text(json.dumps({"header": str(header), "before_sha256": hashlib.sha256(before.encode()).hexdigest(), "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
            "change": "rsqrt/rsqrtf declarations use __THROW when glibc provides the IEC60559 functions; no implementation changed", "reference": "https://forums.developer.nvidia.com/t/cuda-headers-in-crt-math-functions-h-still-broken-in-debian-13-repo/362708"}, indent=2))
    print("Project-local CUDA/glibc declaration compatibility applied.")


if __name__ == "__main__":
    main()
