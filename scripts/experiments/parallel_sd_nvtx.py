"""Opt-in profile-only NVTX emission for SGLang's existing named step spans.

The normal performance arms never import this module. It does not alter model
operators or mutate upstream source. Spawned scheduler processes inherit the opt-in.
"""
import _bootstrap
from functools import wraps


def enable():
    from sglang.srt.utils import nvtx_utils
    if getattr(nvtx_utils.profile_range, "_profile_spec_nvtx", False):
        return
    original = nvtx_utils.profile_range

    @wraps(original)
    def emit(*args, **kwargs):
        kwargs.setdefault("nvtx_enabled", True)
        return original(*args, **kwargs)

    emit._profile_spec_nvtx = True
    nvtx_utils.profile_range = emit
