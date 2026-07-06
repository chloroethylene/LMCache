# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU platform helpers."""

# First Party
from lmcache.v1.platform.base_device_info import DeviceInfo

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class NpuDeviceInfo(DeviceInfo):
    """Ascend NPU device information for the detection registry.

    The compiled ops backend (``lmcache_ascend.c_ops``) and all NPU-specific
    transfer optimisation live in the external ``lmcache_ascend`` plugin; this
    entry only wires device detection and backend selection so ``import lmcache``
    loads the plugin's ``c_ops`` on an NPU host.
    """

    @property
    def device_type(self) -> str:
        return "npu"

    @property
    def torch_module_name(self) -> str:
        return "npu"

    @property
    def ops_module(self) -> str | None:
        return "lmcache_ascend.c_ops"

    def is_available(self) -> bool:
        """Check NPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "npu") and torch.npu.is_available()
        except Exception:
            return False
