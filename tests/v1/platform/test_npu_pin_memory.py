# SPDX-License-Identifier: Apache-2.0
"""Tests for the NPU host-memory pin backend (no Ascend hardware required)."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_align_helpers():
    # First Party
    from lmcache.v1.platform.npu.pin_memory import _align_down, _align_up

    page = 0x1000
    assert _align_down(0x1001, page) == 0x1000
    assert _align_down(0x2000, page) == 0x2000
    assert _align_up(0x1001, page) == 0x2000
    assert _align_up(0x2000, page) == 0x2000


def test_candidate_lib_paths_honors_ascend_home(monkeypatch):
    # First Party
    from lmcache.v1.platform.npu import pin_memory

    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/ascend/ascend-toolkit")
    paths = pin_memory._candidate_lib_paths()
    assert "/opt/ascend/ascend-toolkit/lib64/libascendcl.so" in paths


def test_graceful_degrade_when_lib_absent(monkeypatch):
    # First Party
    from lmcache.v1.platform.npu.pin_memory import NpuPinMemoryBackend

    monkeypatch.setattr(
        "lmcache.v1.platform.npu.pin_memory._load_libascendcl", lambda: None
    )
    backend = NpuPinMemoryBackend()
    assert backend.is_pin_supported is False
    assert backend.pin_memory(0x1000, 0x1000) is False
    assert backend.unpin_memory(0x1000) is False


def test_pin_unpin_roundtrip_with_fake_lib(monkeypatch):
    # First Party
    from lmcache.v1.platform.npu.pin_memory import NpuPinMemoryBackend

    fake_lib = SimpleNamespace(
        aclrtHostRegister=MagicMock(return_value=0),  # _ACL_SUCCESS
        aclrtHostUnregister=MagicMock(return_value=0),
    )
    monkeypatch.setattr(
        "lmcache.v1.platform.npu.pin_memory._load_libascendcl", lambda: fake_lib
    )
    backend = NpuPinMemoryBackend()
    # Bypass per-thread ACL context setup (no torch_npu in CI).
    backend._tls.ensured = True

    page = 0x1000
    assert backend.pin_memory(0x10, 0x20) is True
    register_args = fake_lib.aclrtHostRegister.call_args.args
    # ctypes represents a NULL pointer (page-aligned base of 0x10) as
    # c_void_p(None); normalize to int 0 so the page-algebra assertions below
    # hold.
    base = register_args[0].value or 0  # ctypes.c_void_p
    reg_size = register_args[1].value  # ctypes.c_uint64
    assert base % page == 0  # page-aligned base
    assert reg_size % page == 0  # page-multiple size
    assert base == 0x0 and reg_size == page

    assert backend.unpin_memory(0x10) is True
    # A pointer that was never pinned unpins as a no-op success.
    assert backend.unpin_memory(0x99999) is True
    unpin_base = fake_lib.aclrtHostUnregister.call_args.args[0].value or 0
    assert unpin_base == base  # original ptr -> registered base


def test_pin_returns_false_on_acl_error(monkeypatch):
    # First Party
    from lmcache.v1.platform.npu.pin_memory import NpuPinMemoryBackend

    fake_lib = SimpleNamespace(
        aclrtHostRegister=MagicMock(return_value=507899),  # AscendCL error
        aclrtHostUnregister=MagicMock(return_value=0),
    )
    monkeypatch.setattr(
        "lmcache.v1.platform.npu.pin_memory._load_libascendcl", lambda: fake_lib
    )
    backend = NpuPinMemoryBackend()
    backend._tls.ensured = True
    assert backend.pin_memory(0x1000, 0x1000) is False
