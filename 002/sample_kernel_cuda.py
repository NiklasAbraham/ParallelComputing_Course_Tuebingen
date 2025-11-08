import ctypes
import os
from typing import Optional

import numpy as np


class FutharkError(RuntimeError):
    """Raised when the underlying Futhark runtime reports an error."""


def _ensure_double_array(array: np.ndarray) -> np.ndarray:
    if array.dtype != np.float64:
        return np.asarray(array, dtype=np.float64)
    return array


class SampleKernelCUDA:
    """Thin ctypes-based wrapper around the CUDA Futhark library."""

    def __init__(self, device: Optional[str] = None) -> None:
        lib_path = os.path.join(os.path.dirname(__file__), "libsample_kernel.so")
        self._lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)

        # Configure ctypes signatures.
        self._lib.futhark_context_config_new.restype = ctypes.c_void_p
        self._lib.futhark_context_new.restype = ctypes.c_void_p
        self._lib.futhark_context_new.argtypes = [ctypes.c_void_p]
        self._lib.futhark_context_config_set_device.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
        )
        self._lib.futhark_context_config_add_nvrtc_option.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
        )
        self._lib.futhark_context_free.argtypes = [ctypes.c_void_p]
        self._lib.futhark_context_free.restype = None
        self._lib.futhark_context_config_free.argtypes = [ctypes.c_void_p]
        self._lib.futhark_context_config_free.restype = None
        self._lib.futhark_context_get_error.argtypes = [ctypes.c_void_p]
        self._lib.futhark_context_get_error.restype = ctypes.c_char_p
        self._lib.futhark_entry_sum_squares.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
        ]
        self._lib.futhark_entry_sum_squares.restype = ctypes.c_int
        self._lib.futhark_context_sync.argtypes = [ctypes.c_void_p]
        self._lib.futhark_context_sync.restype = ctypes.c_int
        self._lib.futhark_new_f64_1d.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
        ]
        self._lib.futhark_new_f64_1d.restype = ctypes.c_void_p
        self._lib.futhark_free_f64_1d.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._lib.futhark_free_f64_1d.restype = ctypes.c_int

        cfg = self._lib.futhark_context_config_new()
        if cfg is None:
            raise FutharkError("Failed to create Futhark context config.")

        if device:
            self._lib.futhark_context_config_set_device(cfg, device.encode("utf-8"))

        nvrtc_arch = os.environ.get("FUTHARK_NVRTC_ARCH", "sm_86")
        if nvrtc_arch:
            option = f"--gpu-architecture={nvrtc_arch}".encode("utf-8")
            self._lib.futhark_context_config_add_nvrtc_option(cfg, option)

        ctx = self._lib.futhark_context_new(cfg)
        if ctx is None:
            raise FutharkError("Failed to create Futhark context.")

        self._cfg = cfg
        self._ctx = ctx

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.futhark_context_free(self._ctx)
            self._ctx = None
        if getattr(self, "_cfg", None):
            self._lib.futhark_context_config_free(self._cfg)
            self._cfg = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _check(self, code: int) -> None:
        if code != 0:
            err_ptr = self._lib.futhark_context_get_error(self._ctx)
            message = err_ptr.decode("utf-8") if err_ptr else f"Futhark error {code}"
            raise FutharkError(message)

    def sum_squares(self, xs: np.ndarray) -> float:
        xs = _ensure_double_array(xs)
        data_ptr = xs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        dev_array = self._lib.futhark_new_f64_1d(
            self._ctx, data_ptr, ctypes.c_int64(xs.shape[0])
        )
        if dev_array is None:
            raise FutharkError("Failed to transfer data to device.")

        try:
            result = ctypes.c_double()
            self._check(self._lib.futhark_entry_sum_squares(self._ctx, result, dev_array))
            self._check(self._lib.futhark_context_sync(self._ctx))
            return float(result.value)
        finally:
            self._check(self._lib.futhark_free_f64_1d(self._ctx, dev_array))


