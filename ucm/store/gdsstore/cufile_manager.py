import ctypes
import os
from typing import Optional

from vllm.logger import logger


class CuFileManager:
    def __init__(self, use_direct_io: bool = False):
        self.use_direct_io = use_direct_io
        self.cufile = None
        self._driver = None
        self._initialized = False
        self._init_cufile()

    def _init_cufile(self):
        try:
            import cufile
            self.cufile = cufile
            self._driver = cufile.CuFileDriver()
            self._initialized = True
            logger.info("CuFile initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize cufile: {e}, will fallback to standard I/O")
            self._initialized = False

    def is_available(self) -> bool:
        return self._initialized

    def read(
        self, path: str, file_offset: int, gpu_ptr: int, nbytes: int
    ) -> int:
        if not self._initialized:
            raise RuntimeError("CuFile not initialized")

        try:
            with self.cufile.CuFile(path, "r", use_direct_io=self.use_direct_io) as f:
                return f.read(
                    ctypes.c_void_p(gpu_ptr),
                    nbytes,
                    file_offset=file_offset,
                    dev_offset=0,
                )
        except Exception as e:
            logger.error(f"CuFile read failed for {path}: {e}")
            raise

    def write(
        self, path: str, file_offset: int, gpu_ptr: int, nbytes: int
    ) -> int:
        if not self._initialized:
            raise RuntimeError("CuFile not initialized")

        try:
            with self.cufile.CuFile(path, "r+", use_direct_io=self.use_direct_io) as f:
                return f.write(
                    ctypes.c_void_p(gpu_ptr),
                    nbytes,
                    file_offset=file_offset,
                    dev_offset=0,
                )
        except Exception as e:
            logger.error(f"CuFile write failed for {path}: {e}")
            raise

    def ensure_file_exists(self, path: str, size: int) -> None:
        if os.path.exists(path):
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        tmp_path = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "wb") as f:
                f.truncate(size)
            os.replace(tmp_path, path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise e

