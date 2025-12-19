#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import threading
from typing import Dict, List, Tuple
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from vllm.logger import logger

from ucm.store.ucmstore import Task, UcmKVStoreBase


class CpuTask(Task):
    def __init__(self, is_load: bool = True, future: Future | None = None, status: int = 0):
        self.is_load = is_load
        self.future = future
        self.status = status

class UcmCpuStore(UcmKVStoreBase):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.kv_block_size = config.get("kv_block_size", 0)
        self.max_cache_size = config.get("max_cache_size", 5 * 1024**3)

        self.root = config.get("root", "/home/xujinhui/test_backend")
        os.makedirs(self.root, exist_ok=True)

        self.lock = threading.Lock()
        self.blocks: Dict[str, str] = {}  # block_id -> file path

        # CPU async I/O
        self.executor = ThreadPoolExecutor(max_workers=int(config.get("io_num_workers", 4)))

    def _path(self, block_id: str) -> str:
        return os.path.join(self.root, f"{block_id}.pt")

    def _ensure_paths(self, block_ids: List[str]) -> Dict[str, str]:
        with self.lock:
            for bid in block_ids:
                if bid not in self.blocks:
                    self.blocks[bid] = self._path(bid)
            return {bid: self.blocks[bid] for bid in block_ids}

    def _safe_load_block(self, path: str) -> torch.Tensor | None:
        try:
            if not os.path.exists(path):
                return None
            if os.path.getsize(path) == 0:
                return None
            return torch.load(path, map_location="cpu")
        except (EOFError, RuntimeError, OSError) as e:
            logger.warning(f"Failed to load block file {path}: {e}")
            return None

    def _atomic_save_block(self, buf: torch.Tensor, path: str) -> None:
        tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
        torch.save(buf, tmp)
        os.replace(tmp, path)

    def create(self, block_ids: List[str]) -> List[int]:
        with self.lock:
            for bid in block_ids:
                if bid not in self.blocks:
                    self.blocks[bid] = self._path(bid)
        return [0] * len(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        paths = self._ensure_paths(block_ids)
        hits: List[bool] = []
        for bid in block_ids:
            path = paths[bid]
            hits.append(os.path.exists(path) and os.path.getsize(path) > 0)
        return hits

    def load(
        self,
        block_ids: List[str],
        offset: List[int],
        dst_tensor: List[torch.Tensor],
    ) -> Task:
        task = CpuTask(is_load=True)
        paths = self._ensure_paths(block_ids)

        def _load():
            try:
                reqs: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
                for bid, off, dst in zip(block_ids, offset, dst_tensor):
                    reqs.setdefault(bid, []).append((off, dst))

                for bid, items in reqs.items():
                    path = paths[bid]
                    buf = self._safe_load_block(path)
                    if buf is None:
                        raise RuntimeError(f"Block {bid} not found or corrupted: {path}")

                    for off, dst in items:
                        size = dst.numel() * dst.element_size()
                        src = buf[off : off + size].view(dst.dtype).reshape(dst.shape)
                        dst.copy_(src, non_blocking=False)
            except Exception as e:
                task.status = -1
                logger.error(f"CPU store load failed: {e}")

        task.future = self.executor.submit(_load)
        return task

    def dump(
        self,
        block_ids: List[str],
        offset: List[int],
        src_tensor: List[torch.Tensor],
    ) -> Task:
        task = CpuTask(is_load=False)
        paths = self._ensure_paths(block_ids)

        def _dump():
            try:
                updates: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
                for bid, off, src in zip(block_ids, offset, src_tensor):
                    flat = src.detach().cpu().view(torch.uint8).reshape(-1)
                    updates.setdefault(bid, []).append((off, flat))

                for bid, items in updates.items():
                    path = paths[bid]
                    buf = self._safe_load_block(path)
                    if buf is None:
                        buf = torch.empty(self.kv_block_size, dtype=torch.uint8)

                    for off, flat in items:
                        size = flat.numel()
                        buf[off : off + size].copy_(flat)

                    self._atomic_save_block(buf, path)
            except Exception as e:
                task.status = -1
                logger.error(f"CPU store dump failed: {e}")

        task.future = self.executor.submit(_dump)
        return task

    def wait(self, task: Task) -> int:
        if not isinstance(task, CpuTask):
            return -1
        if task.future:
            try:
                task.future.result()
            except Exception as e:
                task.status = -1 if getattr(task, "status", 0) == 0 else task.status
                logger.error(f"CPU store async task failed: {e}")
        return getattr(task, "status", 0)

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        if not is_success:
            with self.lock:
                for bid in block_ids:
                    path = self.blocks.pop(bid, None)
                    if path and os.path.exists(path):
                        os.remove(path)

    def cc_store(self) -> int:
        return 0

    def prefetch(self, block_ids: List[str]) -> None:
        pass

    def fetch_data(
        self,
        block_ids: List[str],
        offset: List[int],
        dst_addr: List[int],
        size: List[int],
    ) -> Task:
        pass

    def dump_data(
        self,
        block_ids: List[str],
        offset: List[int],
        src_addr: List[int],
        size: List[int],
    ) -> Task:
        pass

    def check(self, task: Task) -> Tuple[int, bool]:
        pass
