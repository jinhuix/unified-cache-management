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
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple

import torch
from vllm.logger import logger

from ucm.store.gdsstore.cufile_manager import CuFileManager
from ucm.store.ucmstore import Task, UcmKVStoreBase


@dataclass
class GdsTask(Task):
    future: Optional[Future] = None
    status: int = 0


class UcmGdsStore(UcmKVStoreBase):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.root = config.get("gds_path", "/mnt/gds/cache")
        self.kv_block_size = int(config.get("kv_block_size", 0))
        self.use_direct_io = bool(config.get("use_direct_io", False))

        self.cufile_manager = CuFileManager(use_direct_io=self.use_direct_io)
        
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gds_io"
        )
        
        os.makedirs(self.root, exist_ok=True)
        
        logger.info(
            f"GDS Store initialized: path={self.root}, "
            f"cufile_available={self.cufile_manager.is_available()}"
        )

    def _path(self, block_id: str) -> str:
        if len(block_id) < 4:
            block_id = block_id.ljust(4, '0')
        l1, l2 = block_id[:2], block_id[2:4]
        d = os.path.join(self.root, l1, l2)
        return os.path.join(d, f"{block_id}.kvcache")

    def cc_store(self) -> int:
        return 0

    def create(self, block_ids: List[str]) -> List[int]:
        results = []
        file_size = self.kv_block_size
        
        for block_id in block_ids:
            try:
                path = self._path(block_id)
                self.cufile_manager.ensure_file_exists(path, file_size)
                results.append(0)
            except Exception as e:
                logger.error(f"GDS create failed for {block_id}: {e}")
                results.append(-1)
        
        return results

    def lookup(self, block_ids: List[str]) -> List[bool]:
        hits = []
        for bid in block_ids:
            path = self._path(bid)
            try:
                exists = os.path.exists(path)
                hits.append(exists)
            except Exception:
                hits.append(False)
        return hits

    def prefetch(self, block_ids: List[str]) -> None:
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        dst_info = [(t.data_ptr(), t.numel() * t.element_size()) for t in dst_tensor]
        task = GdsTask()

        def _run():
            try:
                if not self.cufile_manager.is_available():
                    task.status = -1
                    logger.error("CuFile not available for load operation")
                    return

                for bid, off, (ptr, size) in zip(block_ids, offset, dst_info):
                    path = self._path(bid)
                    if not os.path.exists(path):
                        raise RuntimeError(f"Block file not found: {path}")
                    
                    file_offset = off
                    ret = self.cufile_manager.read(path, file_offset, ptr, size)
                    
                    if ret != size:
                        raise RuntimeError(
                            f"cuFile read short/error: ret={ret}, expected={size}, path={path}"
                        )
            except Exception as e:
                task.status = -1
                logger.error(f"GDS load failed: {e}")

        task.future = self.executor.submit(_run)
        return task

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        src_info = [(t.data_ptr(), t.numel() * t.element_size()) for t in src_tensor]
        task = GdsTask()

        def _run():
            try:
                if not self.cufile_manager.is_available():
                    task.status = -1
                    logger.error("CuFile not available for dump operation")
                    return

                file_size = self.kv_block_size
                for block_id in block_ids:
                    path = self._path(block_id)
                    self.cufile_manager.ensure_file_exists(path, file_size)

                for bid, off, (ptr, size) in zip(block_ids, offset, src_info):
                    path = self._path(bid)
                    file_offset = off
                    ret = self.cufile_manager.write(path, file_offset, ptr, size)
                    
                    if ret != size:
                        raise RuntimeError(
                            f"cuFile write short/error: ret={ret}, expected={size}, path={path}"
                        )
            except Exception as e:
                task.status = -1
                logger.error(f"GDS dump failed: {e}")

        task.future = self.executor.submit(_run)
        return task

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

    def wait(self, task: Task) -> int:
        if not isinstance(task, GdsTask):
            return -1
        
        if task.status != 0:
            return task.status
        
        if task.future:
            try:
                task.future.result()
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                task.status = -1
        
        return task.status

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        if is_success:
            return
        
        for bid in block_ids:
            try:
                path = self._path(bid)
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to remove block {bid}: {e}")

    def check(self, task: Task) -> Tuple[int, bool]:
        pass
