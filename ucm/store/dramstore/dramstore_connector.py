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
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import zmq
from vllm.logger import logger

from ucm.store.ucmstore import Task, UcmKVStoreBase


@dataclass
class DramTask(Task):
    stream: Optional[torch.cuda.Stream] = None
    is_load: bool = True
    status: int = 0
    block_ids: List[str] = field(default_factory=list)
    offsets: List[int] = field(default_factory=list)


class DramStoreCoordinator:
    """ZMQ coordinator for scheduler-worker metadata sync"""

    def __init__(self, role: str, zmq_addr: str = "tcp://127.0.0.1:5555"):
        self.role = role
        self.context = zmq.Context()
        self.cached_blocks = set()
        self.lock = threading.Lock()

        if role == "scheduler":
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(zmq_addr)
            self._running = True
            threading.Thread(target=self._serve, daemon=True).start()
            logger.info(f"DramStore scheduler listening on {zmq_addr}")
        else:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(zmq_addr)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)

    def _serve(self):
        while self._running:
            try:
                msg = json.loads(self.socket.recv_string())
                op, block_ids = msg["op"], msg["block_ids"]
                
                with self.lock:
                    if op == "admit":
                        self.cached_blocks.update(block_ids)
                    elif op == "evict":
                        self.cached_blocks.difference_update(block_ids)
                    elif op == "lookup":
                        hits = [bid in self.cached_blocks for bid in block_ids]
                        self.socket.send_string(json.dumps({"hits": hits}))
                        continue
                
                self.socket.send_string(json.dumps({"ok": True}))
            except:
                break

    def _request(self, op: str, block_ids: List[str]) -> Optional[Dict]:
        try:
            self.socket.send_string(json.dumps({"op": op, "block_ids": block_ids}))
            return json.loads(self.socket.recv_string())
        except:
            return None

    def lookup(self, block_ids: List[str]) -> List[bool]:
        if self.role == "scheduler":
            with self.lock:
                return [bid in self.cached_blocks for bid in block_ids]
        else:
            resp = self._request("lookup", block_ids)
            return resp.get("hits", []) if resp else [False] * len(block_ids)

    def admit(self, block_ids: List[str]):
        if self.role == "worker":
            self._request("admit", block_ids)

    def evict(self, block_ids: List[str]):
        if self.role == "worker":
            self._request("evict", block_ids)

class UcmDramStore(UcmKVStoreBase):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.role = config.get("role", "worker")
        self.kv_block_size = config.get("kv_block_size", 0)
        self.max_cache_size = config.get("max_cache_size", 5368709120)  # Default 5GB
        
        # Local DRAM storage with pin_memory
        self.storage: Dict[str, torch.Tensor] = {}
        self.lock = threading.Lock()
        # ZMQ setup
        self.coordinator = DramStoreCoordinator(self.role, zmq_addr="tcp://127.0.0.1:5555")
        
        self._load_stream: Optional[torch.cuda.Stream] = None
        self._dump_stream: Optional[torch.cuda.Stream] = None
        
        self._pinned_pool: Dict[int, List[torch.Tensor]] = {}
        self._pool_lock = threading.Lock()
        self._preallocate_pinned_pool()
    
    def _preallocate_pinned_pool(self):
        if self.kv_block_size > 0 and self.max_cache_size > 0:
            num_blocks = (self.max_cache_size // self.kv_block_size) + 16

            buffers = []
            for i in range(num_blocks):
                try:
                    buf = torch.empty(self.kv_block_size, dtype=torch.uint8, pin_memory=True)
                    buffers.append(buf)
                except RuntimeError as e:
                    logger.warning(f"Failed to allocate pinned buffer {i}/{num_blocks}: {e}")
                    break
            
            self._pinned_pool[self.kv_block_size] = buffers
    
    def _get_pinned(self, num_bytes: int, count: int) -> List[torch.Tensor]:
        buffers = []
        with self._pool_lock:
            pool = self._pinned_pool.get(num_bytes, [])
            available = min(count, len(pool))
            for _ in range(available):
                buffers.append(pool.pop())
        
        remaining = count - len(buffers)
        if remaining > 0:
            for _ in range(remaining):
                buffers.append(torch.empty(num_bytes, dtype=torch.uint8, pin_memory=True))
        
        return buffers

    def _put_pinned(self, buf: torch.Tensor) -> None:
        if buf is None:
            return
        size = buf.numel()
        with self._pool_lock:
            pool = self._pinned_pool.get(size)
            if pool is not None:
                pool.append(buf)
            else:
                if size == self.kv_block_size:
                    self._pinned_pool[size] = [buf]
                elif len(self._pinned_pool.get(size, [])) < 8:
                    self._pinned_pool.setdefault(size, []).append(buf)

    def cc_store(self) -> int:
        return 0

    def create(self, block_ids: List[str]) -> List[int]:
        new_blocks = []
        with self.lock:
            for block_id in block_ids:
                if block_id not in self.storage:
                    new_blocks.append(block_id)
        
        if new_blocks:
            buffers = self._get_pinned(self.kv_block_size, len(new_blocks))
            
            with self.lock:
                for block_id, buf in zip(new_blocks, buffers):
                    if block_id not in self.storage:
                        self.storage[block_id] = buf
                    else:
                        self._put_pinned(buf)
        
        return [0] * len(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        return self.coordinator.lookup(block_ids)

    def prefetch(self, block_ids: List[str]) -> None:
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        task = DramTask(is_load=True)
        
        buffers = []
        for block_id in block_ids:
            buf = self.storage.get(block_id)
            if buf is None:
                task.status = -1
                return task
            buffers.append(buf)
        
        if self._load_stream is None:
            self._load_stream = torch.cuda.Stream()
            logger.debug("Created dedicated load stream for async H2D transfer")
        task.stream = self._load_stream
        
        with torch.cuda.stream(self._load_stream):
            for buf, off, dst in zip(buffers, offset, dst_tensor):
                size = dst.numel() * dst.element_size()
                src = buf[off : off + size].view(dst.dtype).reshape(dst.shape)
                dst.copy_(src, non_blocking=True)
        
        return task

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        task = DramTask(is_load=False, block_ids=block_ids, offsets=offset)

        if len(src_tensor) == 0:
            return task

        if self._dump_stream is None:
            self._dump_stream = torch.cuda.Stream()
            logger.debug("Created dedicated dump stream for async D2H transfer")
        task.stream = self._dump_stream

        current_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._dump_stream):
            self._dump_stream.wait_stream(current_stream)
            
            for bid, off, src in zip(block_ids, offset, src_tensor):
                storage_buf = self.storage.get(bid)
                if storage_buf is None:
                    logger.error(f"Block {bid} not found in storage during dump")
                    task.status = -1
                    return task
                
                size = src.numel() * src.element_size()
                dst = storage_buf[off : off + size].view(src.dtype).reshape(src.shape)
                dst.copy_(src, non_blocking=True)

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
        if not isinstance(task, DramTask) or task.status != 0:
            return -1 if not isinstance(task, DramTask) else task.status
        
        if task.stream:
            task.stream.synchronize()
        
        return 0

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        if is_success:
            self.coordinator.admit(block_ids)
        else:
            with self.lock:
                for block_id in block_ids:
                    self.storage.pop(block_id, None)
            self.coordinator.evict(block_ids)

    def check(self, task: Task) -> Tuple[int, bool]:
        pass
