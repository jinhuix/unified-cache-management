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
import ctypes
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import zmq
from vllm.logger import logger

from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1


@dataclass
class DramTask(Task):
    stream: Optional[object] = None
    status: int = 0
    done: bool = False
    op: str = ""
    block_ids: List[str] = field(default_factory=list)


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


class UcmDramStore(UcmKVStoreBaseV1):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.role = config.get(
            "role",
            "worker" if config.get("device_id", -1) >= 0 else "scheduler",
        )
        self.kv_block_size = int(config.get("block_size") or config.get("kv_block_size") or 0)
        self.max_cache_size = int(config.get("max_cache_size", 5368709120))  # Default 5GB
        self.tensor_size_list: List[int] = [int(x) for x in (config.get("tensor_size_list") or [])]
        self._layer_buffers: List[Optional[torch.Tensor]] = []
        self.block_num: int = 0

        # ZMQ setup
        self.coordinator = DramStoreCoordinator(self.role, zmq_addr="tcp://127.0.0.1:5555")

        self._trans_stream: Optional[object] = None
        if self.role == "worker":
            self._setup_trans_stream()

        self._pinned_pool: Dict[int, List[torch.Tensor]] = {}
        self._pool_lock = threading.Lock()
        self._preallocate_pinned_pool()

    @staticmethod
    def _key(block_id: bytes) -> str:
        return block_id.hex()

    def _setup_trans_stream(self) -> None:
        device_id = int(self.config.get("device_id", -1))
        if device_id < 0:
            return
        if self.kv_block_size <= 0:
            raise RuntimeError("UcmDramStore requires 'block_size' (> 0) for worker role.")
        from ucm.shared.trans import ucmtrans  # type: ignore

        dev = ucmtrans.Device()
        dev.Setup(device_id)
        self._trans_stream = dev.MakeStream()

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

    def _ensure_layer_buffers(self, n_blocks: int) -> None:
        if not self.tensor_size_list:
            return
        n_segments = len(self.tensor_size_list)
        if n_blocks == self.block_num and len(self._layer_buffers) == n_segments:
            return
        self._layer_buffers.clear()
        self.block_num = n_blocks
        for i in range(n_segments):
            seg_size = n_blocks * self.tensor_size_list[i]
            buf = torch.empty(seg_size, dtype=torch.uint8, pin_memory=True)
            self._layer_buffers.append(buf)

    def cc_store(self) -> int:
        return 0

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        return self.coordinator.lookup([self._key(b) for b in block_ids])

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        res = self.lookup(block_ids)
        for i, hit in enumerate(res):
            if not hit:
                return i - 1
        return len(res) - 1

    def prefetch(self, block_ids: List[bytes]) -> None:
        pass

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        ptrs = np.asarray([[t.data_ptr() for t in row] for row in dst_tensor], dtype=np.uint64)
        return self.load_data(block_ids, shard_index, ptrs)

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        ptrs = np.asarray([[t.data_ptr() for t in row] for row in src_tensor], dtype=np.uint64)
        return self.dump_data(block_ids, shard_index, ptrs)

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]] | np.ndarray,
    ) -> Task:
        keys = [self._key(b) for b in block_ids]
        task = DramTask(stream=self._trans_stream, op="load", block_ids=keys)
        if not self.tensor_size_list:
            task.done = True
            return task

        dev_ptrs = np.asarray(dst_addr, dtype=np.uint64)
        n_blocks = len(keys)
        self._ensure_layer_buffers(n_blocks)
        sizes = np.asarray(self.tensor_size_list, dtype=np.uint64)

        for i in range(len(self.tensor_size_list)):
            size_per_block = int(sizes[i])
            total_size = n_blocks * size_per_block
            host_base_L = np.uintp(self._layer_buffers[i].data_ptr())
            device_base_L = np.uintp(dev_ptrs[i])
            self._trans_stream.HostToDeviceAsync(
                host_base_L, device_base_L, total_size
            )

        return task

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]] | np.ndarray,
    ) -> Task:
        keys = [self._key(b) for b in block_ids]
        task = DramTask(stream=self._trans_stream, op="dump", block_ids=keys)
        if not self.tensor_size_list:
            task.done = True
            return task

        n_blocks = len(keys)
        self._ensure_layer_buffers(n_blocks)
        dev_ptrs = np.asarray(src_addr, dtype=np.uint64)
        sizes = np.asarray(self.tensor_size_list, dtype=np.uint64)

        try:
            # 每层：单次连续拷贝，从 device_base_L 搬移整层 n_blocks*sizes[i] 字节到 host_base_L
            for i in range(len(self.tensor_size_list)):
                size_per_block = int(sizes[i])
                total_size = n_blocks * size_per_block
                host_base_L = self._layer_buffers[i].data_ptr()
                device_base_L = int(dev_ptrs[i])

                self._trans_stream.DeviceToHostAsync(
                    device_base_L, host_base_L, total_size
                )
        except Exception as e:
            task.status = -1
            raise RuntimeError(f"DRAM dump_data failed: {e}") from e
        return task

    def wait(self, task: Task) -> None:
        if task.done:
            return

        if task.status != 0:
            raise RuntimeError("Task is already in a failed state.")

        try:
            if task.stream is not None:
                task.stream.Synchronized()
        finally:
            task.done = True

        if task.op == "dump":
            self.coordinator.admit(task.block_ids)

    def check(self, task: Task) -> bool:
        if not isinstance(task, DramTask):
            return True
        return task.done