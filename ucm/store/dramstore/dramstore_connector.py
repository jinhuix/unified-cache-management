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
    stream: torch.cuda.Stream = None
    event: torch.cuda.Event = None
    pinned_buffers: List[torch.Tensor] = field(default_factory=list)
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
        # Local DRAM storage with pin_memory
        self.storage: Dict[str, torch.Tensor] = {}
        self.lock = threading.Lock()
        # ZMQ setup
        self.coordinator = DramStoreCoordinator(self.role, zmq_addr="tcp://127.0.0.1:5555")

    def cc_store(self) -> int:
        return 0

    def create(self, block_ids: List[str]) -> List[int]:
        with self.lock:
            for block_id in block_ids:
                if block_id not in self.storage:
                    self.storage[block_id] = torch.empty(
                        self.kv_block_size, dtype=torch.uint8, pin_memory=True
                    )
        return [0] * len(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        return self.coordinator.lookup(block_ids)

    def prefetch(self, block_ids: List[str]) -> None:
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        stream = torch.cuda.Stream()
        event = torch.cuda.Event()
        task = DramTask(stream=stream, event=event, is_load=True)
        
        buffers = []
        for block_id in block_ids:
            buf = self.storage.get(block_id)
            if buf is None:
                task.status = -1
                return task
            buffers.append(buf)
        
        with torch.cuda.stream(stream):
            for buf, off, dst in zip(buffers, offset, dst_tensor):
                size = dst.numel() * dst.element_size()
                src = buf[off : off + size].view(dst.dtype).reshape(dst.shape)
                dst.copy_(src, non_blocking=True)
            event.record(stream)
        
        return task

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        stream = torch.cuda.Stream()
        event = torch.cuda.Event()
        task = DramTask(
            stream=stream, 
            event=event, 
            is_load=False, 
            block_ids=block_ids, 
            offsets=offset
        )
        
        with torch.cuda.stream(stream):
            for tensor in src_tensor:
                buf = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                buf.copy_(tensor, non_blocking=True)
                task.pinned_buffers.append(buf)
            event.record(stream)
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
        
        if task.is_load or not task.pinned_buffers:
            return 0
        
        with self.lock:
            for block_id, off, pinned_buf in zip(
                task.block_ids, task.offsets, task.pinned_buffers
            ):
                storage_buf = self.storage.get(block_id)
                if storage_buf is None:
                    continue
                
                size = pinned_buf.numel() * pinned_buf.element_size()
                storage_buf[off : off + size].copy_(pinned_buf.flatten().view(torch.uint8))
        
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
