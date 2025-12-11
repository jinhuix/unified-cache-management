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
"""
DRAM Store with ZMQ-based Scheduler-Worker Coordination

Configuration Example:
    # Scheduler config
    config = {
        "role": "scheduler",
        "enable_coordination": True,
        "scheduler_addr": "tcp://127.0.0.1:5555",
        "zmq_timeout": 1000
    }
    
    # Worker config
    config = {
        "role": "worker",
        "enable_coordination": True,
        "scheduler_addr": "tcp://127.0.0.1:5555",
        "zmq_timeout": 1000
    }

Communication Flow:
    1. Worker dumps KV blocks to local DRAM cache
    2. Worker calls commit() to notify scheduler via ZMQ
    3. Scheduler maintains global cached_blocks registry
    4. Worker calls lookup() to query scheduler for block availability
    5. Worker loads blocks from local DRAM cache if available
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import threading

import torch
import zmq

from ucm.logger import init_logger
from ucm.store.ucmstore import Task, UcmKVStoreBase

logger = init_logger(__name__)

SUCCESS = 0
FAILURE = -1

if torch.cuda.is_available():
    device = torch.cuda
elif hasattr(torch, "musa") and torch.musa.is_available():
    device = torch.musa
elif hasattr(torch, "npu") and torch.npu.is_available():
    device = torch.npu
else:
    raise RuntimeError(
        "No supported accelerator found. "
        "Please ensure either CUDA or NPU is available."
    )


@dataclass
class DramTask(Task):
    task_id: str = "1"
    event: Optional[Any] = None


class DramStoreCoordinator:
    """
    ZMQ-based coordinator for scheduler-worker communication.
    Scheduler acts as server (REP), workers act as clients (REQ).
    """

    def __init__(self, role: str, scheduler_addr: str = "tcp://127.0.0.1:5555", timeout: int = 1000):
        """
        Initialize ZMQ coordinator.

        Args:
            role: "scheduler" or "worker"
            scheduler_addr: ZMQ address for scheduler
            timeout: socket timeout in milliseconds
        """
        self.role = role
        self.scheduler_addr = scheduler_addr
        self.timeout = timeout
        self.context = zmq.Context()
        self.socket = None
        self.cached_blocks = set() if role == "scheduler" else None
        self.lock = threading.Lock()
        self._running = False
        self._server_thread = None

        if role == "scheduler":
            self._start_scheduler_server()
        elif role == "worker":
            self._init_worker_client()
        else:
            raise ValueError(f"Invalid role: {role}, must be 'scheduler' or 'worker'")

    def _start_scheduler_server(self):
        """Start scheduler REP server in background thread."""
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self.scheduler_addr)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self._running = True
        self._server_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._server_thread.start()
        logger.info(f"Scheduler server started at {self.scheduler_addr}")

    def _scheduler_loop(self):
        """Scheduler server loop handling worker requests."""
        while self._running:
            try:
                message = self.socket.recv_string()
                request = json.loads(message)
                response = self._handle_request(request)
                self.socket.send_string(json.dumps(response))
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                try:
                    self.socket.send_string(json.dumps({"status": "error", "message": str(e)}))
                except:
                    pass

    def _handle_request(self, request: Dict) -> Dict:
        """Handle incoming request from worker."""
        msg_type = request.get("type")
        block_ids = request.get("block_ids", [])

        with self.lock:
            if msg_type == "admit":
                self.cached_blocks.update(block_ids)
                logger.debug(f"Admitted blocks: {block_ids}")
                return {"status": "ok", "admitted": len(block_ids)}
            elif msg_type == "evict":
                self.cached_blocks.difference_update(block_ids)
                logger.debug(f"Evicted blocks: {block_ids}")
                return {"status": "ok", "evicted": len(block_ids)}
            elif msg_type == "lookup":
                hits = [bid in self.cached_blocks for bid in block_ids]
                return {"status": "ok", "hits": hits}
            else:
                return {"status": "error", "message": f"Unknown message type: {msg_type}"}

    def _init_worker_client(self):
        """Initialize worker REQ client."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.scheduler_addr)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        logger.info(f"Worker client connected to {self.scheduler_addr}")

    def send_admit(self, block_ids: List[str]) -> bool:
        """Worker: notify scheduler that blocks are cached."""
        if self.role != "worker":
            return True
        return self._send_request({"type": "admit", "block_ids": block_ids})

    def send_evict(self, block_ids: List[str]) -> bool:
        """Worker: notify scheduler that blocks are evicted."""
        if self.role != "worker":
            return True
        return self._send_request({"type": "evict", "block_ids": block_ids})

    def query_lookup(self, block_ids: List[str]) -> List[bool]:
        """Worker: query scheduler for block availability."""
        if self.role != "worker":
            return [False] * len(block_ids)

        response = self._send_request({"type": "lookup", "block_ids": block_ids})
        if response and response.get("status") == "ok":
            return response.get("hits", [False] * len(block_ids))
        return [False] * len(block_ids)

    def _send_request(self, request: Dict) -> Optional[Dict]:
        """Send request to scheduler and get response."""
        try:
            with self.lock:
                self.socket.send_string(json.dumps(request))
                response = self.socket.recv_string()
                return json.loads(response)
        except zmq.Again:
            logger.warning(f"ZMQ timeout for request: {request.get('type')}")
            self._reconnect_worker()
            return None
        except Exception as e:
            logger.error(f"ZMQ error: {e}")
            self._reconnect_worker()
            return None

    def _reconnect_worker(self):
        """Reconnect worker socket after error."""
        try:
            self.socket.close()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect(self.scheduler_addr)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    def close(self):
        """Clean up resources."""
        self._running = False
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2)
        if self.socket:
            self.socket.close()
        self.context.term()
        logger.info(f"{self.role.capitalize()} coordinator closed")


class UcmDramStore(UcmKVStoreBase):
    """
    Dram Connector
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.dram_cache: Dict[str, any] = {}
        self.max_cache_byte = int(config.get("max_cache_size", 5368709120))
        self.kv_block_size = int(config.get("kv_block_size", 262144))
        self.max_block_num = self.max_cache_byte // self.kv_block_size
        self.role = config.get("role", "worker")
        
        # Initialize ZMQ coordinator if enabled
        self.enable_coordination = config.get("enable_coordination", True)
        self.coordinator = None
        if self.enable_coordination:
            scheduler_addr = config.get("scheduler_addr", "tcp://127.0.0.1:5555")
            timeout = config.get("zmq_timeout", 1000)
            try:
                self.coordinator = DramStoreCoordinator(self.role, scheduler_addr, timeout)
            except Exception as e:
                logger.warning(f"Failed to initialize coordinator: {e}, running in standalone mode")
                self.enable_coordination = False
        
        # Legacy: local cached_blocks for scheduler (when coordination disabled)
        if self.role == "scheduler" and not self.enable_coordination:
            self.cached_blocks = set()

    def cc_store(self) -> int:
        """
        get the underlying implementation of Store

        Returns:
            cc pointer to Store
        """
        return 0

    def create(self, block_ids: List[str]) -> List[int]:
        """
        create kv cache space in storage

        Args:
            block_ids (List[str]): vLLM block hash.
        Returns:
            success mask
        """
        return [SUCCESS] * len(block_ids)

    def lookup(self, block_ids: List[str]) -> List[bool]:
        """
        Get number of blocks that can be loaded from the
        external KV cache.

        Args:
            block_ids (List[str]): vLLM block hash.

        Returns:
            hit block mask, True -> hit
        """
        # Worker queries scheduler via ZMQ
        if self.role == "worker" and self.enable_coordination and self.coordinator:
            return self.coordinator.query_lookup(block_ids)
        
        # Scheduler checks local cache (for coordination server)
        if self.role == "scheduler" and self.enable_coordination and self.coordinator:
            with self.coordinator.lock:
                return [block_id in self.coordinator.cached_blocks for block_id in block_ids]
        
        # Legacy mode: check local cached_blocks
        if hasattr(self, "cached_blocks"):
            return [block_id in self.cached_blocks for block_id in block_ids]
        
        return [False] * len(block_ids)

    def prefetch(self, block_ids: List[str]) -> None:
        """
        prefetch kv cache to high speed cache according to block_ids.

        Args:
            block_ids (List[str]): vLLM block hash.
        """
        pass

    def load(
        self, block_ids: List[str], offset: List[int], dst_tensor: List[torch.Tensor]
    ) -> Task:
        """
        load kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            dst_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        task = DramTask()
        stream = device.Stream()
        task.event = device.Event(enable_timing=True)
        with device.stream(stream):
            for i, block_id in enumerate(block_ids):
                key = block_id + "_" + str(offset[i])
                dst_tensor[i].copy_(self.dram_cache[key], non_blocking=True)
            task.event.record(stream=stream)
        logger.debug(f"load block {block_ids} finished.")
        return task

    def dump(
        self, block_ids: List[str], offset: List[int], src_tensor: List[torch.Tensor]
    ) -> Task:
        """
        dump kv cache to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_tensor: List[torch.Tensor]: device tensor addr.
        Returns:
            task(Task).
        """
        task = DramTask()
        if len(self.dram_cache) > self.max_block_num:
            logger.warning(
                "Dram cache usage exceeds limit! No more kv cache offload! Try to increase your initial max_cache_size."
            )
            task.task_id = "-1"
            return task
        else:
            stream = device.Stream()
            task.event = device.Event(enable_timing=True)
            with device.stream(stream):
                for i, block_id in enumerate(block_ids):
                    key = block_id + "_" + str(offset[i])
                    self.dram_cache[key] = src_tensor[i].to("cpu", non_blocking=True)
                task.event.record(stream=stream)
        logger.debug(f"dump block {block_ids} finished.")
        return task

    def fetch_data(
        self,
        block_ids: List[str],
        offset: List[int],
        dst_addr: List[int],
        size: List[int],
    ) -> Task:
        """
        load kv cache data to device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            dst_addr: List[int]: device tensor addr ptr.
            size: List[int]: device tensor size.
        Returns:
            task(Task).
        """
        pass

    def dump_data(
        self,
        block_ids: List[str],
        offset: List[int],
        src_addr: List[int],
        size: List[int],
    ) -> Task:
        """
        dump kv cache data from device.

        Args:
            block_ids (List[str]): vLLM block hash.
            offset(List[int]): tp > 1 scene
            src_addr: List[int]: device tensor addr ptr.
            size: List[int]: device tensor size.
        Returns:
            task(Task).
        """
        pass

    def wait(self, task: DramTask) -> int:
        """
        wait kv cache kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - success
            others - failed.
        """
        if task.task_id == "-1":
            logger.warning("Dump failure with full cache usage!")
            return FAILURE
        try:
            event = task.event
            event.synchronize()
            return SUCCESS
        except Exception as e:
            logger.error(f"Error waiting cache for block IDs: {e}")
            return FAILURE

    def commit(self, block_ids: List[str], is_success: bool = True) -> None:
        """
        commit kv cache, now kv cache can be reused.

        Args:
            block_ids (List[str]): vLLM block hash.
            is_success(bool): if False, we need release block
        """
        if not is_success:
            return
        
        # Worker notifies scheduler via ZMQ
        if self.role == "worker" and self.enable_coordination and self.coordinator:
            self.coordinator.send_admit(block_ids)
            logger.debug(f"Worker committed blocks to scheduler: {block_ids}")
        
        # Legacy mode: update local cached_blocks
        if hasattr(self, "cached_blocks"):
            self.cached_blocks.update(block_ids)

    def check(self, task: Task) -> int:
        """
        check if kv transfer task finished.

        Args:
            task (Task): transfer engine task.
        Returns:
            0 - finished
            others - in process.
        """
        pass
