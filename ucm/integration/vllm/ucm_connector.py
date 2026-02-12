import hashlib
import itertools
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tp_group, get_world_group
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import SchedulerOutput

from ucm.logger import init_logger
from ucm.shared.metrics import ucmmonitor
from ucm.shared.metrics.observability import UCMStatsLogger
from ucm.store.factory import UcmConnectorFactory
from ucm.store.ucmstore import Task, UcmKVStoreBase
from ucm.utils import Config

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class RequestMeta:
    ucm_block_ids_by_group: list[list[str]] = field(default_factory=list)
    hbm_hit_block_num_by_group: list[int] = field(default_factory=list)
    total_hit_block_num_by_group: list[int] = field(default_factory=list)
    vllm_block_ids_by_group: list[list[int]] = field(default_factory=list)
    num_token_ids: int = 0
    token_processed: int = 0

    @property
    def ucm_block_ids(self) -> list[str]:
        return self.ucm_block_ids_by_group[0] if self.ucm_block_ids_by_group else []

    @property
    def hbm_hit_block_num(self) -> int:
        return self.hbm_hit_block_num_by_group[0] if self.hbm_hit_block_num_by_group else 0

    @property
    def total_hit_block_num(self) -> int:
        return self.total_hit_block_num_by_group[0] if self.total_hit_block_num_by_group else 0

    @property
    def vllm_block_ids(self) -> list[int]:
        return self.vllm_block_ids_by_group[0] if self.vllm_block_ids_by_group else []


@dataclass
class RequestDispatchMeta:
    load_block_ids_by_group: list[tuple[list[str], list[int]]] = field(default_factory=list)
    dump_block_ids_by_group: list[tuple[list[str], list[int]]] = field(default_factory=list)

    @property
    def load_block_ids(self) -> tuple[list[str], list[int]]:
        return self.load_block_ids_by_group[0] if self.load_block_ids_by_group else ([], [])

    @property
    def dump_block_ids(self) -> tuple[list[str], list[int]]:
        return self.dump_block_ids_by_group[0] if self.dump_block_ids_by_group else ([], [])


@dataclass
class UCMConnectorMetadata(KVConnectorMetadata):
    request_meta: dict[str, RequestDispatchMeta] = field(default_factory=dict)


class RequestHasher:
    """hash(md5) request to generate ucm block id"""

    _SEED_HASH = None

    def __init__(self, vllm_config, rank_id):
        meta = f"{vllm_config.model_config.model}:{vllm_config.parallel_config.world_size}:{vllm_config.model_config.dtype}:{rank_id}"
        self.meta_bytes = meta.encode("utf-8")

        if RequestHasher._SEED_HASH is None:
            RequestHasher._SEED_HASH = self("UCM_HASH_SEED")

    def __call__(self, input_data) -> int:
        if isinstance(input_data, str):
            input_bytes = input_data.encode("utf-8")
        else:
            input_bytes = pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL)

        h = hashlib.md5(self.meta_bytes + input_bytes)
        return int.from_bytes(h.digest(), byteorder="big")


class UCMDirectConnector(KVConnectorBase_V1, SupportsHMA):
    """
    This connector means synchronize:
    load -> forward -> save
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self.kv_caches: dict[str, Any] = {}
        self.local_rank = (
            -1 if role == KVConnectorRole.SCHEDULER else get_world_group().local_rank
        )
        self.global_rank = self._vllm_config.parallel_config.rank
        self.block_size = self._vllm_config.cache_config.block_size
        self.is_mla = self._vllm_config.model_config.is_deepseek_mla
        self.is_dsa = False
        self.kv_cache_dtype: torch.dtype = None
        
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 1:
            self._num_groups = len(kv_cache_config.kv_cache_groups)
            self._block_sizes = [
                g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups
            ]
            self._group_layer_names = [
                list(g.layer_names) for g in kv_cache_config.kv_cache_groups
            ]
        else:
            self._num_groups = 1
            self._block_sizes = [self.block_size]
            self._group_layer_names = None

        if current_platform.is_cuda_alike():
            logger.info("CUDA device is available.")
            torch_dev = torch
            dev_name = "cuda"
        elif current_platform.device_type == "npu":
            logger.info("NPU device is available.")
            torch_dev = torch.npu
            dev_name = "npu"
        else:
            raise RuntimeError("Unsupported device platform for UCMDirectConnector.")

        if self.local_rank >= 0:
            self.device = torch_dev.device(f"{dev_name}:{self.local_rank}")
            self._layer_offset_cache = {}

        self.store: UcmKVStoreBase

        if role == KVConnectorRole.SCHEDULER:
            self.request_hasher = RequestHasher(vllm_config, 0)
        else:
            self.request_hasher = RequestHasher(vllm_config, self.global_rank)

        # save block info, avoid hash request twice, and track them until request finished
        self.requests_meta: dict[str, RequestMeta] = {}

        ucm_config = Config(vllm_config.kv_transfer_config)
        self.launch_config = ucm_config.get_config()

        self.load_only_first_rank: bool = (
            self.launch_config.get("load_only_first_rank", self.is_mla) and self.is_mla
        )
        if self.load_only_first_rank:
            if role == KVConnectorRole.WORKER:
                self.group_coordinator = get_tp_group()
                self.broadcast_fn = self.group_coordinator.broadcast
                self.broadcast_stream = torch.cuda.Stream()

        logger.info(f"self.launch_config: {self.launch_config}")
        connector_configs = self.launch_config.get("ucm_connectors", [])
        assert len(connector_configs) > 0, "no storage connector name in config."

        name = connector_configs[0].get("ucm_connector_name")
        config = connector_configs[0].get("ucm_connector_config") or {}
        config["device"] = self.local_rank
        config["role"] = "scheduler" if role == KVConnectorRole.SCHEDULER else "worker"
        element_size = vllm_config.model_config.dtype.itemsize
        single_head_dim = vllm_config.model_config.get_head_size()
        num_head_per_tp = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        total_tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        block_size_per_layer = self.block_size * element_size * single_head_dim
        config["kv_block_size"] = (
            block_size_per_layer
            * num_layers
            * (1 if self.is_mla else num_head_per_tp * 2)
        )
        config["io_size"] = block_size_per_layer * (
            1 if self.is_mla else num_head_per_tp
        )
        self._store_name = name
        self._store_config = dict(config)
        self.store = UcmConnectorFactory.create_connector(name, config)
        self.block_data_size = config["kv_block_size"]

        logger.info("init UCConnectorImpl, connector: %s", name)
        logger.info(
            "single file size = %d MB, io_size = %d KB,",
            config["kv_block_size"] / 1024 / 1024,
            config["io_size"] / 1024,
        )

        self.metrics_config = self.launch_config.get("metrics_config_path", "")
        if self.metrics_config:
            self.stats_logger = UCMStatsLogger(
                vllm_config.model_config.served_model_name,
                self.global_rank,
                self.metrics_config,
            )
            self.monitor = ucmmonitor.StatsMonitor.get_instance()

        self.synchronize = (
            torch.cuda.synchronize
            if current_platform.is_cuda_alike()
            else torch.npu.synchronize
        )

        # invlalid block ids due to load errors
        self._invalid_block_ids: set[int] = set()

    def generate_hash(self, block_size: int, request: "Request") -> list[str]:
        token_ids = request.all_token_ids

        ret = []
        parent_block_hash_value = RequestHasher._SEED_HASH
        for start in range(0, len(token_ids), block_size):
            end = start + block_size
            block_token_ids = token_ids[start:end]
            # Do not hash the block if it is not full.
            if len(block_token_ids) < block_size:
                break

            block_token_ids_tuple = tuple(block_token_ids)
            hash_value = self.request_hasher(
                (parent_block_hash_value, block_token_ids_tuple)
            )
            parent_block_hash_value = hash_value
            ret.append(str(hash_value))

        return ret

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        ucm_block_ids_by_group: list[list[str]] = []
        hbm_hit_block_num_by_group: list[int] = []
        total_hit_block_num_by_group: list[int] = []
        external_hit_tokens_by_group: list[int] = []

        for g in range(self._num_groups):
            block_size_g = self._block_sizes[g]
            hbm_hit_block_num_g = num_computed_tokens // block_size_g
            ucm_block_ids_g = self.generate_hash(block_size_g, request)
            
            ucm_block_ids_by_group.append(ucm_block_ids_g)
            hbm_hit_block_num_by_group.append(hbm_hit_block_num_g)

            external_block_ids_g = ucm_block_ids_g[hbm_hit_block_num_g:]
            if not external_block_ids_g:
                external_hit_tokens_by_group.append(0)
                total_hit_block_num_by_group.append(hbm_hit_block_num_g)
                continue

            lookup_results = self.store.lookup(external_block_ids_g)
            external_hit_blocks_g = sum(1 for hit in itertools.takewhile(lambda x: x, lookup_results))
            
            total_hit_block_num_g = hbm_hit_block_num_g + external_hit_blocks_g
            total_hit_block_num_by_group.append(total_hit_block_num_g)
            external_hit_tokens_by_group.append(external_hit_blocks_g * block_size_g)

        external_hit_tokens = min(external_hit_tokens_by_group) if external_hit_tokens_by_group else 0
        num_total_hit_tokens = min(
            total_hit_block_num_by_group[g] * self._block_sizes[g]
            for g in range(self._num_groups)
        )
        
        if num_total_hit_tokens == request.num_tokens:
            external_hit_tokens = max(0, external_hit_tokens - 1)

        if self._num_groups == 1:
            logger.info(
                f"request_id: {request.request_id}, "
                f"total_blocks: {len(ucm_block_ids_by_group[0])}, "
                f"hit hbm: {hbm_hit_block_num_by_group[0]}, "
                f"hit external: {external_hit_tokens // self.block_size}"
            )
        else:
            logger.info(
                f"request_id: {request.request_id}, num_groups: {self._num_groups}, "
                f"external_hit_tokens: {external_hit_tokens}"
            )
        
        if self.metrics_config and external_hit_tokens > 0:
            total_blocks = sum(len(ids) for ids in ucm_block_ids_by_group)
            hit_rate = external_hit_tokens / request.num_tokens if total_blocks > 0 else 0
            self.monitor.update_stats("ConnStats", {"interval_lookup_hit_rates": hit_rate})

        self.requests_meta[request.request_id] = RequestMeta(
            ucm_block_ids_by_group=ucm_block_ids_by_group,
            hbm_hit_block_num_by_group=hbm_hit_block_num_by_group,
            total_hit_block_num_by_group=total_hit_block_num_by_group,
            vllm_block_ids_by_group=[[] for _ in range(self._num_groups)],
            num_token_ids=len(request.all_token_ids),
            token_processed=num_total_hit_tokens,
        )

        return external_hit_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def _generate_dispatch_meta(
        self,
        req_meta: RequestMeta,
        new_tokens: int,
        vllm_block_ids_per_group: tuple[list[int], ...],
        need_load: bool = True,
    ) -> RequestDispatchMeta:
        """
        Request Blocks layout:
        ----------------------------------------------------------------------------------------------------
        | local_computed_block(HBM hit) | external_computed_block(external hit) | new_block(need to dump)  |
        ----------------------------------------------------------------------------------------------------
        |      hbm_hit_block_num        |                 LOAD                  |     new_blocks_num       |
        ----------------------------------------------------------------------------------------------------
        |                              total_hit_block_num                      |
        ----------------------------------------------------------------------------------------------------
        |                                         scheduled_block_num                                      |
        """
        load_by_group: list[tuple[list[str], list[int]]] = []
        dump_by_group: list[tuple[list[str], list[int]]] = []

        for g in range(self._num_groups):
            block_size_g = self._block_sizes[g]
            ucm_block_ids_g = req_meta.ucm_block_ids_by_group[g]
            vllm_block_ids_g = vllm_block_ids_per_group[g]
            req_meta.vllm_block_ids_by_group[g].extend(vllm_block_ids_g)

            hbm_hit_g = req_meta.hbm_hit_block_num_by_group[g]
            total_hit_g = req_meta.total_hit_block_num_by_group[g]

            load_ucm_g = ucm_block_ids_g[hbm_hit_g:total_hit_g] if need_load else []
            load_vllm_g = vllm_block_ids_g[hbm_hit_g:total_hit_g] if need_load else []

            if req_meta.token_processed < req_meta.num_token_ids:
                start_idx = req_meta.token_processed // block_size_g
                end_idx = (req_meta.token_processed + new_tokens) // block_size_g
                dump_ucm_g = ucm_block_ids_g[start_idx:end_idx]
                dump_vllm_g = req_meta.vllm_block_ids_by_group[g][start_idx:end_idx]
            else:
                dump_ucm_g, dump_vllm_g = [], []

            load_by_group.append((load_ucm_g, load_vllm_g))
            dump_by_group.append((dump_ucm_g, dump_vllm_g))

        if new_tokens > 0:
            req_meta.token_processed += new_tokens

        return RequestDispatchMeta(
            load_block_ids_by_group=load_by_group,
            dump_block_ids_by_group=dump_by_group,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        requests_dispatch_meta = {}
        
        def normalize_block_ids(block_ids) -> tuple[list[int], ...]:
            if block_ids is None:
                return tuple([] for _ in range(self._num_groups))
            if isinstance(block_ids, tuple):
                if len(block_ids) == self._num_groups:
                    return block_ids
                if self._num_groups == 1:
                    return (block_ids[0],) if block_ids else ([],)
                if len(block_ids) == 1:
                    logger.warning(f"Got single-element tuple for {self._num_groups}-group model, broadcasting")
                    return tuple(block_ids[0] for _ in range(self._num_groups))
                logger.error(f"Tuple length {len(block_ids)} != num_groups {self._num_groups}")
                return tuple([] for _ in range(self._num_groups))
            
            if isinstance(block_ids, list):
                if self._num_groups == 1:
                    return (block_ids,)
                return tuple(block_ids for _ in range(self._num_groups))
            
            logger.error(f"Unexpected block_ids type: {type(block_ids)}")
            return tuple([] for _ in range(self._num_groups))

        for request in scheduler_output.scheduled_new_reqs:
            request_id, vllm_block_ids = request.req_id, request.block_ids
            req_meta = self.requests_meta.get(request_id)
            if req_meta:
                requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                    req_meta,
                    scheduler_output.num_scheduled_tokens[request_id],
                    normalize_block_ids(vllm_block_ids),
                )

        # for cached request, there are 3 situation:
        # 1. chunked prefill: we only need dump
        # 2. resumed: we need to handle like new request
        # 3. TODO decode stage: nothing happened
        scheduled_cached_reqs = scheduler_output.scheduled_cached_reqs
        if not isinstance(scheduled_cached_reqs, list):
            # >= 0.9.2
            for i, request_id in enumerate(scheduled_cached_reqs.req_ids):
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    new_block_ids = []
                    if scheduled_cached_reqs.new_block_ids[i] != None:
                        new_block_ids = scheduled_cached_reqs.new_block_ids[i]
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        normalize_block_ids(new_block_ids),
                        need_load=(request_id in scheduled_cached_reqs.resumed_req_ids),
                    )
        else:
            for request in scheduled_cached_reqs:
                request_id = request.req_id
                req_meta = self.requests_meta.get(request_id)
                if req_meta:
                    requests_dispatch_meta[request_id] = self._generate_dispatch_meta(
                        req_meta,
                        scheduler_output.num_scheduled_tokens[request_id],
                        normalize_block_ids(request.new_block_ids),
                        need_load=request.resumed_from_preemption,
                    )

        # clear finished request
        for request_id in scheduler_output.finished_req_ids:
            self.requests_meta.pop(request_id, None)

        return UCMConnectorMetadata(requests_dispatch_meta)

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        if len(self.kv_caches) > 0:
            return
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]
        # Since vllm_ascend >= 0.10.0, the MLA model's tensor shape has changed to
        # (2, num_blocks, block_size, num_kv_heads, nope_dim/rope_dim).
        # Currently, we treat it as GQA, and use is_dsa to mark it,
        # which works but leads to space inefficiency.
        # TODO: Optimize this to avoid unnecessary space usage.
        sample_kv_layer = next(iter(self.kv_caches.values()))
        if self.is_mla and len(sample_kv_layer) == 2 and isinstance(sample_kv_layer, (list, tuple)) :
            self.is_mla = False
            self.is_dsa = True
        if self.kv_cache_dtype is None:
            if isinstance(sample_kv_layer, torch.Tensor):
                self.kv_cache_dtype = sample_kv_layer.dtype
            else:
                self.kv_cache_dtype = sample_kv_layer[0].dtype

    @staticmethod
    def _extract_layer_index(layer_name: str) -> Optional[int]:
        """
        Extract the layer index from the layer name.
        """
        for chunk in layer_name.split("."):
            if chunk.isdigit():
                return int(chunk)
        return None

    @staticmethod
    def _get_block_view_for_layer(kv_cache_obj: Any, block_id: int) -> list[torch.Tensor]:
        if isinstance(kv_cache_obj, torch.Tensor):
            if kv_cache_obj.dim() >= 3 and kv_cache_obj.shape[0] == 2:
                if block_id >= kv_cache_obj.shape[1]:
                    raise IndexError(
                        f"block_id {block_id} out of range for {kv_cache_obj.shape[1]} blocks"
                    )
                return [kv_cache_obj[0, block_id], kv_cache_obj[1, block_id]]
            if kv_cache_obj.dim() >= 3 and kv_cache_obj.shape[1] == 2:
                if block_id >= kv_cache_obj.shape[0]:
                    raise IndexError(
                        f"block_id {block_id} out of range for {kv_cache_obj.shape[0]} blocks"
                    )
                return [kv_cache_obj[block_id, 0], kv_cache_obj[block_id, 1]]
            return [kv_cache_obj[block_id]]

        if isinstance(kv_cache_obj, (list, tuple)):
            return [t[block_id] for t in kv_cache_obj]

        raise TypeError(f"Unsupported kv_cache type: {type(kv_cache_obj)}")

    def _precompute_layer_offsets(self):
        if not self.kv_caches:
            return

        offset = 0

        def _layer_sort_key(ln: str) -> tuple[int, str]:
            idx = self._extract_layer_index(ln)
            return (idx if idx is not None else 10**9, ln)

        for layer_name in sorted(self.kv_caches.keys(), key=_layer_sort_key):
            kv_cache_obj = self.kv_caches[layer_name]
            block_views = self._get_block_view_for_layer(kv_cache_obj, 0)
            layer_offsets: list[int] = []
            for view in block_views:
                layer_offsets.append(offset)
                offset += view.numel() * view.element_size()
            if len(layer_offsets) == 2:
                self._layer_offset_cache[layer_name] = tuple(layer_offsets)
            elif len(layer_offsets) == 1:
                self._layer_offset_cache[layer_name] = (layer_offsets[0], 0)
            else:
                self._layer_offset_cache[layer_name] = layer_offsets

        self.block_data_size = offset
        if (
            getattr(self, "_store_config", None) is not None
            and self.role == KVConnectorRole.WORKER
            and self.block_data_size > int(self._store_config.get("kv_block_size", 0))
        ):
            new_cfg = dict(self._store_config)
            new_cfg["kv_block_size"] = int(self.block_data_size)
            new_cfg["io_size"] = int(min(new_cfg.get("io_size", self.block_data_size), self.block_data_size))
            logger.info(
                "[UCM] resize kv_block_size %s -> %s for hybrid/state support",
                self._store_config.get("kv_block_size"),
                new_cfg["kv_block_size"],
            )
            self._store_config = new_cfg
            self.store = UcmConnectorFactory.create_connector(self._store_name, new_cfg)

    def _get_tensor_and_offset(
        self, vllm_block_ids: list[int], kv_layer: Any, layer_name: str
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        GQA/MHA: one layer shape is (2, num_blocks, block_size, num_kv_heads, head_size)
        MLA: one layer shape is (num_blocks, block_size, head_size)
        """
        layer_offsets = self._layer_offset_cache[layer_name]

        if isinstance(layer_offsets, tuple):
            if not vllm_block_ids:
                return [], []

            first_views = self._get_block_view_for_layer(kv_layer, vllm_block_ids[0])
            if len(first_views) == 2:
                k_tensors: list[torch.Tensor] = []
                v_tensors: list[torch.Tensor] = []
                for vllm_block_id in vllm_block_ids:
                    views = self._get_block_view_for_layer(kv_layer, vllm_block_id)
                    k_tensors.append(views[0])
                    v_tensors.append(views[1])
                k_off = layer_offsets[0]
                v_off = layer_offsets[1] if len(layer_offsets) > 1 else layer_offsets[0]
                return (
                    k_tensors + v_tensors,
                    [k_off] * len(k_tensors) + [v_off] * len(v_tensors),
                )

            tensors = [self._get_block_view_for_layer(kv_layer, bid)[0] for bid in vllm_block_ids]
            return tensors, [layer_offsets[0]] * len(tensors)

        tensors: list[torch.Tensor] = []
        offsets: list[int] = []
        for vllm_block_id in vllm_block_ids:
            block_views = self._get_block_view_for_layer(kv_layer, vllm_block_id)
            tensors.extend(block_views)
            offsets.extend(layer_offsets[: len(block_views)])

        # Reorder from [b0_s0,b0_s1,b1_s0,b1_s1,...] to [all_s0_blocks, all_s1_blocks,...]
        num_states = len(layer_offsets)
        if num_states > 1 and vllm_block_ids:
            per_state_tensors: list[list[torch.Tensor]] = [[] for _ in range(num_states)]
            for vllm_block_id in vllm_block_ids:
                views = self._get_block_view_for_layer(kv_layer, vllm_block_id)
                for i in range(min(num_states, len(views))):
                    per_state_tensors[i].append(views[i])
            tensors = []
            offsets = []
            for i in range(num_states):
                tensors.extend(per_state_tensors[i])
                offsets.extend([layer_offsets[i]] * len(per_state_tensors[i]))

        return tensors, offsets

    def _generate_task(
        self,
        vllm_block_ids: List[int],
        ucm_block_ids: List[str],
        layer_names: Optional[List[str]] = None,
    ):
        if not self._layer_offset_cache:
            self._precompute_layer_offsets()

        dst_tensor_addr: list[torch.Tensor] = []
        ucm_offsets: list[int] = []

        def _layer_sort_key(ln: str) -> tuple[int, str]:
            idx = self._extract_layer_index(ln)
            return (idx if idx is not None else 10**9, ln)

        keys = (
            sorted(
                (ln for ln in (layer_names or []) if ln in self.kv_caches),
                key=_layer_sort_key,
            )
            if layer_names is not None
            else sorted(self.kv_caches.keys(), key=_layer_sort_key)
        )
        for layer_name in keys:
            one_layer_kv_cache = self.kv_caches[layer_name]
            tensors, offsets = self._get_tensor_and_offset(vllm_block_ids, one_layer_kv_cache, layer_name)
            dst_tensor_addr.extend(tensors)
            ucm_offsets.extend(offsets)

        if not ucm_block_ids:
            return [], [], []

        repeat_times = len(ucm_offsets) // len(ucm_block_ids)
        ucm_total_block_ids = ucm_block_ids * repeat_times

        assert len(ucm_total_block_ids) == len(ucm_offsets) == len(dst_tensor_addr)
        return ucm_total_block_ids, ucm_offsets, dst_tensor_addr

    def _broadcast(self, dst_tensor_addr: list[torch.Tensor]):
        rec_tensor: torch.Tensor = None
        with torch.cuda.stream(self.broadcast_stream):
            # TODO support broadcast when PP
            if self.global_rank == 0:
                tensor_to_broadcast = torch.stack(dst_tensor_addr, dim=0)
                self.broadcast_fn(tensor_to_broadcast, 0)
            else:
                shape = (len(dst_tensor_addr),) + dst_tensor_addr[0].shape
                # TODO create earlier
                rec_tensor = torch.empty(
                    shape, dtype=self.kv_cache_dtype, device=self.device
                )
                self.broadcast_fn(rec_tensor, 0)
        self.broadcast_stream.synchronize()
        if self.global_rank != 0 and rec_tensor is not None:
            for i, tensor in enumerate(dst_tensor_addr):
                tensor.copy_(rec_tensor[i])

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        self._init_kv_caches_from_forward_context(forward_context)

        load_tasks: list[tuple[str, Optional[Task], list[torch.Tensor], list[int]]] = []
        num_loaded_block = 0
        num_loaded_request = 0
        load_start_time = time.perf_counter() * 1000
        for request_id, request in metadata.request_meta.items():
            req_had_load = False
            for g, (load_ucm, load_vllm) in enumerate(request.load_block_ids_by_group):
                if not load_ucm:
                    continue
                
                req_had_load = True
                num_loaded_block += len(load_ucm)
                
                ucm_block_ids = list(load_ucm)
                if self.global_rank != 0 and not self.is_mla and not self.is_dsa:
                    ucm_block_ids = [str(self.request_hasher(uid)) for uid in ucm_block_ids]
                
                layer_names = self._group_layer_names[g] if self._group_layer_names else None
                ucm_total_ids, offsets, dst_addrs = self._generate_task(
                    list(load_vllm), ucm_block_ids, layer_names=layer_names
                )
                
                task = None
                if self.global_rank == 0 or not self.load_only_first_rank:
                    task = self.store.load(ucm_total_ids, offsets, dst_addrs)
                load_tasks.append((request_id, task, dst_addrs, list(load_vllm)))
            
            if req_had_load:
                num_loaded_request += 1

        for request_id, task, dst_addrs, vllm_ids in load_tasks:
            # TODO error handling
            if self.global_rank == 0 or not self.load_only_first_rank:
                if self.store.wait(task) != 0:
                    self._invalid_block_ids.update(vllm_ids)
                    logger.error(f"request {request_id} load kv cache failed.")
            if self.load_only_first_rank and dst_addrs:
                self._broadcast(dst_addrs)
        load_end_time = time.perf_counter() * 1000
        load_speed = (
            num_loaded_block
            * self.block_data_size
            / (load_end_time - load_start_time)
            / 1024
            / 1024
        )  # GB/s
        if self.metrics_config and num_loaded_block > 0:
            self.monitor.update_stats(
                "ConnStats",
                {
                    "load_requests_num": num_loaded_request,
                    "load_blocks_num": num_loaded_block,
                    "load_duration": load_end_time - load_start_time,
                    "load_speed": load_speed,
                },
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self) -> None:

        # TODO support PP
        if (self.is_mla or self.is_dsa) and self.global_rank != 0:
            return
        if self.metrics_config or current_platform.device_type == "npu":
            # When use vllm_ascend, we should add synchronize here, otherwise accuracy problem will raise
            # This has already been fixed in the latest main branch of vllm_ascend, so synchronize will no longer be needed in future versions.
            self.synchronize()

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        save_tasks: list[tuple[str, Task, list[str]]] = []
        num_saved_block = 0
        num_saved_request = 0
        save_start_time = time.perf_counter() * 1000
        for request_id, request in metadata.request_meta.items():
            req_had_save = False
            for g, (dump_ucm, dump_vllm) in enumerate(request.dump_block_ids_by_group):
                if not dump_ucm:
                    continue
                
                req_had_save = True
                num_saved_block += len(dump_ucm)
                
                ucm_block_ids = list(dump_ucm)
                if self.global_rank != 0:
                    ucm_block_ids = [str(self.request_hasher(uid)) for uid in ucm_block_ids]
                
                rets = self.store.create(ucm_block_ids)
                end = next((i for i, ret in enumerate(rets) if ret != 0), len(rets))
                if end == 0:
                    logger.error(f"create blocks for {request_id} group {g} failed")
                    continue
                ucm_block_ids = ucm_block_ids[:end]
                
                layer_names = self._group_layer_names[g] if self._group_layer_names else None
                ucm_total_ids, offsets, src_addrs = self._generate_task(
                    list(dump_vllm[:end]), ucm_block_ids, layer_names=layer_names
                )
                task = self.store.dump(ucm_total_ids, offsets, src_addrs)
                save_tasks.append((request_id, task, ucm_block_ids))
            
            if req_had_save:
                num_saved_request += 1

        for request_id, task, ucm_block_ids in save_tasks:
            if self.store.wait(task) == 0:
                self.store.commit(ucm_block_ids, True)
            else:
                logger.error(f"request {request_id} dump kv cache failed.")
                self.store.commit(ucm_block_ids, False)
        save_end_time = time.perf_counter() * 1000
        save_speed = (
            num_saved_block
            * self.block_data_size
            / (save_end_time - save_start_time)
            / 1024
            / 1024
        )  # GB/s
        if self.metrics_config and num_saved_block > 0:
            self.monitor.update_stats(
                "ConnStats",
                {
                    "save_requests_num": num_saved_request,
                    "save_blocks_num": num_saved_block,
                    "save_duration": save_end_time - save_start_time,
                    "save_speed": save_speed,
                },
            )

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        res = self._invalid_block_ids
        self._invalid_block_ids = set()
        return res

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None


class UCMLayerWiseConnector(UCMDirectConnector):
    """
    This Connector means overlap:
    load l0 -> forward l0 -> save l0
               load l1    -> forward l1 -> save l1
                             load l2    -> forward l2 -> save l2
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self.layerwise_load_tasks: dict[str, dict[str, Task]] = {}
        self.layerwise_dump_tasks: dict[str, dict[str, list[Task]]] = {}
        self.created_blocks: set[str] = set()
        self.layer_names_list: list[str] = []
        self.request_load_metadata: dict = {}

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        self._init_kv_caches_from_forward_context(forward_context)

        self.layerwise_load_tasks.clear()
        self.layerwise_dump_tasks.clear()
        self.request_load_metadata.clear()

        if not self._layer_offset_cache:
            self._precompute_layer_offsets()

        self.layer_names_list = list(self.kv_caches.keys())
        if not self.layer_names_list:
            return

        for request_id, request_meta in metadata.request_meta.items():
            if len(request_meta.load_block_ids[0]) == 0:
                continue

            ucm_block_ids, vllm_block_ids = request_meta.load_block_ids
            if self.global_rank != 0 and not self.is_mla and not self.is_dsa:
                ucm_block_ids = [str(self.request_hasher(bid)) for bid in ucm_block_ids]

            self.request_load_metadata[request_id] = {
                'ucm_block_ids': ucm_block_ids,
                'vllm_block_ids': vllm_block_ids,
            }
            self.layerwise_load_tasks[request_id] = {}

        if self.layer_names_list:
            logger.debug("Pipeline start: loading layers")
            for layer_name in self.layer_names_list:
                self._load_single_layer(layer_name)

    def _load_single_layer(self, layer_name: str) -> None:
        kv_layer = self.kv_caches.get(layer_name)
        if kv_layer is None:
            return

        for request_id, req_meta in self.request_load_metadata.items():
            tensors, offsets = self._get_tensor_and_offset(
                req_meta['vllm_block_ids'], kv_layer, layer_name
            )
            block_ids = req_meta['ucm_block_ids'] * (1 if self.is_mla else 2)
            self.layerwise_load_tasks[request_id][layer_name] = self.store.load(
                block_ids, offsets, tensors
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.layerwise_load_tasks:
            return
        
        for request_id, layer_tasks in self.layerwise_load_tasks.items():
            task = layer_tasks.get(layer_name)
            if task:
                self.store.wait(task)
                return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if (self.is_mla or self.is_dsa) and self.global_rank != 0:
            return

        metadata = self._get_connector_metadata()
        assert isinstance(metadata, UCMConnectorMetadata)

        if self._is_first_layer(layer_name):
            self.created_blocks.clear()
            self.layerwise_dump_tasks.clear()
            self.save_start_time = time.perf_counter() * 1000

            for request_id, request_meta in metadata.request_meta.items():
                if len(request_meta.dump_block_ids[0]) == 0:
                    continue

                ucm_block_ids = request_meta.dump_block_ids[0]
                if self.global_rank != 0:
                    ucm_block_ids = [str(self.request_hasher(bid)) for bid in ucm_block_ids]

                rets = self.store.create(ucm_block_ids)
                self.created_blocks.update(
                    bid for bid, ret in zip(ucm_block_ids, rets) if ret == 0
                )

        for request_id, request_meta in metadata.request_meta.items():
            if len(request_meta.dump_block_ids[0]) == 0:
                continue

            ucm_block_ids, vllm_block_ids = request_meta.dump_block_ids
            if self.global_rank != 0:
                ucm_block_ids = [str(self.request_hasher(bid)) for bid in ucm_block_ids]

            valid_pairs = [
                (u, v) for u, v in zip(ucm_block_ids, vllm_block_ids)
                if u in self.created_blocks
            ]
            if not valid_pairs:
                continue

            ucm_ids, vllm_ids = zip(*valid_pairs)
            tensors, offsets = self._get_tensor_and_offset(list(vllm_ids), kv_layer, layer_name)
            block_ids = list(ucm_ids) * (1 if self.is_mla else 2)
            task = self.store.dump(block_ids, offsets, tensors)
            
            self.layerwise_dump_tasks.setdefault(request_id, {}).setdefault(layer_name, []).append(task)

    def wait_for_save(self) -> None:
        if (self.is_mla or self.is_dsa) and self.global_rank != 0:
            return

        for request_id, layer_tasks in self.layerwise_dump_tasks.items():
            for layer_name, tasks in layer_tasks.items():
                for task in tasks:
                    if self.store.wait(task) != 0:
                        logger.error(f"Save failed: layer {layer_name}, request {request_id}")

        if self.created_blocks:
            self.store.commit(list(self.created_blocks), True)

        self.layerwise_dump_tasks.clear()
        self.created_blocks.clear()

    def _is_first_layer(self, layer_name: str) -> bool:
        return self.kv_caches and layer_name == next(iter(self.kv_caches.keys()))


class UCMPDConnector(UCMDirectConnector):
    """
    This Connector means overlap (especially for Decode Instance):
    step (req0,1,2) forward -> step (req0,1,2,3) forward
    load req3               -> load req4
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config, role)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """
        raise NotImplementedError


class UCMMockConnector(UCMDirectConnector):
    """
    This Connector can control hit ratio, for example: if your hit ratio is 100%,
    you can set "hit_ratio" by config or env_vars, then get_num_new_matched_tokens()
    will reduce hit_tokens under the hit_ratio you set.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._hit_ratio = float(self.launch_config["hit_ratio"])
        logger.info(f"hit_ratio: {self._hit_ratio}")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        hit_tokens, _ = super().get_num_new_matched_tokens(request, num_computed_tokens)
        expect_hit_tokens = int(self._hit_ratio * request.num_prompt_tokens)
        if hit_tokens <= expect_hit_tokens:
            return hit_tokens, False
        expect_hit_block_num = expect_hit_tokens // self.block_size
        request_meta = self.requests_meta[request.request_id]
        request_meta.total_hit_block_num = expect_hit_block_num
        request_meta.hbm_hit_block_num = min(
            expect_hit_block_num, request_meta.hbm_hit_block_num
        )

        logger.info(
            "Hijacked By MockConnector,"
            f"request_id: {request.request_id}, "
            f"total_blocks_num: {len(request_meta.ucm_block_ids)}, "
            f"hit hbm: {request_meta.hbm_hit_block_num}, "
            f"hit external: {request_meta.total_hit_block_num - request_meta.hbm_hit_block_num}"
        )

        return expect_hit_block_num * self.block_size, False


class UCMConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config=vllm_config, role=role)
        self.connector: KVConnectorBase_V1
        
        extra_config = (
            self._vllm_config.kv_transfer_config.kv_connector_extra_config
            if self._vllm_config.kv_transfer_config is not None
            else {}
        )
        ucm_config = Config(vllm_config.kv_transfer_config)
        launch_config = ucm_config.get_config()
        
        if "hit_ratio" in extra_config or "hit_ratio" in launch_config:
            self.connector = UCMMockConnector(vllm_config, role, kv_cache_config)
            logger.info("Using UCMMockConnector for hit ratio testing")
        elif extra_config.get("use_layerwise", False) or launch_config.get("use_layerwise", False):
            self.connector = UCMLayerWiseConnector(vllm_config, role, kv_cache_config)
            logger.info("Using UCMLayerWiseConnector for layer-wise pipelining")
        else:
            self.connector = UCMDirectConnector(vllm_config, role, kv_cache_config)
            logger.info("Using UCMDirectConnector for synchronous mode")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        return self.connector.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        self.connector.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self.connector.build_connector_meta(scheduler_output)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Set the connector metadata from the scheduler.

        This function should be called by the model runner every time
        before the model execution. The metadata will be used for runtime
        KV cache loading and saving.

        Args:
            connector_metadata (dict): the connector metadata.
        """
        self.connector.bind_connector_metadata(connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.

        """
        self.connector.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self.connector.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self.connector.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self.connector.wait_for_save()

    def clear_connector_metadata(self) -> None:
        """Clear the connector metadata.

        This function should be called by the model runner every time
        after the model execution.
        """
        self.connector.clear_connector_metadata()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Get the set of block IDs that failed to load.

        Returns:
            Set of block IDs that encountered load errors.
            Empty set if no load errors occurred.
        """
        return self.connector.get_block_ids_with_load_errors()

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None
