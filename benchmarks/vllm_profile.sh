rm -rf /home/xujinhui/test_backend/data

export PYTHONHASHSEED=123456
export CUDA_VISIBLE_DEVICES=6,7
vllm serve /home/models/DeepSeek-V2-Lite  \
    --max-model-len 5000 \
    --tensor-parallel-size 2 \
    --gpu_memory_utilization 0.7 \
    --trust-remote-code \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --enforce-eager \
    --max-num-batched-tokens 80000 \
    --max-num-seqs 20 \
    --host 0.0.0.0 \
    --port 7885 \
    --kv-transfer-config \
    '{
        "kv_connector": "UnifiedCacheConnectorV1",
        "kv_connector_module_path": "ucm.integration.vllm.uc_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "ucm_connector_name": "UcmNfsStore",
            "ucm_connector_config": {
                "storage_backends": "/home/xujinhui/test_backend"
            },
            "use_layerwise": false
        }
    }' \
    > /home/xujinhui/unified-cache-management/benchmarks/trace/serve1.log 2>&1 &

# 等待 vLLM 服务真正就绪
echo "⏳ Waiting for vLLM server to be ready on http://localhost:7885"
MAX_WAIT=600  # 最多等待 600 秒
COUNT=0
while [ "$COUNT" -lt "$MAX_WAIT" ]; do
    if curl -s http://localhost:7885/v1/models >/dev/null; then
        echo "✅ vLLM server is ready!"
        break
    fi
    sleep 5
    COUNT=$((COUNT + 5))
    echo -n "."
done

if [ "$COUNT" -ge "$MAX_WAIT" ]; then
    echo ""
    echo "❌ vLLM server failed to start within $MAX_WAIT seconds."
    exit 1
fi

# 启动测试

echo "🚀 预热..."
vllm bench serve \
    --backend vllm \
    --model /home/models/DeepSeek-V2-Lite \
    --host 127.0.0.1 \
    --port 7885 \
    --seed 123456  \
    --dataset-name random \
    --num-prompts 200 \
    --random-input-len 2000 \
    --random-output-len 1 \
    --request-rate inf \
    --percentile-metrics  "ttft,tpot,itl,e2el" \
    --metric-percentiles  "90,99" \
    --goodput "ttft:2000" "tpot:40" \
    --ignore-eos \
    --save-result \
    --save-detailed \
    --result-dir /home/xujinhui/unified-cache-management/benchmarks/trace \
    --result-filename data.json 
# --seed $(date +%s) \
sleep 1
echo "🚀 Starting test..."
vllm bench serve \
    --backend vllm \
    --model /home/models/DeepSeek-V2-Lite \
    --host 127.0.0.1 \
    --port 7885 \
    --seed 123456  \
    --dataset-name random \
    --num-prompts 200 \
    --random-input-len 2000 \
    --random-output-len 1 \
    --request-rate inf \
    --percentile-metrics  "ttft,tpot,itl,e2el" \
    --metric-percentiles  "90,99" \
    --goodput "ttft:2000" "tpot:40" \
    --ignore-eos \
    --save-result \
    --save-detailed \
    --result-dir /home/xujinhui/unified-cache-management/benchmarks/trace \
    --result-filename data.json 
# --seed $(date +%s) \

# python3 result_to_excel.py 

pkill vllm