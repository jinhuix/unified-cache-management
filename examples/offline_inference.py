import contextlib
import os
import time
import torch
from dataclasses import asdict
from datetime import datetime

# 指定设备号
TP_SIZE = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["UNIFIED_CACHE_LOG_LEVEL"] = "INFO"

# Profiler 配置
# - ENABLE_NVTX=1：仅添加 NVTX 标记，配合 nsys 的 nvtx/cuda trace 便于在时间线上定位 decode
ENABLE_NVTX = os.getenv("ENABLE_NVTX", "1") == "1"
ENABLE_TORCH_PROFILER = os.getenv("ENABLE_TORCH_PROFILER", "0") == "1"
if ENABLE_TORCH_PROFILER:
    trace_dir = f"/home/xujinhui/unified-cache-management/examples/trace/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(trace_dir, exist_ok=True)
    os.environ["VLLM_TORCH_PROFILER_DIR"] = trace_dir

from transformers import AutoTokenizer

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from ucm.logger import init_logger

logger = init_logger(__name__)

@contextlib.contextmanager
def build_llm_with_uc(module_path: str, name: str, model: str):
    ktc = KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_connector_extra_config={
            "UCM_CONFIG_FILE": "/home/xujinhui/unified-cache-management/examples/ucm_config_example.yaml"
        },
    )

        # compilation_config=CompilationConfig(
        #     cudagraph_mode="FULL_AND_PIECEWISE",
        # ),
        # kv_transfer_config=ktc,

    llm_args = EngineArgs(
        model=model,
        max_model_len=5000,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=30000,
        block_size=128,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )

    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        logger.info("LLM engine is exiting.")


def print_output(
    llm: LLM,
    prompt: list[str],
    sampling_params: SamplingParams,
    req_str: str,
):
    start = time.time()
    outputs = llm.generate(prompt, sampling_params)
    print(f"Generation took {time.time() - start:.2f} seconds, {req_str} request done.")
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids) if hasattr(output, 'prompt_token_ids') else 0
        output_tokens = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else 0
        
        print(f"Request {req_str}:")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Output tokens: {output_tokens}")
        print(f"Generated text: {generated_text!r}")
    print("-" * 50)

def generate_prompt(
    num_prompts: int = 1,
    repetitions: int = 3,
    request_id: int = 1,
    reuse_ratio: float = 1.0,
) -> list[str]:
    base_text = """
    You are a highly specialized assistant whose mission is to faithfully reproduce English literary texts verbatim, without any deviation, paraphrasing, or omission. Your primary responsibility is accuracy: every word, every punctuation mark, and every line must appear exactly as in the original source.
    
    Core Principles:
    1. Verbatim Reproduction: If the user asks for a passage, you must output the text word-for-word. Do not alter spelling, punctuation, capitalization, or line breaks. Do not paraphrase, summarize, modernize, or "improve" the language.
    2. Consistency: The same input must always yield the same output. Do not generate alternative versions or interpretations.
    3. Clarity of Scope: Your role is not to explain, interpret, or critique. You are not a storyteller or commentator, but a faithful copyist of English literary and cultural texts.
    4. Recognizability: Because texts must be reproduced exactly, they will carry their own cultural recognition. You should not add labels, introductions, or explanations before or after the text.
    5. Coverage: You must handle passages from classic literature, poetry, speeches, or cultural texts. Regardless of tone—solemn, visionary, poetic, persuasive—you must preserve the original form, structure, and rhythm by reproducing it precisely.
    
    Success Criteria:
    - A human reader should be able to compare your output directly with the original and find zero differences.
    - The measure of success is absolute textual fidelity.
    
    Your function can be summarized as follows: verbatim reproduction only, no paraphrase, no commentary, no embellishment, no omission.
    
    Example Texts for Reproduction:
    
    1. United States Declaration of Independence (1776):
    "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation."
    
    2. Shakespeare's Sonnet 18:
    "Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the darling buds of May, And summer's lease hath all too short a date; Sometime too hot the eye of heaven shines, And often is his gold complexion dimm'd; And every fair from fair sometime declines, By chance or nature's changing course untrimm'd; But thy eternal summer shall not fade, Nor lose possession of that fair thou ow'st; Nor shall death brag thou wander'st in his shade, When in eternal lines to time thou grow'st: So long as men can breathe or eyes can see, So long lives this, and this gives life to thee."
    
    3. Martin Luther King Jr.'s "I Have a Dream" Speech:
    "I have a dream that one day this nation will rise up and live out the true meaning of its creed: 'We hold these truths to be self-evident, that all men are created equal.' I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood."
    
    4. Jane Austen's Pride and Prejudice:
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters."
    
    5. Charles Dickens' A Tale of Two Cities:
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair."
    
    6. Robert Frost's "The Road Not Taken":
    "Two roads diverged in a yellow wood, And sorry I could not travel both And be one traveler, long I stood And looked down one as far as I could To where it bent in the undergrowth; Then took the other, as just as fair, And having perhaps the better claim, Because it was grassy and wanted wear; Though as for that the passing there Had worn them really about the same."
    
    Now, please reproduce verbatim the following texts exactly as they appear in their original form. Do not add any commentary, explanation, or interpretation.
    
    Request: Please provide the complete text of the Gettysburg Address by Abraham Lincoln, exactly as it was delivered on November 19, 1863.
    """
    
    prompts = []
    
    extended_text = base_text * repetitions
    for i in range(num_prompts):
        prompt = f"[Prompt {i+1}] {extended_text}"
        prompts.append(prompt)
    
    if reuse_ratio < 1.0:
        for idx, prompt in enumerate(prompts):
            prefix_len = int(len(prompt) * reuse_ratio)
            prefix = prompt[:prefix_len]
            suffix = f" [Different Request {request_id}] " + "This is a different request with modified content. " * 20
            prompts[idx] = prefix + suffix
    
    return prompts

def main():
    module_path = "ucm.integration.vllm.ucm_connector"
    name = "UCMConnector"
    model = os.getenv("MODEL_PATH", "/home/models/Qwen2.5-14B-Instruct")

    tokenizer = AutoTokenizer.from_pretrained(model, use_chat_template=True)

    with build_llm_with_uc(module_path, name, model) as llm:
        prompts_1 = generate_prompt(num_prompts=1, repetitions=3)
        prompts_2 = generate_prompt(num_prompts=1, repetitions=3, reuse_ratio=0.3, request_id=2)
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=100)

        if ENABLE_TORCH_PROFILER:
            print("\n[INFO] Starting profiler...")
            llm.start_profile()

        print(f"[RUN] decode iter 1")
        if ENABLE_NVTX:
            torch.cuda.nvtx.range_push("decode-1")
        print_output(llm, prompts_1, sampling_params, f"decode-1")
        if ENABLE_NVTX:
            torch.cuda.nvtx.range_pop()
        print(f"[RUN] decode iter 2")
        if ENABLE_NVTX:
            torch.cuda.nvtx.range_push("decode-2")
        print_output(llm, prompts_2, sampling_params, f"decode-2")
        if ENABLE_NVTX:
            torch.cuda.nvtx.range_pop()

        if ENABLE_TORCH_PROFILER:
            print("\n[INFO] Stopping profiler...")
            llm.stop_profile()
            print("[SUCCESS] Profiler traces saved. Check vLLM logs for trace location.")

if __name__ == "__main__":
    main()