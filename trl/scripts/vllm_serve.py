# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import torch
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import chain
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional, List, Dict, Any, Union, Tuple

from pathlib import Path

from trl import TrlParser
from trl.import_utils import (
    is_fastapi_available,
    is_pydantic_available,
    is_uvicorn_available,
    is_vllm_ascend_available,
    is_vllm_available,
)

import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM, AutoConfig
from esm.sdk.api import ESMProtein, SamplingConfig


# Import DNA processing components with proper error handling
DNA_LLM_AVAILABLE = False

_MAIN_TEXT_TOKENIZER = None

def get_main_text_tokenizer(model_name: str) -> "transformers.PreTrainedTokenizer":
    """
    Lazily load the text tokenizer once in the FastAPI (parent) process.
    Uses the same chat template constant as the workers so formatting matches.
    """
    global _MAIN_TEXT_TOKENIZER

    if _MAIN_TEXT_TOKENIZER is None:
        print(f"📝 Loading main-process tokenizer for template formatting …")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if PROTEIN_CHAT_TEMPLATE:
            print("chat template found, applying to tokenizer:", PROTEIN_CHAT_TEMPLATE)
            tok.chat_template = PROTEIN_CHAT_TEMPLATE
        tok.pad_token = tok.bos_token
        _MAIN_TEXT_TOKENIZER = tok
    return _MAIN_TEXT_TOKENIZER


def load_dna_components():
    """Load DNA components only when needed."""
    global DNA_LLM_AVAILABLE, DNAInput, DLProcessor, CHAT_TEMPLATE, EVO2_AVAILABLE
    
    if DNA_LLM_AVAILABLE:
        return  # Already loaded
        
    try:
        import torch
        import torch.nn as nn
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoConfig,
        )
        
        from bioreason.utils.dna_utils import DNAInput as RealDNAInput
        from bioreason.models.dl.processing_dl import DLProcessor as RealDLProcessor
        from bioreason.models.dl.chat_template_dl import CHAT_TEMPLATE as RealCHAT_TEMPLATE
        
        # Only override if real components are available
        DNAInput = RealDNAInput
        DLProcessor = RealDLProcessor
        CHAT_TEMPLATE = RealCHAT_TEMPLATE
        
        # Try to import Evo2 components
        try:
            from bioreason.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer
            register_evo2_tokenizer()
            EVO2_AVAILABLE = True
        except ImportError:
            EVO2_AVAILABLE = False
            print("Warning: Evo2 tokenizer not available")
        
        DNA_LLM_AVAILABLE = True
        print("✅ DNA-LLM components loaded successfully")
        
    except ImportError as e:
        print(f"Warning: DNA-LLM components not available: {e}")
        print("DNA functionality will be disabled - using fallback classes")

def load_protein_components():
    """Load protein components only when needed."""
    global ProteinInput, PLProcessor, PROTEIN_CHAT_TEMPLATE
        
    try:
        from bioreason2.models.protein_llm import ProteinLLMModel
        from bioreason2.utils.protein_utils import ProteinInput as RealProteinInput
        from bioreason2.models.pl.processing_pl import PLProcessor as RealPLProcessor  
        from bioreason2.models.pl.chat_template_pl import CHAT_TEMPLATE as RealProteinCHAT_TEMPLATE
        
        
        # ESM3 imports for protein processing
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, SamplingConfig
        from esm.utils.constants.models import ESM3_OPEN_SMALL
        
        # Only override if real components are available
        ProteinInput = RealProteinInput
        PLProcessor = RealPLProcessor
        PROTEIN_CHAT_TEMPLATE = RealProteinCHAT_TEMPLATE
        
        print("✅ Protein-LLM components loaded successfully")
        
    except ImportError as e:
        print(f"Warning: Protein-LLM components not available: {e}")
        print("Protein functionality will be disabled - using fallback classes")

CHAT_TEMPLATE = "{%- set dna_count = namespace(value=0) %}{%- set protein_count = namespace(value=0) %}{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content is string and message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' }} {%- if message.content is string %}{{- message.content + '<|im_end|>' + '\\n' }}{%- else %}{%- for content in message.content %}{%- if content.type == 'dna' or 'dna' in content %}{%- set dna_count.value = dna_count.value + 1 %}{%- if add_dna_id %}DNA Sequence {{- dna_count.value }}: {%- endif %}<|dna_start|><|dna_pad|><|dna_end|>{%- elif content.type == 'protein' or 'protein' in content %}{%- set protein_count.value = protein_count.value + 1 %}{%- if add_protein_id %}Protein Sequence {{- protein_count.value }}: {%- endif %}<|protein_start|><|protein_pad|><|protein_end|>{%- elif 'text' in content %}{{- content.text }}{%- endif %}{%- endfor %}{{- '<|im_end|>' + '\\n' }}{%- endif %}{%- elif message.role == \"assistant\" %}\n        {%- set content = message.content[0].text %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in message.content %}\n                {%- set content = message.content[0].text.split('</think>')[-1].lstrip('\\n') %}\n                {%- set reasoning_content = message.content[0].text.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"

EVO2_AVAILABLE = False

# Fallback classes when components aren't available
class FallbackDNAInput:
    def __init__(self, *args, **kwargs):
        pass

class FallbackDLProcessor:
    def __init__(self, *args, **kwargs):
        pass

class FallbackProteinInput:
    def __init__(self, *args, **kwargs):
        pass

class FallbackPLProcessor:
    def __init__(self, *args, **kwargs):
        pass

# Initialize fallback variables
DNAInput = FallbackDNAInput
DLProcessor = FallbackDLProcessor
ProteinInput = FallbackProteinInput
PLProcessor = FallbackPLProcessor
PROTEIN_CHAT_TEMPLATE = None


if is_fastapi_available():
    from fastapi import FastAPI
else:
    FastAPI = None

if is_pydantic_available():
    from pydantic import BaseModel
else:
    BaseModel = None

if is_uvicorn_available():
    import uvicorn
else:
    uvicorn = None

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.utils import get_open_port

    if is_vllm_ascend_available():
        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator as PyNcclCommunicator
else:
    LLM = None
    SamplingParams = None
    PyNcclCommunicator = None
    get_world_group = None
    StatelessProcessGroup = None
    GuidedDecodingParams = None
    get_open_port = None


logger = logging.getLogger(__name__)

# We use CUDA with multiprocessing, so we must use the 'spawn' start method
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "0"


class WeightSyncWorkerExtension:
    """
    A vLLM worker extension that enables weight synchronization between a client and multiple server workers.
    """

    # The following attributes are initialized when `init_communicator` method is called.
    pynccl_comm = None  # Communicator for weight updates
    client_rank = None  # Source rank for broadcasting updated weights

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        """Initialize the weight update communicator using a stateless process group."""
        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")

        # Get the rank of the current worker in the global world group.
        rank = get_world_group().rank

        # Create a stateless process group to manage communication
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)

        # Initialize the NCCL-based communicator for weight synchronization.
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

        # The client process that sends updated weights has the highest rank
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype, shape: Sequence[int]) -> None:
        """Receive updated weights from the client process and update the named parameter in the model."""
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call `init_communicator` first.")

        # Allocate memory for the incoming weight tensor on the correct device.
        weight = torch.empty(shape, dtype=dtype, device=self.device)

        # Use NCCL to broadcast the updated weights from the client (src) to all workers.
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()

        # Load the received weights into the model.
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        """Close the communicator when weight synchronization is no longer needed."""
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


@dataclass
class ScriptArguments:
    """Arguments for the vLLM serve script."""
    model: str = field(metadata={"help": "Model name or path to load the model from."})
    revision: Optional[str] = field(default=None, metadata={"help": "Revision to use for the model."})
    tensor_parallel_size: int = field(default=1, metadata={"help": "Number of tensor parallel workers to use."})
    data_parallel_size: int = field(default=1, metadata={"help": "Number of data parallel workers to use."})
    host: str = field(default="0.0.0.0", metadata={"help": "Host address to run the server on."})
    port: int = field(default=8000, metadata={"help": "Port to run the server on."})
    gpu_memory_utilization: float = field(default=0.8, metadata={"help": "GPU memory utilization ratio for vLLM."})
    dtype: str = field(default="auto", metadata={"help": "Data type to use for vLLM generation."})
    max_model_len: Optional[int] = field(default=None, metadata={"help": "Maximum model length for vLLM."})
    enable_prefix_caching: Optional[bool] = field(default=None, metadata={"help": "Whether to enable prefix caching in vLLM."})
    enforce_eager: Optional[bool] = field(default=False, metadata={"help": "Whether to enforce eager execution."})
    kv_cache_dtype: str = field(default="auto", metadata={"help": "Data type to use for KV cache."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether to trust remote code when loading models."})
    batch_inference: bool = field(default=False, metadata={"help": "Whether to enable batch inference."})
    # Optional custom tokenizer path or name.  If provided, vLLM will load
    # this tokenizer instead of the one bundled with the model.  Use this to
    # point to a folder that contains `tokenizer.json` created with
    # `AutoTokenizer(..., use_fast=True).save_pretrained(...)` in order to
    # enable the much faster Rust tokenizer for models that ship without it.
    tokenizer: Optional[str] = field(default=None, metadata={"help": "Tokenizer name or path to use (defaults to model's tokenizer)."})
    log_level: str = field(default="info", metadata={"help": "Log level for uvicorn."})
    
    # DNA-specific parameters
    dna_model_name: Optional[str] = field(default=None, metadata={"help": "DNA model name or path for multimodal DNA+text generation."})
    dna_is_evo2: bool = field(default=False, metadata={"help": "Whether the DNA model is Evo2."})
    dna_embedding_layer: Optional[str] = field(default=None, metadata={"help": "Name of the layer to use for the Evo2 model."})
    max_length_dna: int = field(default=2048, metadata={"help": "Maximum length of DNA sequences."})
    use_dna_llm: bool = field(default=False, metadata={"help": "Whether to enable DNA processing for multimodal DNA+text generation."})
    
    # Protein-specific parameters
    protein_model_name: Optional[str] = field(default="esm3_sm_open_v1", metadata={"help": "Protein model name or path for multimodal protein+text generation."})
    max_length_protein: int = field(default=2048, metadata={"help": "Maximum length of protein sequences."})
    use_protein_llm: bool = field(default=False, metadata={"help": "Whether to enable protein processing for multimodal protein+text generation."})
    
    # Environment validation
    skip_env_check: bool = field(default=False, metadata={"help": "Skip the fuckvllm environment validation check."})


def llm_worker(script_args: ScriptArguments, data_parallel_rank: int, master_port: int, connection: Connection) -> None:
    """Worker process for handling vLLM generation with optional DNA/protein processing."""
    # Set required environment variables for DP to work with vLLM
    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(script_args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Ensure a fast tokenizer is available. If script_args.tokenizer is not
    # supplied and <model_dir>/tokenizer.json does not exist we auto-create
    # one on the fly (once) so vLLM will always use the Rust tokenizer.
    # ------------------------------------------------------------------

    def ensure_fast_tok(model_dir: str) -> str:
        """Return path that contains tokenizer.json; build it if necessary."""
        # If user explicitly provided --tokenizer we do nothing.
        if script_args.tokenizer:
            return script_args.tokenizer

        model_path = Path(model_dir)
        if (model_path / "tokenizer.json").exists():
            return str(model_path)  # already fast

        fast_dir = model_path / "fast_tok"
        if not (fast_dir / "tokenizer.json").exists():
            fast_dir.mkdir(exist_ok=True)
            try:
                from transformers import AutoTokenizer
                print(f"🪄 Building fast tokenizer in {fast_dir} …")
                tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)
                tok.save_pretrained(fast_dir)
                print("✅ Fast tokenizer created")
            except Exception as e:
                print(f"⚠️ Could not create fast tokenizer automatically: {e}")
                # Fallback to slow tokenizer path
                return str(model_path)
        return str(fast_dir)

    tokenizer_path = ensure_fast_tok(script_args.model)

    # Always use standard vLLM for hosting
    llm = LLM(
        model=script_args.model,
        tokenizer=tokenizer_path,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        enforce_eager=script_args.enforce_eager,
        dtype=script_args.dtype,
        enable_prefix_caching=script_args.enable_prefix_caching,
        kv_cache_dtype=script_args.kv_cache_dtype,
        max_model_len=script_args.max_model_len,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
        trust_remote_code=script_args.trust_remote_code,
        enable_prompt_embeds=True,
    )

    # Initialize biological sequence processing components (DNA OR protein, not both)
    dna_processor = None
    protein_processor = None
    
    # Check for conflicting biological sequence processing requests
    dna_requested = script_args.use_dna_llm and script_args.dna_model_name
    protein_requested = script_args.use_protein_llm and script_args.protein_model_name
    
    if dna_requested and protein_requested:
        print("⚠️  WARNING: Both DNA and protein processing requested!")
        print("⚠️  Only one biological sequence type can be processed at a time.")
        print("⚠️  Prioritizing DNA processing and disabling protein processing.")
        print("⚠️  To use protein processing, set --use_dna_llm=False")
        protein_requested = False
    
    # Initialize DNA processing components if needed
    if dna_requested:
        try:
            # Load DNA components only when needed
            load_dna_components()
            dna_processor = DNAEmbeddingProcessor(
                dna_model_name=script_args.dna_model_name,
                text_model_name=script_args.model,
                dna_is_evo2=script_args.dna_is_evo2,
                dna_embedding_layer=script_args.dna_embedding_layer,
                max_length_dna=script_args.max_length_dna,
                device=device,
            )
            print("✅ DNA processor initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize DNA processor: {e}")
            dna_processor = None

    # Initialize protein processing components if needed (only if DNA not requested)
    elif protein_requested:
        try:
            # Load protein components only when needed
            load_protein_components()
            protein_processor = ProteinEmbeddingProcessor(
                protein_model_name=script_args.protein_model_name,
                text_model_name=script_args.model,
                max_length_protein=script_args.max_length_protein,
                device=device,
            )
            print("✅ Protein processor initialized successfully")
            protein_processor.batch_inference = script_args.batch_inference
        except Exception as e:
            print(f"❌ Failed to initialize protein processor: {e}")
            protein_processor = None
    
    # Log the final configuration
    if dna_processor:
        print("🧬 Worker configured for DNA sequence processing")
    elif protein_processor:
        print("🧬 Worker configured for protein sequence processing")
    else:
        print("🧬 Worker configured for text-only processing")

    # Send ready signal to parent process
    connection.send({"status": "ready"})

    while True:
        try:
            command = connection.recv()
        except KeyboardInterrupt:
            if hasattr(llm, 'collective_rpc'):
                llm.collective_rpc(method="close_communicator")
            break

        # Handle commands
        if command["type"] in ["call", "fire_and_forget"]:
            method_name = command["method"]
            args, kwargs = command.get("args", ()), command.get("kwargs", {})
            
            # NEW: direct prompt_embeds path -----------------------------------------------------
            if method_name == "generate" and "prompt_embeds" in kwargs:
                # kwargs["prompt_embeds"] is either a single 2-D list or a list of 2-D lists (batched)
                embeds_payload = kwargs.pop("prompt_embeds")
                sampling_params = kwargs.get("sampling_params", SamplingParams()) if SamplingParams is not None else kwargs.get("sampling_params")

                # Normalise to list of embeddings
                if isinstance(embeds_payload[0][0], (float, int)):
                    embeds_payload = [embeds_payload]  # single sample -> list of one

                outputs_all = []
                for emb_idx, emb_list in enumerate(embeds_payload):
                    emb_tensor = torch.tensor(emb_list, dtype=torch.float16, device="cuda:0")  # cast to fp16 to save memory
                    out = llm.generate({"prompt_embeds": emb_tensor}, sampling_params=sampling_params)
                    outputs_all.extend(out)
                result = outputs_all
            # -----------------------------------------------------------------------------------
            elif method_name == "generate" and dna_processor is not None:
                result = generate_with_dna_embeddings(llm, dna_processor, kwargs, device)
            # Handle protein-enhanced generation
            elif method_name == "generate" and protein_processor is not None:
                result = generate_with_protein_embeddings(llm, protein_processor, kwargs, device)
            else:
                # Standard vLLM handling (including pre-processed embeddings)
                method = getattr(llm, method_name)
                result = method(*args, **kwargs)
                
            if command["type"] == "call":
                connection.send(result)
        elif command["type"] == "shutdown":
            break


def chunk_list(lst: list, n: int) -> list:
    """Split list `lst` into `n` evenly distributed sublists."""
    k, r = divmod(len(lst), n)
    return [lst[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


class DNAEmbeddingProcessor:
    """
    DNA embedding processor that replicates DNALLMModel's DNA processing functionality.
    Based exactly on the reference DNALLMModel implementation.
    """
    
    def __init__(
        self,
        dna_model_name: str,
        text_model_name: str,
        dna_is_evo2: bool = False,
        dna_embedding_layer: Optional[str] = None,
        max_length_dna: int = 2048,
        device: str = "cpu",
    ):
        if not DNA_LLM_AVAILABLE:
            raise RuntimeError("DNA-LLM components not available. Please install bioreason package.")
        
        self.device = device
        print(f"🧬 Initializing DNAEmbeddingProcessor on device {self.device}...")
        print(f"  DNA model: {dna_model_name}")
        print(f"  Text model: {text_model_name}")
        print(f"  Evo2: {dna_is_evo2}")
        print(f"  Embedding layer: {dna_embedding_layer}")
        
        self.dna_is_evo2 = dna_is_evo2
        self.dna_embedding_layer = dna_embedding_layer
        self.max_length_dna = max_length_dna
        
        # STEP 1: Load DNA model and tokenizer (exactly like DNALLMModel)
        print("🧬 Loading DNA model and tokenizer...")
        if not self.dna_is_evo2:
            self.dna_model = AutoModelForMaskedLM.from_pretrained(
                dna_model_name, trust_remote_code=True
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
            self.dna_config = self.dna_model.config
        else:
            if not EVO2_AVAILABLE:
                raise ImportError("Evo2 is required when dna_is_evo2=True. Please install the evo2 package.")
            try:
                from evo2 import Evo2
                self.dna_model = Evo2(dna_model_name)
                self.dna_tokenizer = Evo2Tokenizer(self.dna_model.tokenizer)
                self.dna_config = self.dna_model.model.config
            except ImportError:
                raise ImportError("Evo2 is required when dna_is_evo2=True. Please install the evo2 package.")
        
        self.dna_model = self.dna_model.to(self.device)
        print("✅ DNA model loaded successfully")

        # STEP 2: Load text model config and tokenizer (exactly like DNALLMModel)
        print("📝 Loading text tokenizer...")
        # Use the fast (Rust) tokenizer for much higher throughput during local preprocessing
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, trust_remote_code=True, use_fast=True)
        self.text_config = AutoConfig.from_pretrained(text_model_name, trust_remote_code=True)

        # Use the same chat template as DNALLMModel (EXACT match)
        if CHAT_TEMPLATE:
            self.text_tokenizer.chat_template = CHAT_TEMPLATE
        # self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Add DNA tokens exactly like DNALLMModel
        new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.dna_token_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_pad|>")

        print("✅ Text tokenizer loaded successfully")

        # STEP 3: Create projection layer (exactly like DNALLMModel)
        print("🔗 Creating DNA projection layer...")
        self.text_hidden_size = self.text_config.hidden_size
        self.dna_hidden_size = self.dna_config.hidden_size
        self.dna_projection = nn.Linear(self.dna_hidden_size, self.text_hidden_size)
        self.dna_projection = self.dna_projection.to(self.device)
        print(f"✅ Projection layer created: {self.dna_hidden_size} -> {self.text_hidden_size}")

        # STEP 4: Load custom components (exactly like DNALLMModel)
        print("🔧 Loading custom projection weights...")
        self.load_custom_components(text_model_name)

        # STEP 5: Create processor (exactly like DNALLMModel)
        self.processor = DLProcessor(tokenizer=self.text_tokenizer, dna_tokenizer=self.dna_tokenizer)

        # STEP 6: Create minimal embedding layer for text embeddings
        self._embedding_layer = None
        self._text_model_name = text_model_name
        
        # Initialize the embedding layer immediately to catch any issues early
        print("🧬 Pre-initializing embedding layer...")
        try:
            # Create a dummy input to trigger embedding layer creation
            dummy_input = torch.tensor([[0]], dtype=torch.long)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            _ = self.get_text_embeddings(dummy_input)
            print("✅ Embedding layer pre-initialization successful")
        except Exception as e:
            print(f"⚠️ Warning: Embedding layer pre-initialization failed: {e}")
            print("   Will retry when needed")

        print("✅ DNAEmbeddingProcessor initialized successfully")
    
    def load_custom_components(self, llm_dir: str) -> None:
        """Load trained DNA projection weights (exactly like DNALLMModel)."""
        import os
        
        # Try to load DNA projection layer weights
        projection_path = os.path.join(llm_dir, 'dna_projection.pt')
        if os.path.exists(projection_path):
            print(f"🔧 Loading trained DNA projection weights from {projection_path}")
            try:
                projection_state = torch.load(projection_path, map_location='cpu')
                
                # Check if we can load the weights
                if self.dna_projection.weight.shape == projection_state['weight'].shape:
                    self.dna_projection.load_state_dict(projection_state)
                    print("✅ Trained DNA projection weights loaded successfully")
                else:
                    print(f"⚠️ Projection layer shape mismatch!")
                    print(f"  Expected: {self.dna_projection.weight.shape}")
                    print(f"  Found: {projection_state['weight'].shape}")
                    print("  Using randomly initialized projection layer")
            except Exception as e:
                print(f"⚠️ Error loading projection weights: {e}")
                print("  Using randomly initialized projection layer")
        else:
            print(f"⚠️ No trained DNA projection weights found at {projection_path}")
            print("  Using randomly initialized projection layer (may affect quality)")
        
        # Check if there's a local DNA model (optional)
        dna_model_path = os.path.join(llm_dir, 'dna_model')
        if os.path.exists(dna_model_path) and not self.dna_is_evo2:
            print(f"📁 Found local DNA model at {dna_model_path}")
            try:
                # Replace the DNA model with the local one
                self.dna_model = AutoModelForMaskedLM.from_pretrained(dna_model_path, trust_remote_code=True)
                self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_path, trust_remote_code=True)
                self.dna_config = self.dna_model.config
                # CRITICAL: Move the local DNA model to the correct device
                self.dna_model = self.dna_model.to(self.device)
                print("✅ Local DNA model loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading local DNA model: {e}")
                print("  Using original DNA model")
    
    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings using a minimal embedding layer (vLLM mode only)."""
        # For vLLM mode, we need to create a minimal embedding layer without loading the full model
        if not hasattr(self, '_embedding_layer') or self._embedding_layer is None:
            print("⚠️ vLLM mode: Creating minimal embedding layer for DNA integration...")
            
            # Load only the embedding layer from the model config
            embed_dim = self.text_config.hidden_size
            vocab_size = self.text_config.vocab_size
            
            print(f"🧬 Creating embedding layer: vocab_size={vocab_size}, embed_dim={embed_dim}")
            
            # Ensure we're working with valid dimensions
            if embed_dim <= 0 or vocab_size <= 0:
                raise ValueError(f"Invalid embedding dimensions: vocab_size={vocab_size}, embed_dim={embed_dim}")
            
            # Create the embedding layer
            try:
                self._embedding_layer = nn.Embedding(vocab_size, embed_dim)
                print(f"🧬 Successfully created embedding layer: {self._embedding_layer}")
            except Exception as e:
                print(f"❌ Failed to create embedding layer: {e}")
                raise RuntimeError(f"Failed to create nn.Embedding: {e}")
            
            # Verify the embedding layer was created
            if self._embedding_layer is None:
                raise RuntimeError("Embedding layer creation returned None")
            
            # Try to load embedding weights from the saved model if available
            try:
                import os
                model_path = self._text_model_name  # Use the text model name path
                print(f"🧬 Looking for embedding weights in: {model_path}")
                
                if model_path and os.path.exists(model_path):
                    # Load just the embedding weights from safetensors
                    try:
                        import safetensors
                        from safetensors import safe_open
                        
                        # Try safetensors first
                        safetensors_path = os.path.join(model_path, "model.safetensors")
                        if os.path.exists(safetensors_path):
                            print(f"🧬 Loading from single safetensors file...")
                            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                                embed_weights = f.get_tensor("model.embed_tokens.weight")
                                self._embedding_layer.weight.data = embed_weights
                                print("✅ Loaded embedding weights from safetensors")
                        else:
                            # Try multiple safetensors files (for sharded models)
                            import glob
                            shard_files = glob.glob(os.path.join(model_path, "model-*.safetensors"))
                            print(f"🧬 Found {len(shard_files)} shard files")
                            if shard_files:
                                for shard_file in shard_files:
                                    try:
                                        with safe_open(shard_file, framework="pt", device="cpu") as f:
                                            if "model.embed_tokens.weight" in f.keys():
                                                embed_weights = f.get_tensor("model.embed_tokens.weight")
                                                self._embedding_layer.weight.data = embed_weights
                                                print("✅ Loaded embedding weights from sharded safetensors")
                                                break
                                    except Exception as e:
                                        print(f"⚠️ Error loading from {shard_file}: {e}")
                                        continue
                    except Exception as e:
                        print(f"⚠️ Could not load from safetensors: {e}")
                        print("   Using randomly initialized embeddings")
                else:
                    print(f"⚠️ Model path does not exist: {model_path}")
                    print("   Using randomly initialized embeddings")
            except Exception as e:
                print(f"⚠️ Could not load embedding weights: {e}")
                print("   Using randomly initialized embeddings")
            
            # Move to same device as input - be careful not to lose the reference
            print(f"🧬 Moving embedding layer to device: {input_ids.device}")
            try:
                if input_ids.device.type != 'cpu':
                    self._embedding_layer = self._embedding_layer.to(input_ids.device)
                print(f"🧬 Embedding layer ready: {self._embedding_layer}")
                print(f"🧬 Embedding layer type: {type(self._embedding_layer)}")
                print(f"🧬 Embedding layer device: {next(self._embedding_layer.parameters()).device}")
            except Exception as e:
                print(f"❌ Error moving embedding layer to device: {e}")
                raise RuntimeError(f"Failed to move embedding layer to device: {e}")
        
        # Final verification before use
        if self._embedding_layer is None:
            raise RuntimeError("CRITICAL: _embedding_layer is None after initialization")
            
        if not callable(self._embedding_layer):
            raise RuntimeError(f"CRITICAL: Embedding layer is not callable - type: {type(self._embedding_layer)}")
        
        if not hasattr(self._embedding_layer, 'weight'):
            raise RuntimeError(f"CRITICAL: Embedding layer has no weight attribute - type: {type(self._embedding_layer)}")
            
        # Ensure input is on the same device as the embedding layer
        embedding_device = next(self._embedding_layer.parameters()).device
        print(f"🧬 Input device: {input_ids.device}, Embedding device: {embedding_device}")
        
        # Move input to the correct device if needed
        if input_ids.device != embedding_device:
            print(f"🧬 Moving input from {input_ids.device} to {embedding_device}")
            input_ids = input_ids.to(embedding_device)
        
        print(f"🧬 Calling embedding layer with input shape: {input_ids.shape}")
        
        try:
            result = self._embedding_layer(input_ids)
            print(f"🧬 Embedding layer output shape: {result.shape}")
            return result
        except Exception as e:
            print(f"❌ Error calling embedding layer: {e}")
            print(f"❌ Embedding layer: {self._embedding_layer}")
            print(f"❌ Input shape: {input_ids.shape}")
            print(f"❌ Input dtype: {input_ids.dtype}")
            raise RuntimeError(f"Failed to call embedding layer: {e}")

    def process_dna_embeddings(
        self,
        dna_tokenized: Dict[str, torch.Tensor],
        batch_idx_map: List[int],
        batch_size: int,
    ) -> List[torch.Tensor]:
        """
        Process DNA sequences to obtain embeddings.

        Args:
            dna_tokenized: Tokenized DNA sequences
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch

        Returns:
            List of tensor embeddings for each batch item
        """
        # Process all sequences to get DNA representations
        with torch.no_grad():
            # Handle different model types based on dna_is_evo2 attribute
            if self.dna_is_evo2 and self.dna_embedding_layer is not None:  # Evo2 model
                # Get embeddings from the specific layer in Evo2
                hidden_states_list = []
                
                for seq_idx in range(len(dna_tokenized["input_ids"])):
                    # Extract single sequence
                    input_ids = dna_tokenized["input_ids"][seq_idx:seq_idx+1]
                    
                    # Call Evo2 with return_embeddings=True
                    _, embeddings = self.dna_model(
                        input_ids,
                        return_embeddings=True,
                        layer_names=[self.dna_embedding_layer]
                    )
                    
                    # Get embeddings for the specified layer
                    seq_embeddings = embeddings[self.dna_embedding_layer].squeeze(0)
                    hidden_states_list.append(seq_embeddings)
                
                # Stack to get same format as non-Evo2 output
                if hidden_states_list:
                    hidden_states = torch.stack(hidden_states_list)
                else:
                    return [torch.zeros((0, self.text_hidden_size)) for _ in range(batch_size)]
                    
            else:  # Standard HuggingFace model
                # Use existing code path for HF models
                curr_device = next(self.dna_model.parameters()).device
                target_device = torch.device(self.device)
                if curr_device != target_device:
                    print(f"🛠️  Moving DNA model from {curr_device} to {target_device}")
                    self.dna_model = self.dna_model.to(target_device)

                outputs = self.dna_model(
                    input_ids=dna_tokenized["input_ids"].to(self.device),
                    attention_mask=dna_tokenized["attention_mask"].to(self.device),
                    output_hidden_states=True,
                )
                # Get the last hidden state
                hidden_states = outputs.hidden_states[-1]  # shape: [n_seqs, seq_len, hidden_dim]

        # Project all embeddings at once
        hidden_states = hidden_states.to(device=self.dna_projection.weight.device, dtype=self.dna_projection.weight.dtype)
        projected_states = self.dna_projection(hidden_states)

        # Group embeddings by batch item - use proper typing
        result: List[torch.Tensor] = []

        # For each batch item, collect its embeddings
        for batch_idx in range(batch_size):
            batch_embeddings = []
            for seq_idx, seq_batch_idx in enumerate(batch_idx_map):
                if seq_batch_idx == batch_idx:
                    # Get only the valid (non-padding) tokens
                    valid_length = dna_tokenized["attention_mask"][seq_idx].sum().item()
                    seq_embedding = projected_states[seq_idx, :valid_length]
                    batch_embeddings.append(seq_embedding)

            # Concatenate embeddings for this batch item
            if batch_embeddings:
                result.append(torch.cat(batch_embeddings, dim=0))
            else:
                result.append(torch.zeros((0, self.text_hidden_size)))

        return result


class ProteinEmbeddingProcessor:
    """
    Protein embedding processor that replicates ProteinLLMModel's protein processing functionality.
    Based exactly on the reference ProteinLLMModel implementation.
    """
    
    def __init__(
        self,
        protein_model_name: str,
        text_model_name: str,
        max_length_protein: int = 2048,
        device: str = "cpu",
    ):
        
        self.device = device
        print(f"🧬 Initializing ProteinEmbeddingProcessor on device {self.device}...")
        print(f"  Protein model: {protein_model_name}")
        print(f"  Text model: {text_model_name}")
        
        self.max_length_protein = max_length_protein
        
        # STEP 1: Load protein model (exactly like ProteinLLMModel)
        print("🧬 Loading protein model...")
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, SamplingConfig
        from esm.utils.constants.models import ESM3_OPEN_SMALL
        
        self.protein_model = ESM3.from_pretrained(protein_model_name)
        self.protein_model = self.protein_model.to(self.device)
        print("✅ Protein model loaded successfully")

        # STEP 2: Load text model config and tokenizer (exactly like ProteinLLMModel)
        print("📝 Loading text tokenizer...")
        # Use the fast tokenizer – critical for preprocessing speed
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, trust_remote_code=True, use_fast=True)
        self.text_config = AutoConfig.from_pretrained(text_model_name, trust_remote_code=True)

        # Use the same chat template as ProteinLLMModel (EXACT match)
        if PROTEIN_CHAT_TEMPLATE:
            self.text_tokenizer.chat_template = PROTEIN_CHAT_TEMPLATE
        # self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Add protein tokens exactly like ProteinLLMModel
        new_tokens = ["<|protein_start|>", "<|protein_pad|>", "<|protein_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.protein_token_id = self.text_tokenizer.convert_tokens_to_ids("<|protein_pad|>")

        print("✅ Text tokenizer loaded successfully")

        # STEP 3: Create projection layer (exactly like ProteinLLMModel)
        print("🔗 Creating protein projection layer...")
        self.text_hidden_size = self.text_config.hidden_size
        # ESM3 embedding dimension - typically 2560 for ESM3_OPEN_SMALL (same as protein_llm.py)
        self.protein_hidden_size = self.protein_model.encoder.sequence_embed.embedding_dim
        self.protein_projection = nn.Linear(self.protein_hidden_size, self.text_hidden_size)
        self.protein_projection = self.protein_projection.to(self.device)
        print(f"✅ Projection layer created: {self.protein_hidden_size} -> {self.text_hidden_size}")

        # STEP 4: Load custom components (exactly like ProteinLLMModel)
        print("🔧 Loading custom projection weights...")
        self.load_custom_components(text_model_name)

        # STEP 5: Create processor (exactly like ProteinLLMModel)
        self.processor = PLProcessor(tokenizer=self.text_tokenizer)

        # STEP 6: Create minimal embedding layer for text embeddings
        self._embedding_layer = None
        self._text_model_name = text_model_name
        
        # Initialize the embedding layer immediately to catch any issues early
        print("🧬 Pre-initializing embedding layer...")
        try:
            # Create a dummy input to trigger embedding layer creation
            dummy_input = torch.tensor([[0]], dtype=torch.long)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
            _ = self.get_text_embeddings(dummy_input)
            print("✅ Embedding layer pre-initialization successful")
        except Exception as e:
            print(f"⚠️ Warning: Embedding layer pre-initialization failed: {e}")
            print("   Will retry when needed")

        print("✅ ProteinEmbeddingProcessor initialized successfully")
    
    def load_custom_components(self, llm_dir: str) -> None:
        """Load trained protein projection weights (exactly like ProteinLLMModel)."""
        import os
        
        # Try to load protein projection layer weights
        projection_path = os.path.join(llm_dir, 'protein_projection.pt')
        if os.path.exists(projection_path):
            print(f"🔧 Loading trained protein projection weights from {projection_path}")
            try:
                projection_state = torch.load(projection_path, map_location='cpu')
                
                # Check if we can load the weights
                if self.protein_projection.weight.shape == projection_state['weight'].shape:
                    self.protein_projection.load_state_dict(projection_state)
                    print("✅ Trained protein projection weights loaded successfully")
                else:
                    print(f"⚠️ Projection layer shape mismatch!")
                    print(f"  Expected: {self.protein_projection.weight.shape}")
                    print(f"  Found: {projection_state['weight'].shape}")
                    print("  Using randomly initialized projection layer")
            except Exception as e:
                print(f"⚠️ Error loading projection weights: {e}")
                print("  Using randomly initialized projection layer")
        else:
            print(f"⚠️ No trained protein projection weights found at {projection_path}")
            print("  Using randomly initialized projection layer (may affect quality)")
        
        # Check if there's a local protein model (optional)
        protein_model_path = os.path.join(llm_dir, 'protein_model')
        if os.path.exists(protein_model_path):
            print(f"📁 Found local protein model at {protein_model_path}")
            try:
                # Replace the protein model with the local one
                from esm.models.esm3 import ESM3
                self.protein_model = ESM3.from_pretrained(protein_model_path)
                self.protein_model = self.protein_model.to(self.device)
                print("✅ Local protein model loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading local protein model: {e}")
                print("  Using original protein model")
    
    def get_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get text embeddings using a minimal embedding layer (vLLM mode only)."""
        # For vLLM mode, we need to create a minimal embedding layer without loading the full model
        if not hasattr(self, '_embedding_layer') or self._embedding_layer is None:
            print("⚠️ vLLM mode: Creating minimal embedding layer for protein integration...")
            
            # Load only the embedding layer from the model config
            embed_dim = self.text_config.hidden_size
            vocab_size = self.text_config.vocab_size
            
            print(f"🧬 Creating embedding layer: vocab_size={vocab_size}, embed_dim={embed_dim}")
            
            # Ensure we're working with valid dimensions
            if embed_dim <= 0 or vocab_size <= 0:
                raise ValueError(f"Invalid embedding dimensions: vocab_size={vocab_size}, embed_dim={embed_dim}")
            
            # Create the embedding layer
            try:
                self._embedding_layer = nn.Embedding(vocab_size, embed_dim)
                print(f"🧬 Successfully created embedding layer: {self._embedding_layer}")
            except Exception as e:
                print(f"❌ Failed to create embedding layer: {e}")
                raise RuntimeError(f"Failed to create nn.Embedding: {e}")
            
            # Verify the embedding layer was created
            if self._embedding_layer is None:
                raise RuntimeError("Embedding layer creation returned None")
            
            # Try to load embedding weights from the saved model if available
            try:
                import os
                model_path = self._text_model_name  # Use the text model name path
                print(f"🧬 Looking for embedding weights in: {model_path}")
                
                if model_path and os.path.exists(model_path):
                    # Load just the embedding weights from safetensors
                    try:
                        import safetensors
                        from safetensors import safe_open
                        
                        # Try safetensors first
                        safetensors_path = os.path.join(model_path, "model.safetensors")
                        if os.path.exists(safetensors_path):
                            print(f"🧬 Loading from single safetensors file...")
                            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                                embed_weights = f.get_tensor("model.embed_tokens.weight")
                                self._embedding_layer.weight.data = embed_weights
                                print("✅ Loaded embedding weights from safetensors")
                        else:
                            # Try multiple safetensors files (for sharded models)
                            import glob
                            shard_files = glob.glob(os.path.join(model_path, "model-*.safetensors"))
                            print(f"🧬 Found {len(shard_files)} shard files")
                            if shard_files:
                                for shard_file in shard_files:
                                    try:
                                        with safe_open(shard_file, framework="pt", device="cpu") as f:
                                            if "model.embed_tokens.weight" in f.keys():
                                                embed_weights = f.get_tensor("model.embed_tokens.weight")
                                                self._embedding_layer.weight.data = embed_weights
                                                print("✅ Loaded embedding weights from sharded safetensors")
                                                break
                                    except Exception as e:
                                        print(f"⚠️ Error loading from {shard_file}: {e}")
                                        continue
                    except Exception as e:
                        print(f"⚠️ Could not load from safetensors: {e}")
                        print("   Using randomly initialized embeddings")
                else:
                    print(f"⚠️ Model path does not exist: {model_path}")
                    print("   Using randomly initialized embeddings")
            except Exception as e:
                print(f"⚠️ Could not load embedding weights: {e}")
                print("   Using randomly initialized embeddings")
            
            # Move to same device as input - be careful not to lose the reference
            print(f"🧬 Moving embedding layer to device: {input_ids.device}")
            try:
                if input_ids.device.type != 'cpu':
                    self._embedding_layer = self._embedding_layer.to(input_ids.device)
                print(f"🧬 Embedding layer ready: {self._embedding_layer}")
                print(f"🧬 Embedding layer type: {type(self._embedding_layer)}")
                print(f"🧬 Embedding layer device: {next(self._embedding_layer.parameters()).device}")
            except Exception as e:
                print(f"❌ Error moving embedding layer to device: {e}")
                raise RuntimeError(f"Failed to move embedding layer to device: {e}")
        
        # Final verification before use
        if self._embedding_layer is None:
            raise RuntimeError("CRITICAL: _embedding_layer is None after initialization")
            
        if not callable(self._embedding_layer):
            raise RuntimeError(f"CRITICAL: Embedding layer is not callable - type: {type(self._embedding_layer)}")
        
        if not hasattr(self._embedding_layer, 'weight'):
            raise RuntimeError(f"CRITICAL: Embedding layer has no weight attribute - type: {type(self._embedding_layer)}")
            
        # Ensure input is on the same device as the embedding layer
        embedding_device = next(self._embedding_layer.parameters()).device
        print(f"🧬 Input device: {input_ids.device}, Embedding device: {embedding_device}")
        
        # Move input to the correct device if needed
        if input_ids.device != embedding_device:
            print(f"🧬 Moving input from {input_ids.device} to {embedding_device}")
            input_ids = input_ids.to(embedding_device)
        
        print(f"🧬 Calling embedding layer with input shape: {input_ids.shape}")
        
        try:
            result = self._embedding_layer(input_ids)
            print(f"🧬 Embedding layer output shape: {result.shape}")
            return result
        except Exception as e:
            print(f"❌ Error calling embedding layer: {e}")
            print(f"❌ Embedding layer: {self._embedding_layer}")
            print(f"❌ Input shape: {input_ids.shape}")
            print(f"❌ Input dtype: {input_ids.dtype}")
            raise RuntimeError(f"Failed to call embedding layer: {e}")

    def process_protein_embeddings(
        self,
        protein_sequences: List[str],
        batch_idx_map: List[int],
        batch_size: int,
        structure_coords: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """
        Process protein sequences to obtain embeddings using ESM3.

        Args:
            protein_sequences: List of protein sequence strings
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch

        Returns:
            List of tensor embeddings for each batch item
        """
        # Initialize result list
        result = [[] for _ in range(batch_size)]

        # Process each protein sequence individually (ESM3 doesn't support batching)
        for seq_idx, sequence in enumerate(protein_sequences):
            # Truncate sequence if too long
            if len(sequence) > self.max_length_protein:
                sequence = sequence[: self.max_length_protein]
            
            if structure_coords is not None and structure_coords.shape[1] != 0:
                coords = structure_coords[batch_idx_map[seq_idx]].cpu.float()

                if coords.shape[0] == len(sequence):
                    protein = ESMProtein(sequence=sequence, coords=coords)
                else:
                    if seq_idx == 0:  # Only log once to avoid spam
                        print(
                            f"⚠️  Length mismatch: structure ({coords.shape[0]}) vs sequence ({len(sequence)}) - using sequence only"
                        )
                    protein = ESMProtein(sequence=sequence)
            else:
                protein = ESMProtein(sequence=sequence)


            # Encode protein
            protein_tensor = self.protein_model.encode(protein)
            protein_tensor = protein_tensor.to(self.device)

            # Get embeddings - respect finetune parameter
            with torch.set_grad_enabled(False):
                output = self.protein_model.forward_and_sample(
                    protein_tensor,
                    SamplingConfig(return_per_residue_embeddings=True),
                )
                seq_embeddings = output.per_residue_embedding

            # Get the batch index for this sequence
            batch_idx = batch_idx_map[seq_idx]

            # Add to appropriate batch result
            result[batch_idx].append(seq_embeddings)


        

        # Concatenate embeddings for each batch item
        for i in range(batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                # Empty tensor for batch items with no proteins - ensure on device
                result[i] = torch.zeros((0, self.protein_hidden_size), device=self.device)

        # Project all embeddings to text embedding space
        for i in range(batch_size):
            if result[i].numel() > 0:  # Check if tensor is not empty
                result[i] = result[i].to(
                    device=self.protein_projection.weight.device,
                    dtype=self.protein_projection.weight.dtype,
                )
                result[i] = self.protein_projection(result[i])
            else:
                # Ensure empty tensors have correct dimensions
                result[i] = torch.zeros(
                    (0, self.text_hidden_size),
                    device=self.protein_projection.weight.device,
                    dtype=self.protein_projection.weight.dtype,
                )


        return result


def generate_with_protein_embeddings(llm, protein_processor, kwargs, device):
    """Generate using standard vLLM with protein embeddings - EXACTLY matches DNA logic."""
    print(f"🧬 generate_with_protein_embeddings called with kwargs keys: {kwargs.keys()}")
    
    if "inputs" in kwargs:
        # Protein+text inputs format
        inputs = kwargs["inputs"]
        print(f"🧬 Processing {len(inputs)} input samples")
        
        # STEP 1: Extract text and protein sequences from inputs (EXACTLY like DNA)
        batch_text = []
        batch_protein_sequences = []
        
        for inp in inputs:
            text = inp["text"]
            protein_sequences = inp.get("protein_sequences", [])
            
            batch_text.append(text)
            batch_protein_sequences.append(protein_sequences)
        
        print(f"🧬 Prepared batch with {len(batch_text)} text items and {len(batch_protein_sequences)} protein sequence lists")
        for i, protein_seqs in enumerate(batch_protein_sequences):
            print(f"🧬 Sample {i}: has {len(protein_seqs)} protein sequences")
        
        # STEP 2: Process using PLProcessor (EXACTLY like DNA with DLProcessor)
        print(f"🧬 Calling PLProcessor with text and batch_protein_sequences...")
        print(f"🧬 Text sample: {batch_text[0][:200]}..." if batch_text[0] else "🧬 Empty text")
        
        processed = protein_processor.processor(
            text=batch_text,
            batch_protein_sequences=batch_protein_sequences,
            max_length_text=2048,
            max_length_protein=2048,
            return_tensors="pt"
        )

        structure_coords = processed.get("structure_coords")
        
        
        
        # Get input_ids and attention_mask
        input_ids = processed["input_ids"].to(device)
        attention_mask = processed["attention_mask"].to(device)
        
        print(f"🧬 Input IDs shape: {input_ids.shape}")
        print(f"🧬 Attention mask shape: {attention_mask.shape}")
        for b in range(input_ids.shape[0]):
            decoded = protein_processor.text_tokenizer.decode(
                input_ids[b][attention_mask[b].bool()],
                skip_special_tokens=False
            )
            print(f"[{b}] endswith assistant? ",
                decoded.strip().endswith("<|im_start|>assistant"))
            print(decoded[-200:])
            print("length of decoded: ", len(decoded))
            
        # Check if we have protein data
        protein_sequences_batch = processed.get("protein_sequences")
        batch_idx_map = processed.get("batch_idx_map")
        
        if protein_sequences_batch is not None and len(protein_sequences_batch) > 0:
            print(f"🧬 ✅ Protein data provided - processing protein embeddings...")
            print(f"🧬 Protein sequences count: {len(protein_sequences_batch)}")
            print(f"🧬 Batch idx map: {batch_idx_map}")
            
            batch_size = input_ids.shape[0]
            
            # STEP 3: Process protein embeddings using EXACT same logic as ProteinLLMModel.generate
            protein_embeddings = protein_processor.process_protein_embeddings(
                protein_sequences_batch,
                batch_idx_map,
                batch_size,
                structure_coords=structure_coords
            )
            
            # STEP 4: Get text embeddings (EXACT same as ProteinLLMModel.generate)
            print(f"🧬 About to call get_text_embeddings with input_ids shape: {input_ids.shape}")
            text_embeddings = protein_processor.get_text_embeddings(input_ids)
            print(f"🧬 Text embeddings shape: {text_embeddings.shape}")
            
            # STEP 5: Integrate protein embeddings (EXACT same logic as ProteinLLMModel.generate)
            mask = input_ids == protein_processor.protein_token_id
            n_protein_tokens = mask.sum().item()
            protein_embeds_flat = torch.cat(protein_embeddings, dim=0)
            n_protein_features = protein_embeds_flat.shape[0]
            
            print(f"🧬 Found {n_protein_tokens} protein tokens in text")
            print(f"🧬 Generated {n_protein_features} protein features")

            if n_protein_features != n_protein_tokens:
                raise ValueError(
                    f"Protein features and protein tokens do not match: features {n_protein_features}, tokens: {n_protein_tokens}"
                )

            # Ensure protein embeddings have the same dtype as the text embeddings
            protein_embeds_flat = protein_embeds_flat.to(dtype=text_embeddings.dtype, device=text_embeddings.device)
            
            print(f"🧬 Before protein replacement - text embeds mean: {text_embeddings.mean().item():.4f}")
            text_embeddings[mask] = protein_embeds_flat
            print(f"🧬 After protein replacement - text embeds mean: {text_embeddings.mean().item():.4f}")
            print(f"🧬 Protein successfully integrated into text embeddings!")
           
            
            # STEP 6: Generate using prompt embeddings with vLLM (EXACT same as ProteinLLMModel.generate)
            

            from copy import deepcopy

            base_sampling_params = kwargs.get("sampling_params", SamplingParams())
            
            all_outputs = []
            
            print(f"🧬 ======= PROMPT EMBEDS ANALYSIS =======")
            print(f"🧬 Input text_embeddings shape: {text_embeddings.shape}")
            print(f"🧬 Input text_embeddings dtype: {text_embeddings.dtype}")
            print(f"🧬 Input text_embeddings device: {text_embeddings.device}")
            print(f"🧬 Input batch_size: {batch_size}")
            
            # EXACT same logic as ProteinLLMModel.generate - handle 3D vs 2D tensors
            if text_embeddings.dim() == 3 and not protein_processor.batch_inference:
                print(f"🧬 3D tensor detected - processing each batch item separately")
                
                for batch_idx in range(text_embeddings.shape[0]):
                    single_embeddings = text_embeddings[batch_idx]  # (seq_len, hidden_size)
    
                    print(f"🧬 Generating for batch {batch_idx} with embeddings shape: {single_embeddings.shape}")
                    sampling_params = kwargs.get(
                        "sampling_params",
                        SamplingParams(temperature=0.7, top_p=1.0, max_tokens=5120)
                    )

                    sparams = deepcopy(base_sampling_params)

                    
                    with torch.no_grad():
                        batch_outputs = llm.generate(
                            {"prompt_embeds": single_embeddings},
                            sparams
                        )
                    all_outputs.extend(batch_outputs)
            elif text_embeddings.dim() == 3 and protein_processor.batch_inference:
                print(f"🧬 3D tensor detected - processing batch inference")
                #make it a list of length batch_size
                sparams = deepcopy(base_sampling_params)
                text_embeddings = [text_embeddings[i] for i in range(batch_size)]
                with torch.no_grad():
                    sparams = deepcopy(base_sampling_params)
                    padded_embeddings_list = [text_embeddings[i] for i in range(batch_size)]
                    print(f"   [DEBUG] Converted 3D tensor to a list of {len(padded_embeddings_list)} padded tensors.")

                    print("🧬 Trimming each tensor in the list...")
                    
                    # 2. Create a NEW list to hold the correctly-sized, trimmed tensors.
                    trimmed_embeddings_list = []
                    
                    # 3. Loop through each PADDED tensor in the list.
                    for i in range(batch_size):
                        # 4. Get the true, un-padded length from the attention_mask for this item.
                        actual_length = attention_mask[i].sum().item()
                        
                        # 5. Get the padded tensor from the list.
                        padded_tensor = padded_embeddings_list[i]
                        
                        # 6. Slice THIS 2D TENSOR to its actual length.
                        trimmed_tensor = padded_tensor[:actual_length]
                        
                        # 7. Add the trimmed tensor to our new list.
                        trimmed_embeddings_list.append(trimmed_tensor)

                        print(f"   [Batch {i}] Padded shape: {padded_tensor.shape}, Trimmed shape: {trimmed_tensor.shape}")
                    with torch.no_grad():
                        print("*"*20)
                        print(f"sparams: {sparams}")
                        
                        # 4. Pass the list of *trimmed* embedding tensors to vLLM
                        all_outputs = llm.generate(
                            [{"prompt_embeds": emb} for emb in trimmed_embeddings_list],
                            sparams
                        )
                    print("*"*20)
                    print(f"sparams: {sparams}")
                    # for i in range(batch_size):
                    #     print(f"text_embeddings[{i}]: {text_embeddings[i].shape}")
                    # #save the text embeddings to a txt file
                    # with open("text_embeddings.txt", "w") as f:
                    #     for i in range(batch_size):
                    #         f.write(f"text_embeddings[{i}]: {text_embeddings[i].shape}\n")
                    #         f.write(f"text_embeddings[{i}]: {text_embeddings[i]}\n")
                    #         f.write("\n")
                    
                    # all_outputs = llm.generate(
                    #     [{"prompt_embeds": text_embeddings[i]} for i in range(batch_size)],
                    #     sparams
                    # )
            elif text_embeddings.dim() == 2:
                # Single item format
                print(f"🧬 2D tensor detected - single item format")
                with torch.no_grad():
                    all_outputs = llm.generate(
                        {"prompt_embeds": text_embeddings},
                        sampling_params
                    )
            else:
                raise ValueError(f"Unexpected embedding dimensions: {text_embeddings.dim()}D")
            torch.cuda.empty_cache()
            return all_outputs
        else:
            print(f"🧬 No protein data provided - using text-only generation")
            # Text-only generation - convert input_ids to prompts for vLLM
            prompts = []
            for batch_idx in range(input_ids.shape[0]):
                # Find non-padding tokens
                non_pad = (input_ids[batch_idx] != protein_processor.text_tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad) > 0:
                    start_idx = non_pad[0].item()
                    # Decode to get the prompt text
                    prompt_text = protein_processor.text_tokenizer.decode(
                        input_ids[batch_idx, start_idx:], 
                        skip_special_tokens=False
                    ).strip()
                    prompts.append(prompt_text)
                else:
                    prompts.append("")  # Handle empty case
            
            sampling_params = kwargs.get("sampling_params", SamplingParams())
            print(f"🧬 Text-only generation for {len(prompts)} prompts")
            return llm.generate(prompts, sampling_params)
    else:
        # Standard vLLM prompts format - no protein processing needed
        prompts = kwargs.get("prompts", [])
        sampling_params = kwargs.get("sampling_params", SamplingParams())
        print(f"🧬 Standard vLLM generation for {len(prompts)} prompts")
        return llm.generate(prompts, sampling_params)


def generate_with_dna_embeddings(llm, dna_processor, kwargs, device):
    """Generate using standard vLLM with DNA embeddings - EXACTLY matches test_kegg_checkpoint.py logic."""
    print(f"🧬 generate_with_dna_embeddings called with kwargs keys: {kwargs.keys()}")
    
    if "inputs" in kwargs:
        # DNA+text inputs format
        inputs = kwargs["inputs"]
        print(f"🧬 Processing {len(inputs)} input samples")
        
        # STEP 1: Extract text and DNA sequences from inputs (EXACTLY like test_kegg_checkpoint.py)
        batch_text = []
        batch_dna_sequences = []
        
        for inp in inputs:
            text = inp["text"]
            dna_sequences = inp.get("dna_sequences", [])
            
            batch_text.append(text)
            batch_dna_sequences.append(dna_sequences)
        
        print(f"🧬 Prepared batch with {len(batch_text)} text items and {len(batch_dna_sequences)} DNA sequence lists")
        for i, dna_seqs in enumerate(batch_dna_sequences):
            print(f"🧬 Sample {i}: has {len(dna_seqs)} DNA sequences")
        
        # STEP 2: Process using DLProcessor (EXACTLY like test_kegg_checkpoint.py and DNALLMModel)
        # This is the key - the processor expects text as first arg, batch_dna_sequences as kwarg
        print(f"🧬 Calling DLProcessor with text and batch_dna_sequences...")
        print(f"🧬 Text sample: {batch_text[0][:200]}..." if batch_text[0] else "🧬 Empty text")
        
        processed = dna_processor.processor(
            text=batch_text,
            batch_dna_sequences=batch_dna_sequences,
            max_length_text=2048,  # Match test_kegg_checkpoint.py
            max_length_dna=2048,
            return_tensors="pt"
        )
        
        # Get input_ids and attention_mask
        input_ids = processed["input_ids"].to(device)
        attention_mask = processed["attention_mask"].to(device)
        
        print(f"🧬 Input IDs shape: {input_ids.shape}")
        print(f"🧬 Attention mask shape: {attention_mask.shape}")
        
        # Check if we have DNA data
        dna_tokenized = processed.get("dna_tokenized")
        batch_idx_map = processed.get("batch_idx_map")
        
        if dna_tokenized is not None and len(dna_tokenized.input_ids) > 0:
            print(f"🧬 ✅ DNA data provided - processing DNA embeddings...")
            print(f"🧬 DNA tokenized input_ids shape: {dna_tokenized.input_ids.shape}")
            print(f"🧬 Batch idx map: {batch_idx_map}")
            
            batch_size = input_ids.shape[0]
            
            # STEP 3: Process DNA embeddings using EXACT same logic as DNALLMModel.generate
            dna_embeddings = dna_processor.process_dna_embeddings(
                {
                    "input_ids": dna_tokenized.input_ids,
                    "attention_mask": dna_tokenized.attention_mask
                },
                batch_idx_map,
                batch_size
            )
            
            # STEP 4: Get text embeddings (EXACT same as DNALLMModel.generate)
            print(f"🧬 About to call get_text_embeddings with input_ids shape: {input_ids.shape}")
            print(f"🧬 Embedding layer status: {hasattr(dna_processor, '_embedding_layer')} / {getattr(dna_processor, '_embedding_layer', None)}")
            text_embeddings = dna_processor.get_text_embeddings(input_ids)
            print(f"🧬 Text embeddings shape: {text_embeddings.shape}")
            
            # STEP 5: Integrate DNA embeddings (EXACT same logic as DNALLMModel.generate)
            mask = input_ids == dna_processor.dna_token_id
            n_dna_tokens = mask.sum().item()
            dna_embeds_flat = torch.cat(dna_embeddings, dim=0)
            n_dna_features = dna_embeds_flat.shape[0]
            
            print(f"🧬 Found {n_dna_tokens} DNA tokens in text")
            print(f"🧬 Generated {n_dna_features} DNA features")

            if n_dna_features != n_dna_tokens:
                raise ValueError(
                    f"DNA features and DNA tokens do not match: features {n_dna_features}, tokens: {n_dna_tokens}"
                )

            # Ensure DNA embeddings have the same dtype as the text embeddings
            dna_embeds_flat = dna_embeds_flat.to(dtype=text_embeddings.dtype, device=text_embeddings.device)
            
            print(f"🧬 Before DNA replacement - text embeds mean: {text_embeddings.mean().item():.4f}")
            text_embeddings[mask] = dna_embeds_flat
            print(f"🧬 After DNA replacement - text embeds mean: {text_embeddings.mean().item():.4f}")
            print(f"🧬 DNA successfully integrated into text embeddings!")
            
            # STEP 6: Generate using prompt embeddings with vLLM (EXACT same as DNALLMModel.generate)
            from copy import deepcopy

            base_sampling_params = kwargs.get("sampling_params", SamplingParams())
            all_outputs = []
            
            print(f"🧬 ======= PROMPT EMBEDS ANALYSIS =======")
            print(f"🧬 Input text_embeddings shape: {text_embeddings.shape}")
            print(f"🧬 Input text_embeddings dtype: {text_embeddings.dtype}")
            print(f"🧬 Input text_embeddings device: {text_embeddings.device}")
            print(f"🧬 Input batch_size: {batch_size}")

            print(f"🧬 text_embeddings: {text_embeddings}")
            
            # EXACT same logic as DNALLMModel.generate - handle 3D vs 2D tensors
            if text_embeddings.dim() == 3:
                print(f"🧬 3D tensor detected - processing each batch item separately")
                
                for batch_idx in range(text_embeddings.shape[0]):
                    single_embeddings = text_embeddings[batch_idx]  # (seq_len, hidden_size)
                    print(f"🧬 Generating for batch {batch_idx} with embeddings shape: {single_embeddings.shape}")
                    sparams = deepcopy(base_sampling_params)
                    print(f"🧬 sparams: {sparams}")
                    with torch.no_grad():
                        batch_outputs = llm.generate(
                            {"prompt_embeds": single_embeddings},
                            sparams
                        )
                    all_outputs.extend(batch_outputs)
            elif text_embeddings.dim() == 2:
                # Single item format
                print(f"🧬 2D tensor detected - single item format")
                with torch.no_grad():
                    all_outputs = llm.generate(
                        {"prompt_embeds": text_embeddings},
                        sampling_params
                    )
            else:
                raise ValueError(f"Unexpected embedding dimensions: {text_embeddings.dim()}D")
            torch.cuda.empty_cache()
            return all_outputs
        else:
            print(f"🧬 No DNA data provided - using text-only generation")
            # Text-only generation - convert input_ids to prompts for vLLM
            prompts = []
            for batch_idx in range(input_ids.shape[0]):
                # Find non-padding tokens
                non_pad = (input_ids[batch_idx] != dna_processor.text_tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad) > 0:
                    start_idx = non_pad[0].item()
                    # Decode to get the prompt text
                    prompt_text = dna_processor.text_tokenizer.decode(
                        input_ids[batch_idx, start_idx:], 
                        skip_special_tokens=False
                    ).strip()
                    prompts.append(prompt_text)
                else:
                    prompts.append("")  # Handle empty case
            
            sampling_params = kwargs.get("sampling_params", SamplingParams())
            print(f"🧬 Text-only generation for {len(prompts)} prompts")
            return llm.generate(prompts, sampling_params)
    else:
        # Standard vLLM prompts format - no DNA processing needed
        prompts = kwargs.get("prompts", [])
        sampling_params = kwargs.get("sampling_params", SamplingParams())
        print(f"🧬 Standard vLLM generation for {len(prompts)} prompts")
        return llm.generate(prompts, sampling_params)


def main(script_args: ScriptArguments):
    """Main function to start the vLLM serve with DNA support."""
    # Check that we're running in the correct environment
    import sys
    
    def check_environment():
        """Verify we're running in the fuckvllm environment."""
        current_env = None
        
        # Check conda environment
        conda_env = os.environ.get('CONDA_DEFAULT_ENV')
        if conda_env:
            current_env = conda_env
            print(f"🌍 Running in conda environment: {conda_env}")
        
        # Check virtual environment
        virtual_env = os.environ.get('VIRTUAL_ENV')
        if virtual_env:
            env_name = os.path.basename(virtual_env)
            current_env = env_name
            print(f"🌍 Running in virtual environment: {env_name}")
        
        # Check if we're in the expected environment
        if current_env != 'fuckvllm':
            print(f"⚠️  WARNING: Expected to run in 'fuckvllm' environment, but currently in: {current_env or 'unknown'}")
            print(f"⚠️  Python executable: {sys.executable}")
            print(f"⚠️  Consider activating the fuckvllm environment first:")
            print(f"     conda activate fuckvllm")
            print(f"     # or")
            print(f"     source fuckvllm/bin/activate")
            
            # Give user a chance to abort
            try:
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    print("❌ Aborting. Please activate the fuckvllm environment first.")
                    sys.exit(1)
            except (KeyboardInterrupt, EOFError):
                print("\n❌ Aborting. Please activate the fuckvllm environment first.")
                sys.exit(1)
        else:
            print(f"✅ Running in correct environment: {current_env}")
    
    # Validate environment (unless skipped)
    if not script_args.skip_env_check:
        check_environment()
    else:
        print("⚠️  Environment check skipped via --skip_env_check flag")
    
    # Check required dependencies
    if not is_fastapi_available():
        raise ImportError("FastAPI is required. Install with: pip install fastapi")
    if not is_pydantic_available():
        raise ImportError("Pydantic is required. Install with: pip install pydantic")
    if not is_uvicorn_available():
        raise ImportError("Uvicorn is required. Install with: pip install uvicorn")
    if not is_vllm_available():
        raise ImportError("vLLM is required. Install with: pip install vllm")

    # CRITICAL: Set multiprocessing start method BEFORE any CUDA operations
    import multiprocessing
    will_use_protein_processing = script_args.use_protein_llm and script_args.protein_model_name
    
    if will_use_protein_processing:
        # When using CUDA in main process (for protein processing), we need spawn method
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print("🔧 Set multiprocessing start method to 'spawn' for CUDA compatibility")
        except RuntimeError:
            print("⚠️ Multiprocessing start method already set")

    # Spawn dp workers, and setup pipes for communication
    master_port = get_open_port()
    connections = []
    processes = []
    for data_parallel_rank in range(script_args.data_parallel_size):
        parent_connection, child_connection = Pipe()
        process = Process(target=llm_worker, args=(script_args, data_parallel_rank, master_port, child_connection))
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Wait for all workers to send "ready"
        ready_connections = set()
        while len(ready_connections) < script_args.data_parallel_size:
            for connection in connections:
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)

        yield

        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Process {process} is still alive after 10 seconds, attempting to terminate...")
                process.terminate()
                process.join()

    app = FastAPI(lifespan=lifespan)

    # Health check endpoint
    @app.get("/health/")
    async def health():
        """Health check endpoint to verify that the server is running."""
        return {
            "status": "ok",
            "dna_llm_available": DNA_LLM_AVAILABLE,
            "dna_processing_enabled": script_args.use_dna_llm and script_args.dna_model_name,
            "dna_model": script_args.dna_model_name if script_args.dna_model_name else None,
            "protein_llm_available": True,
            "protein_processing_enabled": script_args.use_protein_llm and script_args.protein_model_name,
            "protein_model": script_args.protein_model_name if script_args.protein_model_name else None,
            "text_model": script_args.model,
            "evo2_available": EVO2_AVAILABLE
        }

    @app.get("/get_world_size/")
    async def get_world_size():
        """Retrieves the world size of the LLM engine."""
        return {"world_size": script_args.tensor_parallel_size * script_args.data_parallel_size}

    class GenerateRequest(BaseModel):
        prompts: List[str]  # Text prompts (will be formatted as chat messages if needed)
        dna_sequences: Optional[List[List[str]]] = None  # List of DNA sequences per prompt (outer list length must match prompts)
        protein_sequences: Optional[List[List[str]]] = None  # List of protein sequences per prompt (outer list length must match prompts)
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 5120
        guided_decoding_regex: Optional[str] = None
        generation_kwargs: Dict = field(default_factory=dict)
        
        def model_post_init(self, __context) -> None:
            """Validate that dna_sequences and protein_sequences length matches prompts length if provided."""
            if self.dna_sequences is not None:
                if len(self.dna_sequences) != len(self.prompts):
                    raise ValueError(
                        f"Length of dna_sequences ({len(self.dna_sequences)}) must match "
                        f"length of prompts ({len(self.prompts)})"
                    )
                # Validate that each dna_sequences item is a list
                for i, dna_seqs in enumerate(self.dna_sequences):
                    if not isinstance(dna_seqs, list):
                        raise ValueError(
                            f"dna_sequences[{i}] must be a list of strings, got {type(dna_seqs)}"
                        )
            
            if self.protein_sequences is not None:
                if len(self.protein_sequences) != len(self.prompts):
                    raise ValueError(
                        f"Length of protein_sequences ({len(self.protein_sequences)}) must match "
                        f"length of prompts ({len(self.prompts)})"
                    )
                # Validate that each protein_sequences item is a list
                for i, protein_seqs in enumerate(self.protein_sequences):
                    if not isinstance(protein_seqs, list):
                        raise ValueError(
                            f"protein_sequences[{i}] must be a list of strings, got {type(protein_seqs)}"
                        )
            
            # Ensure only one type of biological sequence is provided
            if self.dna_sequences is not None and self.protein_sequences is not None:
                raise ValueError("Cannot provide both dna_sequences and protein_sequences in the same request")

    class GenerateResponse(BaseModel):
        completion_ids: List[List[int]]
        completions: List[str]
        # Speed metrics
        generation_time: float  # Total generation time in seconds
        tokens_per_second: float  # Average tokens per second
        total_tokens: int  # Total tokens generated

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        """Generate completions for the provided prompts, with optional DNA or protein sequences."""
        import time
        start_time = time.time()
        
        print(f"🧬 API received request:")
        print(f"🧬   - {len(request.prompts)} prompts")
        print(f"🧬   - DNA sequences: {'Yes' if request.dna_sequences else 'No'}")
        print(f"🧬   - Protein sequences: {'Yes' if request.protein_sequences else 'No'}")
        if request.dna_sequences:
            total_seqs = sum(len(seqs) for seqs in request.dna_sequences)
            print(f"🧬   - Total DNA sequences: {total_seqs}")
        if request.protein_sequences:
            total_seqs = sum(len(seqs) for seqs in request.protein_sequences)
            print(f"🧬   - Total protein sequences: {total_seqs}")
        print(f"🧬   - Temperature: {request.temperature}")
        print(f"🧬   - Max tokens: {request.max_tokens}")
        
        # Check if we're using DNA processing
        if request.dna_sequences and script_args.use_dna_llm and script_args.dna_model_name:
            print(f"🧬 Using DNA processing mode")
            result = await generate_with_dna_processing(request)
        # Check if we're using protein processing
        elif request.protein_sequences and script_args.use_protein_llm and script_args.protein_model_name:
            print(f"🧬 Using protein processing mode")
            result = await generate_with_protein_processing(request)
        else:
            print(f"🧬 Using standard vLLM mode")
            result = await generate_with_vllm(request)
        
        # Calculate timing metrics
        end_time = time.time()
        generation_time = end_time - start_time
        total_tokens = sum(len(tokens) for tokens in result["completion_ids"])
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
        
        # Log speed metrics
        print(f"🚀 Generation completed:")
        print(f"🚀   - Total time: {generation_time:.3f}s")
        print(f"🚀   - Total tokens: {total_tokens}")
        print(f"🚀   - Speed: {tokens_per_second:.2f} tokens/sec")
        
        # Add metrics to response
        result["generation_time"] = generation_time
        result["tokens_per_second"] = tokens_per_second
        result["total_tokens"] = total_tokens
        
        return result

    # ----------------------------------------------------------------------
    #  DNA-aware generation  ― no more DNAInput wrappers
    # ----------------------------------------------------------------------
    async def generate_with_dna_processing(request: GenerateRequest):
        """
        Same behaviour as generate_with_vllm, but optionally injects per-prompt
        DNA sequences that will be turned into embeddings inside DNALLMModel.

        • request.prompts         : List[str]  (length = B)
        • request.dna_sequences   : List[List[str]] | None
                                    Outer list is length B, inner is any length.
        """
        # Build SamplingParams for vLLM
        generation_kwargs = {
            "temperature":          request.temperature,
            "top_p":                request.top_p,
            "top_k":                request.top_k,
            "min_p":                request.min_p,
            "max_tokens":           request.max_tokens,
            "repetition_penalty":   request.repetition_penalty,
        }
        if request.guided_decoding_regex:
            generation_kwargs["guided_decoding"] = GuidedDecodingParams(
                backend="outlines",
                regex=request.guided_decoding_regex,
            )
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Split prompts (and DNA) evenly across DP ranks
        chunked_prompts     = chunk_list(request.prompts,       script_args.data_parallel_size)
        chunked_dna_seqs    = chunk_list(request.dna_sequences, script_args.data_parallel_size) \
                                if request.dna_sequences else [[] for _ in range(script_args.data_parallel_size)]

        # ------------------------------------------------------------------
        # Dispatch to workers
        # ------------------------------------------------------------------
        for conn, prompts_this_rank, dna_this_rank in zip(connections, chunked_prompts, chunked_dna_seqs):

            # If no real work for this rank, send a placeholder so vLLM won't error.
            if not prompts_this_rank:
                print(f"🧬 No real work for this rank, sending placeholder")
                prompts_this_rank, dna_this_rank = ["<placeholder>"], [[]]

            # ---- build the per-example payload -------------------------------------------------------
            #   DLProcessor expects:
            #       inputs[i]["text"]          : str (formatted chat message, not None!)
            #       inputs[i]["dna_sequences"] : List[str] (may be [])
            # ------------------------------------------------------------------------------------------
            inputs = []
            for i, prompt in enumerate(prompts_this_rank):
                # Ensure prompt is not None
                if prompt is None:
                    print(f"⚠️ Warning: prompt at index {i} is None, using empty string")
                    prompt = ""
                
                # Get DNA sequences for this prompt (ensure it's a list of strings)
                dna_seqs = dna_this_rank[i] if i < len(dna_this_rank) else []
                if not isinstance(dna_seqs, list):
                    print(f"⚠️ Warning: dna_seqs at index {i} is not a list: {type(dna_seqs)}, converting to []")
                    dna_seqs = []
                
                # Format the prompt properly for the DLProcessor using the chat template
                # Use the text tokenizer's chat template to format the conversation
                formatted_text = prompt
                if not prompt.startswith("<|im_start|>"):
                    # If not already formatted, format as a chat conversation
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        tok = get_main_text_tokenizer(script_args.model)
                        msg = {
                            "role": "user",
                            "content": (
                                [{"type": "dna"} for _ in dna_seqs]      # one stub per sequence
                                + [{"type": "text", "text": prompt}]     # the human question
                            ),
                        }
                        # Use the processor's text tokenizer apply_chat_template method
                        tok = get_main_text_tokenizer(script_args.model)
                        msg = {
                            "role": "user",
                            "content": (
                                [{"type": "dna"} for _ in dna_seqs]      # one stub per sequence
                                + [{"type": "text", "text": prompt}]     # the human question
                            ),
                        }

                        formatted_text = tok.apply_chat_template(
                            [msg],
                            tokenize=False,
                            add_generation_prompt=True,             
                        )
                        
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to apply chat template: {e}")
                        # Fallback to simple format
                        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                inputs.append({
                    "text": formatted_text,
                    "dna_sequences": dna_seqs
                })

            
            print(f"🧬 inputs: {inputs}")

            conn.send(
                {
                    "type":   "call",
                    "method": "generate",
                    "kwargs": {
                        "inputs":          inputs,          # << minimalistic now
                        "sampling_params": sampling_params,
                    },
                }
            )

        # ------------------------------------------------------------------
        # Gather & flatten results
        # ------------------------------------------------------------------
        raw_outputs   = [conn.recv() for conn in connections]
        raw_outputs   = [o for o, p in zip(raw_outputs, chunked_prompts) if p]   # drop placeholder ranks
        raw_outputs   = list(chain.from_iterable(raw_outputs))

        completion_ids = [
            list(output.token_ids)
            for req_out in raw_outputs
            for output in req_out.outputs
        ]
        
        completions = [
            output.text
            for req_out in raw_outputs
            for output in req_out.outputs
        ]
        
        # Debug logging for completions
        print(f"🧬 Generated {len(completions)} completions:")
        for i, completion in enumerate(completions):
            print(f"   Completion {i+1} (length={len(completion)}): {completion[:100]}{'...' if len(completion) > 100 else ''}")

        return {"completion_ids": completion_ids, "completions": completions}

    async def generate_with_protein_processing(request: GenerateRequest):
        """
        Same behaviour as generate_with_vllm, but optionally injects per-prompt
        protein sequences that will be turned into embeddings inside ProteinLLMModel.

        • request.prompts         : List[str]  (length = B)
        • request.protein_sequences   : List[List[str]] | None
                                    Outer list is length B, inner is any length.
        """
        # Build SamplingParams for vLLM
        generation_kwargs = {
            "temperature":          request.temperature,
            "top_p":                request.top_p,
            "top_k":                request.top_k,
            "min_p":                request.min_p,
            "max_tokens":           request.max_tokens,
            "repetition_penalty":   request.repetition_penalty,
        }
        if request.guided_decoding_regex:
            generation_kwargs["guided_decoding"] = GuidedDecodingParams(
                backend="outlines",
                regex=request.guided_decoding_regex,
            )
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Split prompts (and protein) evenly across DP ranks
        chunked_prompts     = chunk_list(request.prompts,       script_args.data_parallel_size)
        chunked_protein_seqs    = chunk_list(request.protein_sequences, script_args.data_parallel_size) \
                                if request.protein_sequences else [[] for _ in range(script_args.data_parallel_size)]

        # ------------------------------------------------------------------
        # Dispatch to workers
        # ------------------------------------------------------------------
        for conn, prompts_this_rank, protein_this_rank in zip(connections, chunked_prompts, chunked_protein_seqs):

            # If no real work for this rank, send a placeholder so vLLM won't error.
            if not prompts_this_rank:
                prompts_this_rank, protein_this_rank = ["<placeholder>"], [[]]

            # ---- build the per-example payload -------------------------------------------------------
            #   PLProcessor expects:
            #       inputs[i]["text"]          : str (formatted chat message, not None!)
            #       inputs[i]["protein_sequences"] : List[str] (may be [])
            # ------------------------------------------------------------------------------------------
            inputs = []
            for i, prompt in enumerate(prompts_this_rank):
                # Ensure prompt is not None
                if prompt is None:
                    print(f"⚠️ Warning: prompt at index {i} is None, using empty string")
                    prompt = ""
                
                # Get protein sequences for this prompt (ensure it's a list of strings)
                protein_seqs = protein_this_rank[i] if i < len(protein_this_rank) else []
                if not isinstance(protein_seqs, list):
                    print(f"⚠️ Warning: protein_seqs at index {i} is not a list: {type(protein_seqs)}, converting to []")
                    protein_seqs = []
                
                # Format the prompt properly for the PLProcessor using the chat template
                # Use the text tokenizer's chat template to format the conversation
                formatted_text = prompt
                if not prompt.startswith("<|im_start|>"):
                    # If not already formatted, format as a chat conversation
                    messages = [{"role": "user", "content": prompt}]
                    try:
                        
                        msg = {
                            "role": "user",
                            "content": (
                                [{"type": "protein"} for _ in protein_seqs]      # one stub per sequence
                                + [{"type": "text", "text": prompt}]     # the human question
                            ),
                        }
                        tok = get_main_text_tokenizer(script_args.model)

                        formatted_text = tok.apply_chat_template(
                            [msg],
                            tokenize=False,
                            add_generation_prompt=True,             
                        )
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to apply chat template: {e}")
                        # Fallback to simple format
                        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                inputs.append({
                    "text": formatted_text,
                    "protein_sequences": protein_seqs
                })

            
            print(f"🧬 inputs: {inputs}")

            conn.send(
                {
                    "type":   "call",
                    "method": "generate",
                    "kwargs": {
                        "inputs":          inputs,          # << minimalistic now
                        "sampling_params": sampling_params,
                    },
                }
            )

        # ------------------------------------------------------------------
        # Gather & flatten results
        # ------------------------------------------------------------------
        raw_outputs   = [conn.recv() for conn in connections]
        raw_outputs   = [o for o, p in zip(raw_outputs, chunked_prompts) if p]   # drop placeholder ranks
        raw_outputs   = list(chain.from_iterable(raw_outputs))

        completion_ids = [
            list(output.token_ids)
            for req_out in raw_outputs
            for output in req_out.outputs
        ]
        
        completions = [
            output.text
            for req_out in raw_outputs
            for output in req_out.outputs
        ]
        
        # Debug logging for completions
        print(f"🧬 Generated {len(completions)} completions:")
        for i, completion in enumerate(completions):
            print(f"   Completion {i+1} (length={len(completion)}): {completion[:100]}{'...' if len(completion) > 100 else ''}")

        return {"completion_ids": completion_ids, "completions": completions}

    async def generate_with_vllm(request: GenerateRequest):
        """Generate using standard vLLM (original logic)."""
        # Guided decoding, if enabled
        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(backend="outlines", regex=request.guided_decoding_regex)
        else:
            guided_decoding = None

        generation_kwargs = {
            "n": request.n,
            "repetition_penalty": request.repetition_penalty,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "guided_decoding": guided_decoding,
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        # Evenly distribute prompts across DP ranks
        chunked_prompts = chunk_list(request.prompts, script_args.data_parallel_size)

        # Send the prompts to each worker
        for connection, prompts in zip(connections, chunked_prompts):
            if not prompts:
                prompts = ["<placeholder>"]
            kwargs = {"prompts": prompts, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Receive results
        all_outputs = [connection.recv() for connection in connections]

        # Handle empty prompts
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_prompts) if prompts]

        # Flatten and combine all results
        all_outputs = list(chain.from_iterable(all_outputs))
        completion_ids = [list(output.token_ids) for outputs in all_outputs for output in outputs.outputs]
        completions = [output.text for outputs in all_outputs for output in outputs.outputs]
        return {"completion_ids": completion_ids, "completions": completions}

    # Additional endpoints (init_communicator, update_named_param, etc.)
    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        """Initialize the communicator for synchronizing model weights."""
        world_size = script_args.tensor_parallel_size * script_args.data_parallel_size + 1
        kwargs = {"method": "init_communicator", "args": (request.host, request.port, world_size)}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: List[int]

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        """Update the model weights with the provided tensor."""
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        kwargs = {"method": "update_named_param", "args": (request.name, dtype, tuple(request.shape))}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        """Reset the prefix cache for the model."""
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        all_outputs = [connection.recv() for connection in connections]
        success = all(output for output in all_outputs)
        return {"message": "Request received, resetting prefix cache status: " + str(success)}

    @app.post("/close_communicator/")
    async def close_communicator():
        """Close the weight update group and clean up associated resources."""
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "Request received, closing communicator"}

    class GenerateEmbedsRequest(BaseModel):
        """Request model for direct prompt embeddings generation."""
        prompt_embeds: List[List[List[float]]]  # List of (seq_len x hidden_size) embeddings per sample
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 5120
        repetition_penalty: float = 1.0
        generation_kwargs: Dict = field(default_factory=dict)

    class GenerateEmbedsResponse(BaseModel):
        completion_ids: List[List[int]]
        completions: List[str]
        generation_time: float
        tokens_per_second: float
        total_tokens: int

    @app.post("/generate_embeds/", response_model=GenerateEmbedsResponse)
    async def generate_embeds(request: GenerateEmbedsRequest):
        """Generate completions given *pre-computed* prompt embeddings."""
        import time
        start_time = time.time()

        # Build SamplingParams
        generation_kwargs = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "max_tokens": request.max_tokens,
            "repetition_penalty": request.repetition_penalty,
        }
        generation_kwargs.update(request.generation_kwargs)
        sampling_params = SamplingParams(**generation_kwargs) if SamplingParams is not None else None

        # Split embeddings evenly across DP ranks
        chunked_embeds = chunk_list(request.prompt_embeds, script_args.data_parallel_size)

        # Dispatch to workers
        for connection, embeds_chunk in zip(connections, chunked_embeds):
            if not embeds_chunk:
                embeds_chunk = [[[0.0]]]  # dummy
            kwargs = {"prompt_embeds": embeds_chunk, "sampling_params": sampling_params}
            connection.send({"type": "call", "method": "generate", "kwargs": kwargs})

        # Gather outputs
        all_outputs = [connection.recv() for connection in connections]
        all_outputs = [output for output, embeds in zip(all_outputs, chunked_embeds) if embeds]
        all_outputs = list(chain.from_iterable(all_outputs))

        completion_ids = [list(out.token_ids) for out in all_outputs for out in out.outputs]
        completions = [out.text for out in all_outputs for out in out.outputs]

        end_time = time.time()
        generation_time = end_time - start_time
        total_tokens = sum(len(ids) for ids in completion_ids)
        tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0.0

        return {
            "completion_ids": completion_ids,
            "completions": completions,
            "generation_time": generation_time,
            "tokens_per_second": tokens_per_second,
            "total_tokens": total_tokens,
        }

    # Start the server
    uvicorn.run(app, host=script_args.host, port=script_args.port, log_level=script_args.log_level)


def make_parser(subparsers=None):
    """Create argument parser for the vLLM serve script."""
    if subparsers is not None:
        parser = subparsers.add_parser("vllm-serve", help="Run the vLLM serve script", dataclass_types=ScriptArguments)
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    main(script_args)