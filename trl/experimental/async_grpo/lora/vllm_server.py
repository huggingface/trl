"""Custom vLLM server entry point with /update_lora endpoint for direct NCCL LoRA sync.

Uses vLLM as a library -- no vLLM source modifications required.
The ``LoRADirectSyncExtension`` worker extension is automatically registered.

Usage:
    python -m trl.experimental.async_grpo.lora.vllm_server --model <base> --enable-lora ...
"""

from __future__ import annotations

import asyncio
import logging
import signal
from argparse import Namespace

import uvloop
from pydantic import BaseModel
from vllm._version import version as vllm_version
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient

logger = logging.getLogger(__name__)


class LoRAUpdateRequest(BaseModel):
    manifest_json: str


class LoRASyncManager:
    """Manages the LoRA NCCL sync lifecycle on the server side."""

    def __init__(self, engine: AsyncLLM, engine_client: AsyncMPClient):
        self.engine = engine
        self.engine_client = engine_client
        self.update_lock = asyncio.Lock()

    async def init_sync_group(
        self,
        master_address: str,
        master_port: int,
        world_size: int,
        rank_offset: int,
    ) -> None:
        await self.engine_client.collective_rpc_async(
            "init_lora_sync_group",
            args=(master_address, master_port, world_size, rank_offset),
        )
        logger.info("LoRA sync group initialized on all workers")

    async def receive_lora_update(self, manifest_json: str) -> None:
        async with self.update_lock:
            logger.info("Pausing generation for LoRA update")
            await self.engine.pause_generation(mode="keep", clear_cache=False)
            try:
                await self.engine_client.collective_rpc_async(
                    "receive_lora_update",
                    args=(manifest_json,),
                )
            finally:
                await self.engine.resume_generation()
            logger.info("LoRA update complete, generation resumed")

    async def close(self) -> None:
        await self.engine_client.collective_rpc_async("close_lora_sync_group")


async def run_server(args: Namespace, **uvicorn_kwargs) -> None:
    logger.info("vLLM API server version %s (LoRA direct sync)", vllm_version)

    sock_addr = (args.host or "", args.port)
    from vllm.entrypoints.openai.api_server import create_server_socket

    sock = create_server_socket(sock_addr)
    set_ulimit()

    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = (
        "trl.experimental.async_grpo.lora.worker_extension.LoRADirectSyncExtension"
    )
    engine_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    engine = AsyncLLM.from_vllm_config(
        vllm_config=engine_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        disable_log_stats=engine_args.disable_log_stats,
        enable_log_requests=engine_args.enable_log_requests,
    )
    assert isinstance(engine.engine_core, AsyncMPClient)

    lora_sync_manager = LoRASyncManager(engine, engine.engine_core)

    supported_tasks = await engine.get_supported_tasks()
    model_config = engine.model_config if hasattr(engine, "model_config") else None
    app = build_app(args, supported_tasks, model_config)

    @app.post("/init_lora_sync_group")
    async def _init_lora_sync_group(request: dict):
        info = request["init_info"]
        await lora_sync_manager.init_sync_group(
            master_address=info["master_address"],
            master_port=info["master_port"],
            world_size=info["world_size"],
            rank_offset=info["rank_offset"],
        )
        return {"status": "ok"}

    @app.post("/update_lora")
    async def _update_lora(request: LoRAUpdateRequest):
        await lora_sync_manager.receive_lora_update(request.manifest_json)
        return {"status": "ok"}

    @app.get("/get_world_size")
    async def _get_world_size():
        return {"world_size": engine_config.parallel_config.world_size}

    await init_app_state(engine, app.state, args, supported_tasks)

    shutdown_task = await serve_http(
        app, sock=sock, host=args.host, port=args.port, **uvicorn_kwargs
    )

    try:
        await shutdown_task
    finally:
        await lora_sync_manager.close()
        sock.close()


def main() -> None:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible server with direct LoRA NCCL sync"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
