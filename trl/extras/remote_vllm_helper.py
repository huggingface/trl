"""
Easy vllm with nccl link with training process, build for LLM researchers

@Binary-Husky

launch vllm fastapi server (infer + nccl link)
    ```bash
    python trl/extras/remote_vllm_helper.py \
        --host-ip 127.0.0.1 \
        --master-port 51216 \
        --api-port 8000 \
        --nccl-link True \
        --model-path Qwen/Qwen/QwQ-32B \
        --dtype auto \
        --max-model-len 8192 \
        --num-gpus 4 \
        --gpu-memory-utilization 0.95 \
        --max-lora-rank 0 \
        --temperature 0.9 \
        --num-generations 1
    ```

client demo (infer + nccl link)
    ```python
    from trl.extras.remote_vllm_helper import VllmRemoteClient
    self.remote_llm = VllmRemoteClient(nccl_link=True)
    self.remote_llm.update_attr(key="temperature", value=args.temperature)
    self.remote_llm.update_attr(key="max_completion_length", value=self.max_completion_length)
    self.remote_llm.update_attr(key="vllm_max_model_len", value=self.args.vllm_max_model_len)
    self.remote_llm.update_attr(key="num_generations", value=args.num_generations)

    # read zero3 param and update model [assert not is_peft_model(self.model)]
    import deepspeed
    import time
    import tqdm
    for name, param in tqdm.tqdm(self.model.named_parameters(), total=len(list(self.model.named_parameters()))):
        with deepspeed.zero.GatheredParameters([param], enabled=self.accelerator.state.deepspeed_plugin.zero_stage == 3):
            if self.accelerator.is_main_process:
                weight = param.data
                assert weight.shape == param.ds_shape
                self.remote_llm.advanced_update_param(name, weight)
                logger.error(str(weight.sum().item()) + "\t" +  str(param.shape) + "\t" + str(param.ds_shape))
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # predict as usual
    prompt = tokenizer.apply_chat_template(conversation=[
        {"role": "user", "content": "黄金的延展性"},
    ], tokenize=False, add_generation_prompt=True)
    tokens = vrc.generate([prompt])
    result = tokenizer.batch_decode(tokens)
    ```

"""

import argparse
import logging
import os
import sys
import threading
import time

import cloudpickle as pickle
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from vllm.lora.request import LoRARequest
from vllm.worker.worker import Worker


logger = logging.getLogger(__name__)


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def get_remote_vllm_script_file():
    return f"{sys.executable} {os.path.abspath(__file__)}"


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    logger.warning("master_address " + str(master_address))
    logger.warning("master_port " + str(master_port))
    logger.warning("rank " + str(rank))
    logger.warning("world_size " + str(world_size))
    logger.warning("device " + str(device))
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class VllmRemoteClient:
    def __init__(self, nccl_link=False, target=None, master_port=51216, deploy_msg=None, num_gpu=None):
        self.url = f"http://{target}/api/request"
        self.remote_vllm_ip = target.split(":")[0]
        # client is not group master, so it must take the last position in world_size
        self.client_world_rank = int(num_gpu)
        if deploy_msg:
            logger.warning(deploy_msg)
        logger.warning("create process_group")
        self.nccl_link = nccl_link
        if self.nccl_link:
            self.model_update_group = stateless_init_process_group(
                master_address=self.remote_vllm_ip,
                master_port=master_port,
                rank=self.client_world_rank,
                world_size=num_gpu + 1,
                device=torch.device("cuda:0"),  # todo: this can cause problems
            )
        logger.warning("create process_group done")

        while True:
            try:
                self.check_online()
                break
            except Exception as e:
                logger.warning(f"waiting for vllm connection ({self.url}).")
                logger.warning(e)
                time.sleep(1)

    def connect(self, method, kwargs):
        serialized_req = pickle.dumps({"cmd": method, "kwargs": kwargs})
        logger.warning(f"vllm command ({method}) has been sent")
        response = requests.post(self.url, data=serialized_req)
        logger.warning(f"vllm command ({method}) response received")
        if response.status_code != 200:
            raise Exception(f"Server returned error status: {response.status_code}")
        return pickle.loads(response.content)

    def generate(self, all_prompts_text):
        try:
            return self.connect("generate", {"all_prompts_text": all_prompts_text})
        except Exception:
            logger.exception("unable to generate")

    def update_lora_param(self, lora_path):
        try:
            return self.connect("update_lora_param", {"lora_path": lora_path})
        except Exception:
            logger.exception("unable to update_lora_param")

    def update_attr(self, key, value):
        try:
            return self.connect("update_attr", {"key": key, "value": value})
        except Exception:
            logger.exception("unable to update_attr")

    def get_model(self):
        try:
            return self.connect("get_model", {})
        except Exception:
            logger.exception("unable to get_model")

    def prepare_advanced_update_weight(self, name, dtype, shape):
        try:
            return self.connect("prepare_advanced_update_weight", {"name": name, "dtype": dtype, "shape": shape})
        except Exception:
            logger.exception("unable to prepare_advanced_update_weight")

    def advanced_update_param(self, name, p):
        # clear cuda cache
        # sync weight from the training process to the inference engine.
        # self.prepare_advanced_update_weight(name, p.dtype, p.shape)
        t = threading.Thread(target=self.prepare_advanced_update_weight, args=(name, p.dtype, p.shape), daemon=True)
        t.start()
        self.model_update_group.broadcast(p, src=self.client_world_rank, stream=torch.cuda.current_stream())
        logger.warning("broadcast complete")
        t.join()

    def check_online(self):
        return self.connect("check_online", {"check_online": "check_online"})


class MyWorker(Worker):
    """
    The `MyWorker` class inherits from `Worker` to provide custom functions.
    For simplicity, we define the `MyWorker` class in this self-contained
    script. Normally, we should define the `MyWorker` class in a separate
    file and pass the qualified name of the class to the `worker_cls`
    parameter.
    """

    def init_weight_update_group(self, master_address, master_port, rank_offset, world_size):
        logger.warning("worker init")
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )
        # client 排倒数第一
        self.client_world_rank = world_size - 1

    def update_weight(self, name, dtype, shape):
        logger.warning("worker update weight")
        logger.warning(self.device)
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.model_update_group.broadcast(weight, src=self.client_world_rank, stream=torch.cuda.current_stream())
        logger.warning("broadcast complete")
        # logger.warning(weight.sum())
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


class VllmRemote:
    def __init__(
        self,
        name_or_path,
        vllm_gpu_memory_utilization,
        vllm_dtype,
        vllm_max_model_len,
        max_completion_length,
        temperature,
        guided_decoding,
        num_generations,
    ):
        from vllm import LLM

        logger.warning(f"llm init tensor_parallel_size {ENV['NUM_GPUS']}")
        self.base_model_name_or_path = name_or_path
        self.llm = LLM(
            model=self.base_model_name_or_path,
            tensor_parallel_size=ENV["NUM_GPUS"],
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_model_len=vllm_max_model_len,
            enable_lora=(ENV["MAX_LORA_RANK"] != 0),
            worker_cls=MyWorker,
            enforce_eager=True,
            max_lora_rank=ENV["MAX_LORA_RANK"],
            dtype=vllm_dtype,
        )

        logger.warning("llm init end")
        self.current_lora_path = None
        self.lora_cnt = 1
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        self.init_vllm_max_model_len = self.vllm_max_model_len = vllm_max_model_len
        self.guided_decoding = guided_decoding
        self.num_generations = num_generations

        self.master_address = ENV["REMOTE_VLLM_HOST_IP"]
        self.master_port = ENV["MASTER_PORT"]

        if ENV["NCCL_LINK"]:
            rank_offset = 0
            world_size = ENV["NUM_GPUS"] + 1
            logger.warning("create process_group")
            self.llm.collective_rpc(
                "init_weight_update_group", args=(self.master_address, self.master_port, rank_offset, world_size)
            )
            logger.warning("create process_group done")

    def generate(self, all_prompts_text):
        try:
            from vllm import SamplingParams

            if self.init_vllm_max_model_len != self.vllm_max_model_len:
                logger.warning(f"init_vllm_max_model_len {self.init_vllm_max_model_len} != vllm_max_model_len {self.vllm_max_model_len}")
            # assert self.current_lora_path is not None
            # Sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_completion_length,
                guided_decoding=self.guided_decoding,
                n=self.num_generations,
            )

            ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
            if self.current_lora_path:
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=True,
                    lora_request=LoRARequest("sql_adapter", self.lora_cnt, self.current_lora_path),
                )
            else:
                all_outputs = self.llm.generate(
                    ordered_set_of_prompts,
                    sampling_params=self.sampling_params,
                    use_tqdm=True,
                )
            completion_ids = []
            for outputs in all_outputs:
                for output in outputs.outputs:
                    completion_ids.append(output.token_ids)
            return completion_ids
        except Exception:
            logger.exception("find error")

    def update_param(self, state_dict):
        try:
            logger.warning("remote: update parm")
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())
            logger.warning("remote: update parm complete")
        except Exception:
            logger.exception("find error")

    def get_model(self):
        try:
            logger.warning("remote: get_model")
            return self.base_model_name_or_path
        except Exception:
            logger.exception("find error")

    def update_attr(self, key, value):
        setattr(self, key, value)

    def update_lora_param(self, lora_path):
        assert ENV["MAX_LORA_RANK"] != 0
        try:
            import shutil

            if self.current_lora_path and self.current_lora_path != lora_path:
                shutil.rmtree(self.current_lora_path)
        except Exception:
            pass
        self.current_lora_path = lora_path
        self.lora_cnt += 1

    def prepare_advanced_update_weight(self, name, dtype, shape):
        assert ENV["MAX_LORA_RANK"] == 0
        self.llm.collective_rpc("update_weight", args=(name, dtype, shape))

    def check_online(self, check_online):
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM Remote Server Configuration")
    parser.add_argument(
        "--host-ip",
        default=os.environ.get("REMOTE_VLLM_HOST_IP", "127.0.0.1"),
        help="Host IP address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--api-port", default=int(os.environ.get("REMOTE_VLLM_HTTP_API_PORT", "8000")), help="API port (default: 8000)"
    )
    parser.add_argument(
        "--master-port",
        default=int(os.environ.get("REMOTE_VLLM_MASTER_PORT", "51216")),
        help="Master port (default: 51216)",
    )
    parser.add_argument(
        "--nccl-link",
        type=lambda x: x.lower() == "true",
        default=os.environ.get("REMOTE_VLLM_NCCL_LINK", "False") == "True",
        help="Enable NCCL link (default: False)",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("REMOTE_VLLM_INIT_MODEL"),
        required=not bool(os.environ.get("REMOTE_VLLM_INIT_MODEL")),
        help="Path to the model",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("REMOTE_VLLM_MAX_MODEL_LEN", "8192")),
        help="Maximum model length (default: 8192)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("REMOTE_VLLM_GPUS", "1")),
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("REMOTE_VLLM_GPU_FRAG", "0.9")),
        help="GPU memory utilization (default: 0.9)",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=int(os.environ.get("REMOTE_VLLM_MAX_LORA_RANK", "0")),
        help="Maximum LoRA rank (default: 0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("REMOTE_VLLM_TEMPERATURE", "0.7")),
        help="Temperature for sampling (default: 0.7)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=int(os.environ.get("REMOTE_VLLM_NUM_GENERATION", "1")),
        help="Number of generations (default: 1)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ENV = {
        "REMOTE_VLLM_HOST_IP": args.host_ip,
        "API_PORT": args.api_port,
        "MASTER_PORT": args.master_port,
        "NCCL_LINK": args.nccl_link,
        "MODEL_PATH": args.model_path,
        "MAX_MODEL_LEN": args.max_model_len,
        "NUM_GPUS": args.num_gpus,
        "GPU_MEMORY_UTILIZATION": args.gpu_memory_utilization,
        "MAX_LORA_RANK": args.max_lora_rank,
        "MAX_COMPLETION_LENGTH": 1024,  # can be override dynamically
        "TEMPERATURE": args.temperature,  # can be override dynamically
        "NUM_GENERATIONS": args.num_generations,  # can be override dynamically
    }

    request_lock = threading.Lock()
    app = FastAPI()

    vr = VllmRemote(
        ENV["MODEL_PATH"],
        ENV["GPU_MEMORY_UTILIZATION"],
        "auto",
        vllm_max_model_len=ENV["MAX_MODEL_LEN"],
        max_completion_length=ENV["MAX_COMPLETION_LENGTH"],
        temperature=ENV["TEMPERATURE"],
        guided_decoding=None,
        num_generations=ENV["NUM_GENERATIONS"],
    )

    def process_command(cmd: str, kwargs) -> dict:
        """
        process client command
        """
        return getattr(vr, cmd)(**kwargs)

    @app.post("/api/request")
    async def handle_request(request: Request):
        try:
            body = await request.body()
            try:
                req_obj = pickle.loads(body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"unable to unpickle command: {str(e)}") from e

            with request_lock:
                logger.warning(req_obj["cmd"])
                result = process_command(req_obj["cmd"], req_obj["kwargs"])
                return Response(content=pickle.dumps(result), media_type="application/octet-stream")
        except Exception as e:
            logger.warning(e)
            raise HTTPException(status_code=500, detail=f"error when dealing with: {str(e)}") from e

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=int(ENV["API_PORT"]), log_level="info")

    # 主函数
    def main():
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("terminate")

    main()
