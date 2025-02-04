# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM
import torch
import argparse
import uvicorn
import argparse
from trl import ModelConfig
"""
Usage 
python trl/models/remote_model_app.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8000 
"""

app = FastAPI()
model = None

class ForwardPassRequest(BaseModel):
    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    logits_to_keep: int

@app.post("/forward/")
async def forward_pass(request: ForwardPassRequest):
    print(request)
    device = model.device
    input_ids = torch.LongTensor(request.input_ids).to(device)
    attention_mask = torch.LongTensor(request.attention_mask).to(device)
    logits_to_keep = request.logits_to_keep
    # Perform the forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep,
        )
        logits = outputs.logits

    # Convert logits to CPU and then to a list for JSON serialization
    logits_list = logits.cpu().tolist()

    return {"logits": logits_list}
    
@app.get("/health")
async def health_check():
    """
    Provides a health check endpoint for the server.

    Returns:
        dict: A dictionary indicating the server's health status.
    """
    return {"status": "OK"}
    
def init_model(model_config: ModelConfig):
    global model
    
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, 
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        )

    if torch.cuda.is_available():
        model.to("cuda")
        print(f"Model '{model_config.model_name_or_path}' loaded on GPU")
    else:
        print(f"Model '{model_config.model_name_or_path}' loaded on CPU")

if __name__ == "__main__":
    from trl import ModelConfig, TrlParser
    parser = TrlParser(ModelConfig)
    model_args = parser.parse_args_and_config()[0]
    init_model(model_args)
    uvicorn.run(app)