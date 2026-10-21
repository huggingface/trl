"""CPU-only source-extracted async OPD request provenance regression."""
import ast
import asyncio
import random
from pathlib import Path
p=Path("trl/experimental/async_distillation/async_rollout_worker.py")
tree=ast.parse(p.read_text(),filename=str(p))
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="_AsyncRolloutLoop")
names={"_request_model_for_version","_generate_one_turn"}
methods=[n for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
assert {n.name for n in methods}==names
namespace={"random":random}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(p),"exec"),namespace)
generate=namespace["_generate_one_turn"]
route=namespace["_request_model_for_version"]
class Loop:
    lora_name="policy"
    model_name="base"
    model_version=9
    max_tokens=4
    temperature=1.
    top_p=1.
    top_k=0
    repetition_penalty=1.
    min_p=None
    vllm_server_url="http://vllm"
    request_timeout=2
    _request_model_for_version=route
    async def _retry(self, thunk, **kw):
        return await thunk()
    async def _post(self, url, endpoint, payload, timeout):
        assert url==self.vllm_server_url and endpoint=="/v1/completions"
        assert payload["model"]==("policy-v9" if self.lora_name else "base"),payload
        self.model_version=10  # publish new policy while request is in flight
        return {"choices":[{"token_ids":[7]}]}
loop=Loop()
tokens, sampled_version=asyncio.run(generate(loop,[1,2]))
assert tokens==[7] and sampled_version==9
assert loop.model_version==10 and route(loop,sampled_version)=="policy-v9"
print("PASS LoRA request dispatch version 9 survives in-flight publish of version 10")
loop=Loop()
loop.lora_name=None
tokens, version=asyncio.run(generate(loop,[1,2]))
assert version==9 and route(loop,version)=="base"
print("PASS dense request identity without adapter")
