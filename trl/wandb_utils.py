import os
import wandb
import hashlib
from time import time
from typing import Any, Optional

def setup_wandb(wandb_cfg: Optional[dict], wandb_dir: str, config: Optional[dict] = None, resume: bool = True) -> Any:
    if wandb_cfg is None:
        return None

    os.makedirs(wandb_dir, exist_ok=True)

    if resume and os.path.exists(os.path.join(wandb_dir, "wandb_id.txt")):
        with open(os.path.join(wandb_dir, "wandb_id.txt"), "r") as fin:
            wandb_id = fin.read().strip()
    else:
        wandb_id = hashlib.sha1(wandb_cfg["name"].encode("utf-8")).hexdigest() + str(int(time()))

    logger = wandb.init(
        job_type=wandb_cfg.get("job_type", None),
        dir=wandb_dir,
        entity=wandb_cfg["entity"],
        project=wandb_cfg["project"],
        group=wandb_cfg.get("group", None),
        name=wandb_cfg["name"],
        id=wandb_id,
        mode=wandb_cfg.get("mode", "online"),
        resume="allow",
        config=config,
    )
    
    with open(os.path.join(wandb_dir, "wandb_id.txt"), "w") as fout:
        fout.write(wandb_id)

    return logger