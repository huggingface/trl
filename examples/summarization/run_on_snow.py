import argparse
import datetime
import os
import subprocess
from copy import deepcopy

import generate_and_eval
import generate_and_llm_judge
import yaml
from accelerate.commands import launch
from generate_vllm import generate_relabel_args_dict
from haven import haven_wizard as hw


def run_exp(exp_dict, savedir, args):
    exp_name = exp_dict.pop("name")
    git_hash = exp_dict.pop("git")
    print(args)

    os.environ["WANDB_RUN_ID"] = os.path.basename(savedir)
    os.environ["WANDB_NAME"] = exp_name
    os.environ["WANDB_RUN_GROUP"] = exp_name + "_" + git_hash

    if args.wandb:
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_PROJECT"] = "trl"
        os.environ["WANDB_ENTITY"] = "mila-language-drift"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    if exp_name.startswith("marlhf"):
        print("MARLHF")
        accelerate_launch("rl_training_with_ma_value.py", exp_dict, args)
    elif exp_name.startswith("vmrlhf"):
        print("Separate Value Model RLHF")
        accelerate_launch("rl_training_value_model.py", exp_dict, args)
    elif exp_name.startswith("rlhf"):
        print("RLHF")
        accelerate_launch("rl_training.py", exp_dict, args)
    elif exp_name.startswith("dpo"):
        print("DPO")
        accelerate_launch("dpo_training.py", exp_dict, args)
    elif exp_name.startswith("rm"):
        accelerate_launch("reward_modeling.py", exp_dict, args)
    elif exp_name.startswith("gptrm"):
        accelerate_launch("gpt_reward_modeling.py", exp_dict, args)
    elif exp_name.startswith("sft"):
        accelerate_launch("supervised_finetuning.py", exp_dict, args)
    elif exp_name.startswith("newsft"):
        accelerate_launch("sft.py", exp_dict, args)
    elif exp_name.startswith("rouge"):
        exp_dict.pop("save_strategy", None)
        accelerate_launch("evaluate_rouge.py", exp_dict, args)
    elif exp_name.startswith("pseudo"):
        exp_dict.pop("save_strategy", None)
        accelerate_launch("inference_pseudolabel.py", exp_dict, args)
    elif exp_name.startswith("create_rlhf"):
        exp_dict.pop("save_strategy", None)
        accelerate_launch("create_rlhf_dataset.py", exp_dict, args)
    elif exp_name.startswith("vllm"):
        exp_dict.pop("save_strategy", None)
        exp_dict["num_gpus"] = args.gpus
        generate_relabel_args_dict(exp_dict)
    elif exp_name.startswith("geneval"):
        exp_dict.pop("save_strategy", None)
        exp_dict["num_gpus"] = args.gpus
        generate_and_eval.main_args_dict(exp_dict)
    elif exp_name.startswith("genjudge"):
        exp_dict.pop("save_strategy", None)
        exp_dict["num_gpus"] = args.gpus
        generate_and_llm_judge.main_args_dict(exp_dict)
    elif exp_name.startswith("scalarrm"):
        exp_dict.pop("save_strategy", None)
        accelerate_launch("scalar_rm_model.py", exp_dict, args)
    elif exp_name.startswith("costa_dpo"):
        accelerate_launch("costa_dpo.py", exp_dict, args)
    else:
        raise Exception(f"Config file {exp_name} does not start with one of the correct prefixes")


def accelerate_launch(training_file, training_args_dict, args):
    parser = launch.launch_command_parser()
    training_cmd_args = []
    if args.accelerate_config is not None and args.accelerate_config != "None":
        training_cmd_args.extend(["--config_file", args.accelerate_config])
        # training_cmd_args.extend(["--num_processes", str(args.gpus)])
        # training_cmd_args.extend(
        #     ["--gradient_accumulation_steps", str(training_args_dict["gradient_accumulation_steps"])]
        # )
    elif args.gpus > 1:
        training_cmd_args.append("--multi_gpu")

    # if training_args_dict.pop("fp16", False):
    #     mixed_precision = "fp16"
    # elif training_args_dict.pop("bf16", False):
    #     mixed_precision = "bf16"
    if training_args_dict.get("fp16", False):
        mixed_precision = "fp16"
    elif training_args_dict.get("bf16", False):
        mixed_precision = "bf16"
    else:
        mixed_precision = "no"
    training_cmd_args.extend(["--mixed_precision", mixed_precision])
    #

    training_cmd_args.extend(["--num_machines", "1"])
    training_cmd_args.extend(["--num_processes", str(args.gpus)])
    # if args.gpus > 1:
    #     if args.deepspeed is not None and args.deepspeed != "None":
    #         assert (
    #             "gradient_accumulation_steps" in training_args_dict
    #         ), "Must include gradient_accumulation_steps in config"
    #         training_cmd_args.append("--use_deepspeed")
    #         training_cmd_args.extend(["--zero_stage", str(args.deepspeed)])
    #         training_cmd_args.extend(
    #             ["--gradient_accumulation_steps", str(training_args_dict["gradient_accumulation_steps"])]
    #         )

    training_cmd_args.append(training_file)
    for key, val in training_args_dict.items():
        training_cmd_args.append(f"--{key}")
        if not (isinstance(val, bool) and val is True):
            training_cmd_args.append(str(val))

    print(" ".join(training_cmd_args))
    args = parser.parse_args(training_cmd_args)
    launch.launch_command(args)


if __name__ == "__main__":
    # Specify arguments regarding save directory and job scheduler
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_group",
        help="Define the experiment group to run.",
        nargs="+",
    )
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default="/home/toolkit/trl/results",
        help="Define the base directory where the experiments will be saved.",
    )
    parser.add_argument(
        "-r",
        "--reset",
        type=int,
        default=0,
        help="If true, reset the experiment. Else, resume.",
    )
    parser.add_argument(
        "-j",
        "--job_scheduler",
        default=None,
        type=str,
        help="Run the experiments as jobs in the cluster.",
    )
    parser.add_argument(
        "-p",
        "--python_binary",
        default="/home/toolkit/.conda/envs/trl/bin/python",
        help="path to your python executable",
    )
    parser.add_argument("-n", "--gpus", default=1, type=int, help="number of gpus to use for experiment")
    parser.add_argument("-a", "--accelerate_config", default=None, help="accelerate config")
    # parser.add_argument("-d", "--deepspeed", default=None, help="ds stage")
    parser.add_argument("--gpu-mem", default=32, type=int, help="mem of gpus to use for experiment")
    parser.add_argument("--wandb", action="store_true", help="force enable wandb", default=False)
    parser.add_argument("--local-save", action="store_true", help="force local save", default=False)
    parser.add_argument("--search", default=None)
    # parser.add_argument(
    #     "--exp-id", default=None, help="id used to resume an experiment"
    # )

    args, extra_args = parser.parse_known_args()

    exp_list = []
    for exp_file in args.exp_group:
        with open(exp_file, "r") as fp:
            exp_dict = yaml.safe_load(fp)

        exp_dict["name"] = os.path.basename(exp_file)
        exp_dict["git"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()

        if args.search is not None and args.search != "None":
            search_key, search_val_str = args.search.split("=")
            search_vals = search_val_str.split(",")
            exps = []
            for val in search_vals:
                exp_dict_copy = deepcopy(exp_dict)
                exp_dict_copy[search_key] = val
                # exp_dict_copy["name"] = exp_dict_copy["name"] + f"/{search_key}={val}"
                exps.append(exp_dict_copy)
            # for key, val in vars(extra_args).items():
            #     exp_dict[key] = val
            # print(exps)
        else:
            exps = [exp_dict]

        exp_list.extend(exps)

    args.exp_group = " ".join(args.exp_group)
    print(args.exp_group)

    if args.job_scheduler == "toolkit":
        with open("/home/toolkit/wandb_api_key", "r") as f:
            wandb_api_key = f.read().rstrip()

        job_config = {
            "account_id": os.environ["EAI_ACCOUNT_ID"],
            # "image": "registry.console.elementai.com/snow.colab/cuda",
            # "image": "registry.console.elementai.com/snow.colab_public/ssh",
            # "image": "registry.console.elementai.com/snow.mnoukhov/rl4lms",
            "image": "registry.console.elementai.com/snow.interactive_toolkit/default",
            "data": [
                "snow.mnoukhov.home:/home/toolkit",
                "snow.colab.public:/mnt/public",
            ],
            "environment_vars": [
                "HOME=/home/toolkit",
                "HF_HOME=/home/toolkit/huggingface/",
                f"WANDB_API_KEY={wandb_api_key}",
                "WANDB_RESUME=allow",
                "WANDB__SERVICE_WAIT=300",
                "WANDB_PROJECT=trl",
                "WANDB_ENTITY=mila-language-drift",
            ],
            "restartable": True,
            "resources": {
                "cpu": 4 * args.gpus,
                "mem": 64 * args.gpus,
                "gpu_mem": args.gpu_mem,
                "gpu": args.gpus,
            },
            "interactive": False,
            "bid": 9999,
        }
        job_scheduler = "toolkit"
        args.wandb = True
    else:
        job_config = None
        job_scheduler = None

        if args.wandb:
            timenow = datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
            exp_list[0]["name"] = exp_list[0]["name"] + f"_local_{timenow}"

        if not args.local_save:
            exp_list[0]["save_strategy"] = "no"

    # Run experiments and create results file
    # if job_scheduler == "toolkit":
    from haven import haven_wizard as hw

    hw.run_wizard(
        func=run_exp,
        exp_list=exp_list,
        savedir_base=args.savedir_base,
        reset=args.reset,
        job_config=job_config,
        job_scheduler=job_scheduler,
        results_fname="results/notebook.ipynb",
        python_binary_path=args.python_binary,
        args=args,
        use_threads=True,
        save_logs=False,
    )
    # else:
    #     run_exp(exp_list[0], "output", args)
