import argparse
import os

import yaml
from accelerate.commands import launch
from haven import haven_wizard as hw


def run_exp(exp_dict, savedir, args):
    exp_name = exp_dict.pop("name")
    print(args)

    if args.wandb:
        os.environ["WANDB_MODE"] = "online"
        os.environ["WANDB_RUN_ID"] = os.path.basename(savedir)
        os.environ["WANDB_NAME"] = exp_name
    else:
        os.environ["WANDB_MODE"] = "disabled"

    if exp_name.startswith("marlhf"):
        print("MARLHF")
        accelerate_launch("rl_training_with_ma_value.py", exp_dict, args.gpus, args.accelerate_config)
    elif exp_name.startswith("rlhf"):
        print("RLHF")
        accelerate_launch("rl_training.py", exp_dict, args.gpus, args.accelerate_config)
    elif exp_name.startswith("rm"):
        accelerate_launch("reward_modeling.py", exp_dict, args.gpus, args.accelerate_config)
    elif exp_name.startswith("gptrm"):
        accelerate_launch("gpt_reward_modeling.py", exp_dict, args.gpus, args.accelerate_config)
    elif exp_name.startswith("sft"):
        accelerate_launch("supervised_finetuning.py", exp_dict, args.gpus, args.accelerate_config)
    elif exp_name.startswith("rouge"):
        accelerate_launch("evaluate_rouge.py", exp_dict, args.gpus, args.accelerate_config)
    elif exp_name.startswith("pseudo"):
        accelerate_launch("inference_pseudolabel.py", exp_dict, args.gpus, args.accelerate_config)
    else:
        raise Exception(f"Config file {exp_name} does not start with one of the correct prefixes")


def accelerate_launch(training_file, training_args_dict, num_gpus=1, config_file=None):
    parser = launch.launch_command_parser()
    training_cmd_args = []
    # if config_file is not None:
    #     training_cmd_args.extend(["--config_file", config_file])

    if num_gpus > 1:
        training_cmd_args.append("--multi_gpu")
        training_cmd_args.extend(["--num_machines", "1"])
        training_cmd_args.extend(["--num_processes", str(num_gpus)])
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
    parser.add_argument("--gpu-mem", default=32, type=int, help="mem of gpus to use for experiment")
    parser.add_argument("--wandb", action="store_true", help="force enable wandb", default=False)
    # parser.add_argument(
    #     "--exp-id", default=None, help="id used to resume an experiment"
    # )

    args, extra_args = parser.parse_known_args()

    with open(args.exp_group, "r") as fp:
        exp_dict = yaml.safe_load(fp)

    exp_dict["name"] = os.path.basename(args.exp_group)

    # for key, val in vars(extra_args).items():
    #     exp_dict[key] = val

    exp_list = [exp_dict]

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

    # Run experiments and create results file
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
        # exp_id=args.exp_id,
    )
