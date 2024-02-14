import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import GenerationConfig, TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class TextGenerationCallback(TrainerCallback):
    def __init__(self, prompt_dataset: Dataset, prompt_column: str, generation_config: GenerationConfig):
        self.prompts = prompt_dataset[prompt_column]
        self.generation_config = generation_config
        self.completions = []

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs["model"]
        tokenizer = kwargs["tokenizer"]
        distributed_state = PartialState()
        model.to(distributed_state.device)
        model.eval()  # Set model to evaluation mode
        completions_per_process = []
        with distributed_state.split_between_processes(self.prompts, apply_padding=True) as prompts:
            for message in tqdm(prompts, desc="Generating completions"):
                tokenized_message = tokenizer.apply_chat_template(
                    [message], tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(model.device)
                output = model.generate(tokenized_message, self.generation_config)
                # Slice out prompt tokens and decode
                generated_text = tokenizer.decode(output[0][tokenized_message.shape[-1] :], skip_special_tokens=True)
                completions_per_process.append(generated_text)

        # Gather completions across all processes
        completions_gather = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(completions_gather, completions_per_process)
        # Flatten data
        completions = [msg for sublist in completions_gather for msg in sublist]
        # Drop duplicates produced by padding
        self.completions = completions[: len(self.prompts)]
