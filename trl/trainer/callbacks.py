import torch.distributed as dist
from accelerate import PartialState
from tqdm.auto import tqdm
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class TextGenerationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, messages, output_dataset_name: str, push_to_hub: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.messages = messages
        self.output_dataset_name = output_dataset_name
        self.push_to_hub = push_to_hub

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"KWARGS: {kwargs}")
        distributed_state = PartialState()
        self.model.to(distributed_state.device)
        self.model.eval()  # Set model to evaluation mode
        completions_per_process = []
        with distributed_state.split_between_processes(self.messages, apply_padding=True) as messages:
            for message in tqdm(messages, desc="Generating completions"):
                print(f"\nGenerating text for message: {message} on device: {self.model.device}")
                tokenized_message = self.tokenizer.apply_chat_template(
                    [message], tokenize=True, add_generation_prompt=True, return_tensors="pt"
                ).to(self.model.device)
                output = self.model.generate(
                    tokenized_message, max_new_tokens=32, do_sample=True, temperature=0.7, synced_gpus=True
                )
                # Slice out prompt tokens and decode
                generated_text = self.tokenizer.decode(
                    output[0][tokenized_message.shape[-1] :], skip_special_tokens=True
                )
                completions_per_process.append(generated_text)
                print(f"Prompt: {message['content']}\nGenerated Text: {generated_text}\n")

        # Gather messages across all processes
        completions_gather = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(completions_gather, completions_per_process)

        # Flatten data
        completions = [msg for sublist in completions_gather for msg in sublist]
        # if len(completions) != len(self.messages):
        #     # Drop last sample as it's a duplicate from padding
        #     completions = completions[:-1]
        print(f"Generated {len(completions)} completions: {completions}\n\n")
