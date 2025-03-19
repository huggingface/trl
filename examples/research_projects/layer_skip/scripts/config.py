model_name = "unsloth/Llama-3.2-3B"
tokenizer_name = "unsloth/Llama-3.2-3B"
dataset_name = "WillHeld/top_v2"

output_root_dir = "./checkpoints/"
hub_model_id = f"ariG23498/layerskip-{model_name.split('/')[1]}-{dataset_name.split('/')[1]}"
output_dir = f"{output_root_dir}/{hub_model_id}"

per_device_train_batch_size = 8
gradient_accumulation_steps = 1
learning_rate = 2e-5
