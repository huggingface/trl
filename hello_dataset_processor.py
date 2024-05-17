import pprint

from datasets import load_dataset
from transformers import AutoTokenizer

from trl.dataset_processor import DatasetConfig, PreferenceDatasetProcessor, visualize_token


CHATML_CHAT_TEMPLATE = """{% for message in messages %}{{'\n' if not loop.first else ''}}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>'}}{% endfor %}"""

tok = AutoTokenizer.from_pretrained("gpt2")
if tok.chat_template is None:
    tok.chat_template = CHATML_CHAT_TEMPLATE
sanity_check = True  # deal with a small chunk of dataset to test run the code

preference_datasets = load_dataset("trl-internal-testing/hh-rlhf-trl-style")
if sanity_check:
    for key in preference_datasets:
        preference_datasets[key] = preference_datasets[key].select(range(1000))


dataset_config = DatasetConfig(max_token_length=1024, max_prompt_token_lenth=128)
dataset_processor = PreferenceDatasetProcessor(tokenizer=tok, config=dataset_config)
train_dataset = dataset_processor.tokenize(preference_datasets["train"])
stats = dataset_processor.get_token_length_stats(train_dataset)
pprint.pp(stats)
train_dataset = dataset_processor.filter(train_dataset)
stats = dataset_processor.get_token_length_stats(train_dataset)
pprint.pp(stats)
dataset_processor.get_token_length_visualization(train_dataset)
visualize_token(train_dataset[0]["chosen"], tok)
