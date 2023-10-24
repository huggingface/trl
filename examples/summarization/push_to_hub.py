# from transformers import AutoModelForCausalLM
#
#
# name_or_path = "/home/toolkit/huggingface/tldr_sft_pythia7b_4V100_seq550"
#
# model = AutoModelForCausalLM.from_pretrained(name_or_path)
#
# model.push_to_hub("tldr_sft_pythia7b_4V100_seq550", use_temp_dir=False)

from datasets import load_from_disk


ds = load_from_disk("/home/toolkit/huggingface/openai_summarize_comparisons_pythia1b")
ds.push_to_hub("openai_summarize_comparisons_relabel_pythia1b")
