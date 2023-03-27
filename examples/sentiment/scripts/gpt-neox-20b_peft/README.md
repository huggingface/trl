
# Fine-tuning 20B LLMs with RL on a 24GB consumer GPU
The scripts in the section detail the fine-tuning a 20b LLM in 8-bit, in order to generate positive imdb reviews. You
can find out more in our [blogpost](https://huggingface.co/blog/trl-peft).

Overall there were three key steps and training scripts:

1. **clm_finetune_peft_imdb.py** - Fine tuning a Low Rank Adapter on a frozen 8-bit model for text generation on the imdb dataset.
2. **merge_peft_adapter.py** - Merging of the adapter layers into the base model’s weights and storing these on the hub.
3. **gpt-neo-20b_sentiment_peft.py** - Sentiment fine-tuning of a Low Rank Adapter to create positive reviews.
