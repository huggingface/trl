# Best-of-n sampling as an alternative to RLHF

Paraphrasing from [OpenAI's blog post on best-of-n sampling](https://openai.com/research/measuring-goodharts-law)

With `RLHF` we try to optimize w.r.t to a proxy objective. `RLHF` is not the only way to do this. 
One of the many other ways is `best-of-n sampling`. It is simple to implement and competitive to `RLHF` in some cases.
That said, `best-of-n sampling` is expensive when it comes to inference time compute.

The included notebook compares reward-model scores of prompt based responses from 
1. a base model (`gpt2-imdb`)
2. `RLHF` tuned model based on this base-model 
3. the base-model again from which we sample n responses to each prompt, score them and take the best scored one AKA the `best-of-n sampled` model




