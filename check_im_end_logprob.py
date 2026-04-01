import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Check <|im_end|> log-probability during generation")
    parser.add_argument("--model_path", type=str, required=True, help="Local model/checkpoint path")
    parser.add_argument("--prompt", type=str, default="请简要介绍一下大语言模型。", help="User prompt")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    # Some local checkpoints store extra_special_tokens as a list, while newer
    # transformers expects a mapping in this code path.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        extra_special_tokens={},
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    im_end_token = "<|im_end|>"
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)
    if im_end_id is None or im_end_id < 0:
        raise ValueError(f"Token {im_end_token} not found in tokenizer vocab")

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    input_len = model_inputs.input_ids.shape[1]
    sequences = outputs.sequences
    gen_token_ids = sequences[0, input_len:].tolist()
    scores = outputs.scores

    print(f"model_path: {args.model_path}")
    print(f"im_end_id: {im_end_id}")
    print(f"generated_tokens: {len(gen_token_ids)}")

    eos_step = None
    eos_logprob_at_step = None

    for step, (token_id, score_t) in enumerate(zip(gen_token_ids, scores), start=1):
        log_probs = torch.log_softmax(score_t, dim=-1)
        lp_im_end = log_probs[0, im_end_id].item()
        if token_id == im_end_id:
            eos_step = step
            eos_logprob_at_step = lp_im_end
            break

    if eos_step is not None:
        print(f"im_end_generated: yes")
        print(f"im_end_step: {eos_step}")
        print(f"im_end_logprob_when_generated: {eos_logprob_at_step:.6f}")
        print(f"im_end_prob_when_generated: {math.exp(eos_logprob_at_step):.8f}")
    else:
        print("im_end_generated: no")

    if len(scores) > 0:
        last_step = len(scores)
        last_log_probs = torch.log_softmax(scores[-1], dim=-1)
        last_lp_im_end = last_log_probs[0, im_end_id].item()
        print(f"last_step: {last_step}")
        print(f"im_end_logprob_at_last_step: {last_lp_im_end:.6f}")
        print(f"im_end_prob_at_last_step: {math.exp(last_lp_im_end):.8f}")

    decoded = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
    print("generated_text_with_special_tokens:")
    print(decoded)


if __name__ == "__main__":
    main()
