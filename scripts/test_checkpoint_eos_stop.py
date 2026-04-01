import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test EOS stopping behavior for a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/root/autodl-tmp/Qwen2-1.5B-SFT-IF/checkpoint-867",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="国庆节的日期是什么时候",
        help="Prompt to test generation with.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens for generation.",
    )
    parser.add_argument(
        "--eos_token_id",
        type=int,
        default=151645,
        help="Expected EOS token id.",
    )
    return parser


def run_case(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    generation_kwargs: dict[str, Any],
    expected_eos_token_id: int,
    title: str,
) -> None:
    messages = [{"role": "user", "content": prompt}]
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(model_inputs, "to"):
        model_inputs = model_inputs.to(model.device)

    if isinstance(model_inputs, torch.Tensor):
        input_ids = model_inputs
    else:
        input_ids = model_inputs["input_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **generation_kwargs,
        )

    input_len = input_ids.shape[-1]
    new_token_ids = output_ids[0, input_len:].tolist()

    eos_positions = [i for i, tid in enumerate(new_token_ids) if tid == expected_eos_token_id]
    has_eos = len(eos_positions) > 0
    first_eos_pos = eos_positions[0] if has_eos else None
    last_token_id = new_token_ids[-1] if new_token_ids else None
    stopped_by_eos = has_eos and first_eos_pos == len(new_token_ids) - 1

    print("=" * 80)
    print(title)
    print(f"new_tokens_len: {len(new_token_ids)}")
    print(f"last_token_id: {last_token_id}")
    print(f"expected_eos_token_id: {expected_eos_token_id}")
    print(f"has_expected_eos: {has_eos}")
    print(f"first_expected_eos_pos: {first_eos_pos}")
    print(f"stopped_by_expected_eos: {stopped_by_eos}")
    print("new_tokens_tail(32):", new_token_ids[-32:])
    print("decoded_output:")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=False))


def main() -> None:
    args = build_parser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    print("checkpoint:", args.checkpoint)
    print("tokenizer.eos_token:", tokenizer.eos_token)
    print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
    print("<|im_end|> id:", tokenizer.convert_tokens_to_ids("<|im_end|>"))
    print("<|endoftext|> id:", tokenizer.convert_tokens_to_ids("<|endoftext|>"))
    print("model.generation_config.eos_token_id:", model.generation_config.eos_token_id)
    print("model.generation_config.pad_token_id:", model.generation_config.pad_token_id)

    run_case(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        generation_kwargs={
            "eos_token_id": args.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643,
        },
        expected_eos_token_id=args.eos_token_id,
        title="CASE A: Explicit eos_token_id=151645",
    )

    run_case(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        generation_kwargs={},
        expected_eos_token_id=args.eos_token_id,
        title="CASE B: Default generation config",
    )


if __name__ == "__main__":
    main()
