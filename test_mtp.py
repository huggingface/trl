#!/usr/bin/env python3
"""
Compare outputs between original Qwen2.5-0.5B and MTP fine-tuned model on Capybara samples
"""

import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_models_and_tokenizer():
    """Load both original and fine-tuned models with tokenizer"""
    print("Loading models and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load original model
    print("Loading original Qwen2.5-0.5B...")
    original_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load fine-tuned model
    print("Loading MTP fine-tuned model...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        "/root/autodl-tmp/Qwen2.5-0.5B-MTP-Identical-Capybara",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return tokenizer, original_model, finetuned_model

def format_conversation(messages):
    """Format conversation messages into a prompt"""
    formatted = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            formatted += f"User: {content}\n"
        elif role == 'assistant':
            formatted += f"Assistant: {content}\n"
    return formatted.strip()

def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens (response)
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response.strip()

def main():
    # Load models
    tokenizer, original_model, finetuned_model = load_models_and_tokenizer()
    
    # Load Capybara dataset
    print("Loading Capybara test dataset...")
    dataset = load_dataset("trl-lib/Capybara", split="test")
    
    # Select 5 random samples
    random.seed(42)  # For reproducibility
    sample_indices = random.sample(range(len(dataset)), 5)
    
    print(f"\n{'='*80}")
    print("COMPARISON: Original vs MTP Fine-tuned Model")
    print(f"{'='*80}")
    
    for i, idx in enumerate(sample_indices, 1):
        sample = dataset[idx]
        messages = sample['messages']
        
        # Create prompt from conversation (exclude last assistant message for generation)
        conversation_for_prompt = []
        for msg in messages:
            if msg['role'] == 'user':
                conversation_for_prompt.append(msg)
            elif msg['role'] == 'assistant':
                # Only include first assistant message as context, then break
                conversation_for_prompt.append(msg)
                break
        
        # Find the last user message to use as prompt
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        if user_messages:
            last_user_msg = user_messages[-1]['content']
            prompt = f"User: {last_user_msg}\nAssistant:"
        else:
            continue
        
        print(f"\n{'-'*60}")
        print(f"SAMPLE {i} (Index: {idx})")
        print(f"{'-'*60}")
        print(f"PROMPT: {last_user_msg}")
        
        # Get ground truth (last assistant message)
        assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
        if assistant_messages:
            ground_truth = assistant_messages[-1]['content']
            print(f"\nGROUND TRUTH:\n{ground_truth}")
        
        # Generate from original model
        print(f"\nORIGINAL MODEL OUTPUT:")
        try:
            original_response = generate_response(original_model, tokenizer, prompt)
            print(original_response)
        except Exception as e:
            print(f"Error: {e}")
        
        # Generate from fine-tuned model
        print(f"\nMTP FINE-TUNED MODEL OUTPUT:")
        try:
            finetuned_response = generate_response(finetuned_model, tokenizer, prompt)
            print(finetuned_response)
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()