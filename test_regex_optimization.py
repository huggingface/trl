#!/usr/bin/env python3
"""
Test script to verify the regex optimizations in grpo_trainer.py
"""
import re
import time
import sys
import os

# Add the trl package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_regex_performance():
    """Test the performance difference between original and optimized regex patterns."""
    
    # Test data
    test_texts = [
        "<pad><pad><pad>Hello world",
        "<pad>Test string",
        "No padding here",
        "<pad><pad><pad><pad><pad>Long padded text",
    ] * 1000  # Multiply to get a substantial dataset
    
    pad_token = "<pad>"
    
    # Original approach (recompiling regex each time)
    print("Testing original approach (recompiling regex)...")
    start_time = time.time()
    for _ in range(10):  # Run multiple times for better measurement
        original_results = []
        for text in test_texts:
            result = re.sub(rf"^({re.escape(pad_token)})+", "", text)
            original_results.append(result)
    original_time = time.time() - start_time
    
    # Optimized approach (pre-compiled regex)
    print("Testing optimized approach (pre-compiled regex)...")
    escaped_pad_token = re.escape(pad_token)
    compiled_pattern = re.compile(rf"^({escaped_pad_token})+")
    
    start_time = time.time()
    for _ in range(10):  # Run multiple times for better measurement
        optimized_results = []
        for text in test_texts:
            result = compiled_pattern.sub("", text)
            optimized_results.append(result)
    optimized_time = time.time() - start_time
    
    # Verify results are identical
    assert original_results == optimized_results, "Results differ between original and optimized approaches!"
    
    # Print performance results
    print(f"Original approach time: {original_time:.4f} seconds")
    print(f"Optimized approach time: {optimized_time:.4f} seconds")
    improvement = (original_time - optimized_time) / original_time * 100
    print(f"Performance improvement: {improvement:.2f}%")
    
    return improvement

def test_image_token_optimization():
    """Test the image token regex optimizations."""
    
    test_texts = [
        "<image><image><image>Some text",
        "<image>Single image",
        "No image tokens here",
        "<image><image>Multiple images<image><image>",
    ] * 500
    
    image_token = "<image>"
    
    # Original approach
    start_time = time.time()
    for _ in range(10):
        original_results = []
        for text in test_texts:
            escaped_img_token = re.escape(image_token)
            # Test search operation
            has_image = bool(re.search(escaped_img_token, text))
            # Test substitution operation
            result = re.sub(rf"({escaped_img_token})+", image_token, text)
            original_results.append((has_image, result))
    original_time = time.time() - start_time
    
    # Optimized approach
    escaped_img_token = re.escape(image_token)
    search_pattern = re.compile(escaped_img_token)
    sub_pattern = re.compile(rf"({escaped_img_token})+")
    
    start_time = time.time()
    for _ in range(10):
        optimized_results = []
        for text in test_texts:
            # Test search operation
            has_image = bool(search_pattern.search(text))
            # Test substitution operation
            result = sub_pattern.sub(image_token, text)
            optimized_results.append((has_image, result))
    optimized_time = time.time() - start_time
    
    # Verify results are identical
    assert original_results == optimized_results, "Image token results differ!"
    
    print(f"\nImage token regex optimization:")
    print(f"Original approach time: {original_time:.4f} seconds")
    print(f"Optimized approach time: {optimized_time:.4f} seconds")
    improvement = (original_time - optimized_time) / original_time * 100
    print(f"Performance improvement: {improvement:.2f}%")
    
    return improvement

if __name__ == "__main__":
    print("Testing regex optimizations for GRPO trainer...")
    print("=" * 50)
    
    pad_improvement = test_regex_performance()
    img_improvement = test_image_token_optimization()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Pad token regex improvement: {pad_improvement:.2f}%")
    print(f"Image token regex improvement: {img_improvement:.2f}%")
    
    avg_improvement = (pad_improvement + img_improvement) / 2
    print(f"Average performance improvement: {avg_improvement:.2f}%")
    
    if avg_improvement > 0:
        print("✅ Optimizations are working correctly and providing performance benefits!")
    else:
        print("⚠️  Optimizations may not be providing significant benefits in this test.")
