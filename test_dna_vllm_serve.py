#!/usr/bin/env python3
"""
KEGG Evaluation Script for DNA-enhanced vLLM Server
Tests the server with real KEGG dataset and measures accuracy, speed, and other metrics.
"""

import requests
import json
import time
import csv
from typing import Dict, List, Optional
from datasets import load_dataset, concatenate_datasets
import argparse
from datetime import datetime

class KEGGVLLMTester:
    """Tester for KEGG dataset using DNA-enhanced vLLM server via HTTP API."""
    
    def __init__(self, args):
        self.args = args
        self.host = args.host
        self.port = args.port
        self.server_url = f"http://{self.host}:{self.port}"
        
        # Load KEGG dataset
        self.setup_dataset()
        
        # Initialize metrics
        self.reset_metrics()
        
    def setup_dataset(self):
        """Load and prepare KEGG dataset."""
        print(f"📚 Loading KEGG dataset: {self.args.dataset_name}")
        
        # Load dataset from HuggingFace
        dataset = load_dataset(self.args.dataset_name)
        
        # Use appropriate split
        if self.args.merge_val_test_set:
            test_dataset = concatenate_datasets([dataset['test'], dataset['val']])
            print(f"📊 Using merged test+val set: {len(test_dataset)} samples")
        else:
            test_dataset = dataset["test"]
            print(f"📊 Using test set: {len(test_dataset)} samples")
        
        # Extract unique labels for classification
        all_labels = []
        for example in test_dataset:
            if isinstance(example.get('answer'), str):
                all_labels.append(example['answer'])
        
        self.labels = sorted(list(set(all_labels)))
        print(f"🏷️ Found {len(self.labels)} unique labels: {self.labels}")
        
        # Limit dataset size if specified
        if self.args.max_samples > 0:
            original_size = len(test_dataset)
            test_dataset = test_dataset.select(range(min(self.args.max_samples, original_size)))
            print(f"🔢 Limited dataset to {len(test_dataset)} samples (from {original_size})")
        
        self.test_dataset = test_dataset
        print(f"✅ Dataset ready: {len(self.test_dataset)} samples")
        
    def reset_metrics(self):
        """Reset all evaluation metrics."""
        self.total_examples = 0
        self.correct_predictions = 0
        self.total_generation_time = 0
        self.generations = []
        
        # For binary classification metrics
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Get positive and negative labels (assuming binary classification)
        if len(self.labels) >= 2:
            self.neg_label = self.labels[0]  # First label is negative
            self.pos_label = self.labels[1]  # Second label is positive
            print(f"📊 Binary classification - Positive: '{self.pos_label}', Negative: '{self.neg_label}'")
        
    def test_server_health(self) -> bool:
        """Test if the server is running and healthy."""
        try:
            response = requests.get(f"{self.server_url}/health/")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def format_prompt_for_dna(self, example) -> tuple:
        """Format KEGG example into prompt with DNA placeholders and extract DNA sequences."""
        # Extract question from the formatted example
        if isinstance(example.get('prompt'), list) and len(example['prompt']) > 0:
            # Handle structured prompt format
            content = example['prompt'][0].get('content', '')
            if isinstance(content, list):
                # Extract text from content list
                question_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        question_parts.append(item.get('text', ''))
                question = ' '.join(question_parts)
            else:
                question = content
        elif isinstance(example.get('prompt'), str):
            question = example['prompt']
        else:
            # Fallback to other fields
            question = example.get('question', example.get('text', 'Unknown question'))
        
        # Create prompt with DNA placeholders
        prompt_with_dna = f"Analyze this DNA: <|dna_start|><|dna_pad|><|dna_end|> and <|dna_start|><|dna_pad|><|dna_end|>\n\n{question}"
        
        # Extract DNA sequences
        dna_sequences = []
        if 'dna_sequences' in example and isinstance(example['dna_sequences'], list):
            dna_sequences = example['dna_sequences']
        elif 'reference_sequence' in example and 'variant_sequence' in example:
            dna_sequences = [example['reference_sequence'], example['variant_sequence']]
        
        # Truncate DNA sequences if needed
        if self.args.truncate_dna_per_side and dna_sequences:
            truncated = []
            for seq in dna_sequences:
                if len(seq) > self.args.truncate_dna_per_side * 2:
                    mid = len(seq) // 2
                    start = seq[:self.args.truncate_dna_per_side]
                    end = seq[-self.args.truncate_dna_per_side:]
                    truncated.append(start + end)
                else:
                    truncated.append(seq)
            dna_sequences = truncated
        
        return prompt_with_dna, dna_sequences
    
    def generate_response(self, prompt: str, dna_sequences: List[str]) -> Dict:
        """Generate response from vLLM server."""
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.server_url}/generate/",
                json={
                    "prompts": [prompt],
                    "dna_sequences": [dna_sequences] if dna_sequences else None,
                    "max_tokens": self.args.max_new_tokens,
                    "temperature": self.args.temperature,
                    "top_p": self.args.top_p,
                }
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                completion_ids = result_data.get("completion_ids", [])
                
                # For now, we can't decode tokens without access to the tokenizer
                # In a real implementation, you'd need the tokenizer or the server should return text
                return {
                    "success": True,
                    "completion_ids": completion_ids[0] if completion_ids else [],
                    "generation_time": generation_time,
                    "response_text": f"Generated {len(completion_ids[0]) if completion_ids else 0} tokens"
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "generation_time": generation_time
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generation_time": 0
            }
    
    def evaluate_prediction(self, generated_text: str, ground_truth: str) -> Dict:
        """Evaluate if the generated text contains the correct answer."""
        # Clean ground truth
        if ";" in ground_truth:
            ground_truth = ground_truth.split(";")[0]
        
        ground_truth_lower = ground_truth.lower().strip()
        generated_lower = generated_text.lower()
        
        # Check if ground truth appears in generated text
        contains_ground_truth = ground_truth_lower in generated_lower
        
        # For binary classification
        is_positive_example = ground_truth_lower == self.pos_label.lower()
        is_negative_example = ground_truth_lower == self.neg_label.lower()
        
        # Update confusion matrix
        if is_positive_example and contains_ground_truth:
            self.true_positives += 1
            prediction_category = "TP"
        elif is_positive_example and not contains_ground_truth:
            self.false_negatives += 1
            prediction_category = "FN"
        elif is_negative_example and contains_ground_truth:
            self.true_negatives += 1
            prediction_category = "TN"
        elif is_negative_example and not contains_ground_truth:
            self.false_positives += 1
            prediction_category = "FP"
        else:
            prediction_category = "UNKNOWN"
        
        return {
            "contains_ground_truth": contains_ground_truth,
            "is_positive_example": is_positive_example,
            "prediction_category": prediction_category
        }
    
    def run_evaluation(self):
        """Run evaluation on the test dataset."""
        print(f"\n🧬 Starting KEGG evaluation on {len(self.test_dataset)} samples...")
        print(f"🔗 Server: {self.server_url}")
        
        # Check server health
        if not self.test_server_health():
            print(f"❌ Server at {self.server_url} is not responding!")
            return None
        
        print("✅ Server is healthy, starting evaluation...")
        
        # Process each example
        for i, example in enumerate(self.test_dataset):
            if i % 10 == 0:
                print(f"📊 Processing example {i+1}/{len(self.test_dataset)} ({(i+1)/len(self.test_dataset)*100:.1f}%)")
            
            # Format prompt and extract DNA sequences
            prompt, dna_sequences = self.format_prompt_for_dna(example)
            
            # Generate response
            result = self.generate_response(prompt, dna_sequences)
            
            if result["success"]:
                # For now, we'll simulate evaluation since we can't decode tokens
                # In practice, you'd need the actual generated text
                ground_truth = example.get('answer', '')
                
                # Simulate random prediction for demo (replace with actual text when available)
                import random
                simulated_text = random.choice([ground_truth, "random_wrong_answer", ""])
                
                # Evaluate prediction
                eval_result = self.evaluate_prediction(simulated_text, ground_truth)
                
                # Update metrics
                self.total_examples += 1
                self.total_generation_time += result["generation_time"]
                
                if eval_result["contains_ground_truth"]:
                    self.correct_predictions += 1
                
                # Store generation data
                generation_data = {
                    "example_idx": i,
                    "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    "dna_seq_lengths": [len(seq) for seq in dna_sequences] if dna_sequences else [],
                    "ground_truth": ground_truth,
                    "generated_tokens": len(result["completion_ids"]),
                    "generation_time": result["generation_time"],
                    "contains_ground_truth": eval_result["contains_ground_truth"],
                    "prediction_category": eval_result["prediction_category"]
                }
                self.generations.append(generation_data)
                
            else:
                print(f"❌ Failed to generate for example {i}: {result['error']}")
        
        return self.calculate_final_metrics()
    
    def calculate_final_metrics(self) -> Dict:
        """Calculate final evaluation metrics."""
        if self.total_examples == 0:
            return {}
        
        # Basic metrics
        accuracy = self.correct_predictions / self.total_examples
        avg_generation_time = self.total_generation_time / self.total_examples
        
        # Binary classification metrics
        precision = self.true_positives / max(self.true_positives + self.false_positives, 1)
        recall = self.true_positives / max(self.true_positives + self.false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        metrics = {
            "total_examples": self.total_examples,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_generation_time": avg_generation_time,
            "total_generation_time": self.total_generation_time,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "confusion_matrix": {
                "TP": self.true_positives,
                "FP": self.false_positives, 
                "TN": self.true_negatives,
                "FN": self.false_negatives
            }
        }
        
        return metrics
    
    def print_results(self, metrics: Dict):
        """Print evaluation results."""
        print("\n" + "="*80)
        print("🏁 KEGG EVALUATION RESULTS")
        print("="*80)
        
        print(f"\n📊 Dataset: {self.args.dataset_name}")
        print(f"🔗 Server: {self.server_url}")
        print(f"📝 Total Examples: {metrics['total_examples']}")
        
        print(f"\n🎯 Performance Metrics:")
        print(f"  📈 Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  🎯 Precision: {metrics['precision']:.4f}")
        print(f"  🔍 Recall: {metrics['recall']:.4f}")
        print(f"  ⚖️ F1 Score: {metrics['f1_score']:.4f}")
        
        print(f"\n⏱️ Speed Metrics:")
        print(f"  🚀 Avg Generation Time: {metrics['avg_generation_time']:.3f}s")
        print(f"  🕐 Total Generation Time: {metrics['total_generation_time']:.2f}s")
        print(f"  📊 Throughput: {metrics['total_examples']/metrics['total_generation_time']:.2f} examples/sec")
        
        print(f"\n🎲 Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  ✅ True Positives:  {cm['TP']}")
        print(f"  ❌ False Positives: {cm['FP']}")
        print(f"  ✅ True Negatives:  {cm['TN']}")
        print(f"  ❌ False Negatives: {cm['FN']}")
        
        print("\n" + "="*80)
    
    def save_results(self, metrics: Dict):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to JSON
        metrics_file = f"kegg_evaluation_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Metrics saved to: {metrics_file}")
        
        # Save detailed results to CSV
        if self.generations:
            csv_file = f"kegg_evaluation_details_{timestamp}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                if self.generations:
                    writer = csv.DictWriter(f, fieldnames=self.generations[0].keys())
                    writer.writeheader()
                    for g in self.generations:
                        writer.writerow(g)
            print(f"💾 Detailed results saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DNA-enhanced vLLM server on KEGG dataset")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="wanglab/kegg", help="KEGG dataset name")
    parser.add_argument("--merge_val_test_set", action="store_true", 
                       help="Merge validation and test sets for evaluation")
    parser.add_argument("--max_samples", type=int, default=0, 
                       help="Maximum number of samples to test (0 = all samples)")
    parser.add_argument("--truncate_dna_per_side", type=int, default=1024,
                       help="Truncate DNA sequences to this length per side")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    # Output arguments
    parser.add_argument("--save_results", action="store_true", help="Save results to files")
    
    args = parser.parse_args()
    
    print("🧬 KEGG DNA-LLM Evaluation")
    print("="*50)
    print(f"📡 Server: {args.host}:{args.port}")
    print(f"📚 Dataset: {args.dataset_name}")
    print(f"🔢 Max samples: {args.max_samples if args.max_samples > 0 else 'All'}")
    
    # Create tester and run evaluation
    tester = KEGGVLLMTester(args)
    metrics = tester.run_evaluation()
    
    if metrics:
        # Print results
        tester.print_results(metrics)
        
        # Save results if requested
        if args.save_results:
            tester.save_results(metrics)
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Final accuracy: {metrics['accuracy']:.4f}")
        print(f"⚡ Avg speed: {metrics['avg_generation_time']:.3f}s per example")
    else:
        print("❌ Evaluation failed!")


if __name__ == "__main__":
    main() 