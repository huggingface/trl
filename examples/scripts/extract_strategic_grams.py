# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "sentence-transformers",
#     "scikit-learn",
#     "numpy",
# ]
# ///

"""
# Strategic Gram Extraction Example

This script demonstrates how to extract Strategic Grams (SGs) from a corpus of reasoning solutions.
Strategic Grams are n-grams that function as high-level strategic moves in reasoning, such as
"let's try a different approach" or "notice that".

## Basic Usage

Extract Strategic Grams from a dataset:

```bash
python examples/scripts/extract_strategic_grams.py \
    --dataset_name trl-lib/DeepMath-103K \
    --output_path strategic_grams_math.json \
    --n_min 3 \
    --n_max 5 \
    --n_clusters 100 \
    --top_percentile 0.2
```

## Advanced Usage

Use a custom corpus file:

```bash
python examples/scripts/extract_strategic_grams.py \
    --corpus_file my_reasoning_corpus.txt \
    --output_path strategic_grams_custom.json \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --n_clusters 150
```

## Parameters

- `dataset_name`: HuggingFace dataset to extract from (e.g., "trl-lib/DeepMath-103K")
- `corpus_file`: Path to a text file with one solution per line (alternative to dataset_name)
- `output_path`: Path to save the extracted Strategic Grams JSON file
- `n_min`: Minimum n-gram length (default: 3)
- `n_max`: Maximum n-gram length (default: 5)
- `n_clusters`: Number of semantic clusters for KMeans (default: 100)
- `top_percentile`: Top percentile of clusters by CDF to select (default: 0.2)
- `embedding_model`: Sentence transformer model for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")
- `max_samples`: Maximum number of samples to process (default: None, use all)

## Output Format

The script saves a JSON file with the following structure:

```json
{
  "strategic_grams": [
    "let's try a different approach",
    "we can use the fact that",
    "notice that",
    ...
  ],
  "metadata": {
    "domain": "math",
    "n_range": [3, 5],
    "n_clusters": 100,
    "top_percentile": 0.2,
    "corpus_size": 10000
  }
}
```

## Using Extracted Strategic Grams

Load and use the extracted Strategic Grams in HICRA training:

```python
from trl import HICRAConfig, HICRATrainer

config = HICRAConfig(
    strategic_grams_path="strategic_grams_math.json",
    use_planning_tokens=True,
    hicra_alpha=0.2,
)

trainer = HICRATrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=[accuracy_reward],
    args=config,
    train_dataset=dataset,
)
```

## Example Corpus

If you don't have a corpus, here's a minimal example to get started:

```python
# Create a small example corpus
corpus = [
    "Let's start by analyzing the problem. We need to find the value of x.",
    "Notice that the equation can be simplified. Let's factor it.",
    "Wait, this approach isn't working. Let's try a different method.",
    "We can use the quadratic formula here. First, identify a, b, and c.",
    "The key insight is that this is a geometric series. Let's apply the formula.",
]

# Save to file
with open("example_corpus.txt", "w") as f:
    for solution in corpus:
        f.write(solution + "\\n")

# Extract Strategic Grams
python examples/scripts/extract_strategic_grams.py \
    --corpus_file example_corpus.txt \
    --output_path strategic_grams_example.json
```

## References

- Paper: "Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning" (arXiv:2509.03646)
- Strategic Grams are identified using semantic clustering with Cluster Document Frequency (CDF)
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def extract_ngrams(text, n_min=3, n_max=5):
    """Extract n-grams from text."""
    tokens = text.split()
    ngrams = []
    for n in range(n_min, n_max + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            ngrams.append(ngram)
    return ngrams


def extract_strategic_grams(
    corpus,
    n_min=3,
    n_max=5,
    top_percentile=0.2,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    n_clusters=100,
):
    """
    Extract Strategic Grams from a corpus using semantic clustering.

    Args:
        corpus: List of text documents (reasoning solutions)
        n_min: Minimum n-gram length
        n_max: Maximum n-gram length
        top_percentile: Top percentile of clusters by CDF to select
        embedding_model: Sentence transformer model name
        n_clusters: Number of clusters for KMeans

    Returns:
        List of Strategic Grams (strings)
    """
    print(f"📊 Processing {len(corpus)} documents...")

    # Step 1: Extract n-grams
    print(f"🔍 Extracting n-grams (n ∈ [{n_min}, {n_max}])...")
    all_ngrams = []
    ngram_to_docs = defaultdict(set)

    for doc_idx, doc in enumerate(corpus):
        doc_ngrams = extract_ngrams(doc, n_min, n_max)
        for ngram in doc_ngrams:
            all_ngrams.append(ngram)
            ngram_to_docs[ngram].add(doc_idx)

    unique_ngrams = list(set(all_ngrams))
    print(f"   Found {len(unique_ngrams)} unique n-grams")

    # Step 2: Embed n-grams
    print(f"🧠 Embedding n-grams using {embedding_model}...")
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(unique_ngrams, show_progress_bar=True, batch_size=32)

    # Step 3: Cluster embeddings
    print(f"🎯 Clustering into {n_clusters} semantic clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Step 4: Compute Cluster Document Frequency
    print("📈 Computing Cluster Document Frequency...")
    cluster_to_ngrams = defaultdict(list)
    for ngram, label in zip(unique_ngrams, cluster_labels, strict=False):
        cluster_to_ngrams[label].append(ngram)

    cluster_cdf = {}
    for cluster_id, ngrams in cluster_to_ngrams.items():
        # Union of all documents containing any n-gram in this cluster
        docs = set()
        for ngram in ngrams:
            docs.update(ngram_to_docs[ngram])
        cluster_cdf[cluster_id] = len(docs)

    # Step 5: Select top clusters
    print(f"✨ Selecting top {top_percentile * 100}% of clusters...")
    threshold = np.percentile(list(cluster_cdf.values()), (1 - top_percentile) * 100)
    selected_clusters = [cid for cid, cdf in cluster_cdf.items() if cdf >= threshold]

    print(f"   Selected {len(selected_clusters)} clusters (CDF threshold: {threshold:.1f})")

    # Step 6: Return Strategic Grams
    strategic_grams = []
    for cluster_id in selected_clusters:
        strategic_grams.extend(cluster_to_ngrams[cluster_id])

    strategic_grams = list(set(strategic_grams))
    print(f"✅ Extracted {len(strategic_grams)} Strategic Grams")

    return strategic_grams


def load_corpus_from_dataset(dataset_name, split="train", text_column="answer", max_samples=None):
    """Load corpus from a HuggingFace dataset."""
    print(f"📚 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Try common column names for text
    possible_columns = [text_column, "text", "completion", "response", "solution"]
    text_col = None
    for col in possible_columns:
        if col in dataset.column_names:
            text_col = col
            break

    if text_col is None:
        raise ValueError(
            f"Could not find text column in dataset. Available columns: {dataset.column_names}. "
            f"Please specify the correct column using --text_column"
        )

    print(f"   Using column: {text_col}")
    corpus = [example[text_col] for example in dataset if example[text_col]]
    return corpus


def load_corpus_from_file(file_path):
    """Load corpus from a text file (one document per line)."""
    print(f"📄 Loading corpus from file: {file_path}")
    with open(file_path, encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]
    return corpus


def save_strategic_grams(strategic_grams, output_path, metadata=None):
    """Save Strategic Grams to a JSON file."""
    data = {
        "strategic_grams": strategic_grams,
        "metadata": metadata or {},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved Strategic Grams to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract Strategic Grams from a corpus")

    # Input options
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., 'trl-lib/DeepMath-103K')",
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        default=None,
        help="Path to corpus file (one document per line)",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="answer",
        help="Column name containing text in the dataset",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )

    # Output options
    parser.add_argument(
        "--output_path",
        type=str,
        default="strategic_grams.json",
        help="Path to save extracted Strategic Grams",
    )

    # Extraction parameters
    parser.add_argument("--n_min", type=int, default=3, help="Minimum n-gram length")
    parser.add_argument("--n_max", type=int, default=5, help="Maximum n-gram length")
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=100,
        help="Number of semantic clusters",
    )
    parser.add_argument(
        "--top_percentile",
        type=float,
        default=0.2,
        help="Top percentile of clusters to select",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings",
    )

    args = parser.parse_args()

    # Validate input
    if not args.dataset_name and not args.corpus_file:
        parser.error("Either --dataset_name or --corpus_file must be provided")

    # Load corpus
    if args.corpus_file:
        corpus = load_corpus_from_file(args.corpus_file)
        domain = "custom"
    else:
        corpus = load_corpus_from_dataset(
            args.dataset_name,
            text_column=args.text_column,
            max_samples=args.max_samples,
        )
        # Infer domain from dataset name
        domain = (
            "math"
            if "math" in args.dataset_name.lower()
            else "code"
            if "code" in args.dataset_name.lower()
            else "general"
        )

    # Extract Strategic Grams
    strategic_grams = extract_strategic_grams(
        corpus=corpus,
        n_min=args.n_min,
        n_max=args.n_max,
        top_percentile=args.top_percentile,
        embedding_model=args.embedding_model,
        n_clusters=args.n_clusters,
    )

    # Save results
    metadata = {
        "domain": domain,
        "n_range": [args.n_min, args.n_max],
        "n_clusters": args.n_clusters,
        "top_percentile": args.top_percentile,
        "corpus_size": len(corpus),
        "embedding_model": args.embedding_model,
    }

    save_strategic_grams(strategic_grams, args.output_path, metadata)

    # Print sample Strategic Grams
    print("\n📝 Sample Strategic Grams:")
    for i, sg in enumerate(strategic_grams[:10], 1):
        print(f"   {i}. {sg}")

    if len(strategic_grams) > 10:
        print(f"   ... and {len(strategic_grams) - 10} more")


if __name__ == "__main__":
    main()
