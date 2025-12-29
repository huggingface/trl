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

# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Strategic Gram utilities for HICRA (Hierarchy-Aware Credit Assignment).

Strategic Grams (SGs) are n-grams that function as high-level strategic moves
in reasoning tasks. This module provides utilities for extracting, loading,
and managing Strategic Grams.
"""

import json
from collections import defaultdict

import numpy as np


def extract_strategic_grams(
    corpus: list[str],
    n_range: tuple[int, int] = (3, 5),
    top_percentile: float = 0.2,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    clustering_method: str = "kmeans",
    n_clusters: int = 100,
) -> list[str]:
    """
    Extract Strategic Grams from a corpus using semantic clustering.

    This function implements the Strategic Gram extraction pipeline from the
    HICRA paper (arXiv:2509.03646). It extracts n-grams from a corpus, embeds
    them semantically, clusters them, and selects clusters with high Cluster
    Document Frequency (CDF).

    Args:
        corpus: List of text documents (e.g., reasoning solutions).
        n_range: Tuple of (min_n, max_n) for n-gram extraction. Default: (3, 5).
        top_percentile: Percentile threshold for CDF selection. Default: 0.2 (top 20%).
        embedding_model: Sentence transformer model for embedding n-grams.
            Default: "sentence-transformers/all-MiniLM-L6-v2".
        clustering_method: Clustering algorithm to use. Default: "kmeans".
        n_clusters: Number of clusters for semantic grouping. Default: 100.

    Returns:
        List of Strategic Gram strings selected from top CDF clusters.

    Example:
        >>> corpus = [
        ...     "Let's try a different approach. We can use algebra.",
        ...     "Notice that the pattern repeats. Let's verify this.",
        ... ]
        >>> sgs = extract_strategic_grams(corpus, n_range=(3, 4))
        >>> print(sgs[:3])
        ['try a different', 'a different approach', 'notice that the']
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for Strategic Gram extraction. "
            "Install it with: pip install sentence-transformers"
        ) from e

    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for Strategic Gram extraction. Install it with: pip install scikit-learn"
        ) from e

    # Step 1: Extract n-grams from corpus
    all_ngrams = []
    ngram_to_docs = defaultdict(set)

    for doc_idx, doc in enumerate(corpus):
        tokens = doc.split()
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i : i + n])
                all_ngrams.append(ngram)
                ngram_to_docs[ngram].add(doc_idx)

    unique_ngrams = list(set(all_ngrams))

    if not unique_ngrams:
        return []

    # Step 2: Embed n-grams using sentence transformers
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(unique_ngrams, show_progress_bar=True)

    # Step 3: Cluster embeddings semantically
    if clustering_method == "kmeans":
        # Adjust n_clusters if we have fewer unique n-grams
        actual_n_clusters = min(n_clusters, len(unique_ngrams))
        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")

    # Step 4: Compute Cluster Document Frequency (CDF)
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

    # Step 5: Select clusters in top percentile of CDF
    if not cluster_cdf:
        return []

    threshold = np.percentile(list(cluster_cdf.values()), (1 - top_percentile) * 100)
    selected_clusters = [cid for cid, cdf in cluster_cdf.items() if cdf >= threshold]

    # Step 6: Return Strategic Grams from selected clusters
    strategic_grams = []
    for cluster_id in selected_clusters:
        strategic_grams.extend(cluster_to_ngrams[cluster_id])

    return list(set(strategic_grams))


def get_default_strategic_grams(domain: str = "math") -> list[str]:
    """
    Get pre-computed Strategic Grams for common reasoning domains.

    This function returns a curated set of Strategic Grams that have been
    identified as effective for specific reasoning domains. These SGs capture
    high-level strategic reasoning patterns.

    Args:
        domain: The reasoning domain. Supported: "math", "code". Default: "math".

    Returns:
        List of Strategic Gram strings for the specified domain.

    Raises:
        ValueError: If the domain is not supported.

    Example:
        >>> math_sgs = get_default_strategic_grams("math")
        >>> print(math_sgs[:3])
        ["let's try a different approach", "we can use the fact that", "notice that"]
    """
    DEFAULT_SG_SETS = {
        "math": [
            "let's try a different approach",
            "we can use the fact that",
            "notice that",
            "this suggests that",
            "alternatively we could",
            "let's consider the case",
            "wait this doesn't work",
            "going back to",
            "let's verify this",
            "we need to find",
            "the key insight is",
            "let's break this down",
            "first we need to",
            "this means that",
            "therefore we can conclude",
            "let's think about",
            "we should check",
            "this implies that",
            "let's start by",
            "we can simplify",
            "let's substitute",
            "we can rewrite",
            "this gives us",
            "let's solve for",
            "we can factor",
            "this reduces to",
            "let's apply",
            "we can use",
            "this shows that",
            "let's verify",
        ],
        "code": [
            "let's start by",
            "we need to handle",
            "the approach is to",
            "alternatively we can",
            "this suggests using",
            "let's consider",
            "we should check",
            "the key is to",
            "first we need",
            "this means we",
            "let's implement",
            "we can use",
            "this requires",
            "let's define",
            "we should validate",
            "this will allow",
            "let's optimize",
            "we can refactor",
            "this handles",
            "let's test",
        ],
    }

    if domain not in DEFAULT_SG_SETS:
        raise ValueError(f"Unsupported domain: {domain}. Supported domains: {list(DEFAULT_SG_SETS.keys())}")

    return DEFAULT_SG_SETS[domain]


def load_strategic_grams_from_file(path: str) -> list[str]:
    """
    Load Strategic Grams from a JSON file.

    Args:
        path: Path to the JSON file containing Strategic Grams.
            The file should contain a JSON array of strings.

    Returns:
        List of Strategic Gram strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the file does not contain a list of strings.

    Example:
        >>> sgs = load_strategic_grams_from_file("my_strategic_grams.json")
        >>> print(len(sgs))
        50
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of strings, got {type(data)}")

    if not all(isinstance(item, str) for item in data):
        raise ValueError("All items in the Strategic Grams file must be strings")

    return data


def save_strategic_grams_to_file(strategic_grams: list[str], path: str) -> None:
    """
    Save Strategic Grams to a JSON file.

    Args:
        strategic_grams: List of Strategic Gram strings to save.
        path: Path where the JSON file will be saved.

    Example:
        >>> sgs = ["let's try", "we can use", "notice that"]
        >>> save_strategic_grams_to_file(sgs, "my_strategic_grams.json")
    """
    with open(path, "w") as f:
        json.dump(strategic_grams, f, indent=2)
