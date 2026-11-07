
import os
import numpy as np
import pandas as pd

from itertools import combinations


# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_results_farasa"

OUTPUT_DIR = os.path.join(
    FOLDER,
    "embedding_geometry"
)

MODELS = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. Analyze Embedding Geometry
# ==========================================

def analyze_geometry(model_name, filename):

    print("\n" + "=" * 55)
    print("Analyzing:", model_name)
    print("=" * 55)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    ).astype(np.float64)

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values detected.")

    n, dimensions = embeddings.shape

    print("Number of records:", n)
    print("Dimensions:", dimensions)

    # --------------------------------------
    # Vector norms
    # --------------------------------------

    norms = np.linalg.norm(embeddings, axis=1)

    # --------------------------------------
    # Mean vector direction
    # --------------------------------------

    mean_vector = embeddings.mean(axis=0)

    mean_vector_norm = np.linalg.norm(mean_vector)

    # --------------------------------------
    # Pairwise cosine similarity
    # --------------------------------------

    normalized = embeddings / np.maximum(
        norms[:, None], 1e-12
    )

    similarity_matrix = normalized @ normalized.T

    upper_indices = np.triu_indices(n, k=1)

    similarities = similarity_matrix[upper_indices]

    # --------------------------------------
    # Centered SVD
    # --------------------------------------

    centered = embeddings - mean_vector

    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False
    )

    probabilities = (
        singular_values ** 2
    )

    probabilities /= probabilities.sum()

    # Effective rank
    entropy = -np.sum(
        probabilities * np.log(
            probabilities + 1e-12
        )
    )

    effective_rank = np.exp(entropy)

    # Explained variance
    variance_ratios = probabilities

    # --------------------------------------
    # Results
    # --------------------------------------

    summary = {
        "model": model_name,
        "records": n,
        "dimensions": dimensions,
        "mean_vector_norm": mean_vector_norm,
        "mean_pairwise_cosine": np.mean(similarities),
        "median_pairwise_cosine": np.median(similarities),
        "std_pairwise_cosine": np.std(similarities),
        "min_pairwise_cosine": np.min(similarities),
        "max_pairwise_cosine": np.max(similarities),
        "mean_vector_norm_individual": np.mean(norms),
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / dimensions,
        "first_component_variance": variance_ratios[0],
        "top_5_components_variance": np.sum(
            variance_ratios[:5]
        ),
        "top_10_components_variance": np.sum(
            variance_ratios[:10]
        )
    }

    print("\nGeometry Results:")

    for key, value in summary.items():

        if isinstance(value, (float, np.floating)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    # --------------------------------------
    # Save pairwise similarities
    # --------------------------------------

    pd.DataFrame({
        "cosine_similarity": similarities
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_pairwise_cosines.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------
    # Save singular values
    # --------------------------------------

    pd.DataFrame({
        "component": np.arange(1, len(singular_values) + 1),
        "singular_value": singular_values,
        "variance_ratio": variance_ratios
    }).to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_singular_values.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    return summary


# ==========================================
# 3. Process Both Models
# ==========================================

all_results = []

for model_name, filename in MODELS.items():

    result = analyze_geometry(
        model_name,
        filename
    )

    all_results.append(result)


# ==========================================
# 4. Save Comparison Report
# ==========================================

report = pd.DataFrame(all_results)

report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "embedding_geometry_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 5. Final Verification
# ==========================================

print("\n" + "=" * 55)
print("EMBEDDING GEOMETRY ANALYSIS COMPLETED")
print("=" * 55)

print(report.to_string(index=False))

print("\nResults saved in:", OUTPUT_DIR)
