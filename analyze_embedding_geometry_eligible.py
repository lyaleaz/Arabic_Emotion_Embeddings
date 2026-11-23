
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_984_eligible_farasa"

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
# 2. Geometry Analysis
# ==========================================

def analyze_geometry(model_name, filename):

    print("\n" + "=" * 55)
    print("Analyzing:", model_name)
    print("=" * 55)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    ).astype(np.float64)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values.")

    n, dimensions = embeddings.shape

    print("Records:", n)
    print("Dimensions:", dimensions)

    # --------------------------------------
    # Vector Norms
    # --------------------------------------

    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms < 1e-12):
        raise ValueError("Zero-length embedding detected.")

    normalized = embeddings / norms[:, None]

    print("Mean individual norm:", np.mean(norms))

    # --------------------------------------
    # Mean Vector Direction
    # --------------------------------------

    mean_vector = normalized.mean(axis=0)

    mean_vector_norm = np.linalg.norm(mean_vector)

    # --------------------------------------
    # Mean Pairwise Cosine Similarity
    # --------------------------------------

    similarity_matrix = normalized @ normalized.T

    upper_indices = np.triu_indices(n, k=1)

    similarities = similarity_matrix[upper_indices]

    mean_similarity = np.mean(similarities)

    median_similarity = np.median(similarities)

    # --------------------------------------
    # Centered SVD
    # --------------------------------------

    centered = normalized - mean_vector

    singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False
    )

    squared_values = singular_values ** 2

    total_variance = squared_values.sum()

    if total_variance <= 0:
        raise ValueError("Total variance is zero.")

    variance_ratios = squared_values / total_variance

    # --------------------------------------
    # Effective Rank
    # --------------------------------------

    positive = variance_ratios > 0

    entropy = -np.sum(
        variance_ratios[positive] *
        np.log(variance_ratios[positive])
    )

    effective_rank = np.exp(entropy)

    effective_rank_ratio = effective_rank / dimensions

    # --------------------------------------
    # Summary
    # --------------------------------------

    summary = {
        "model": model_name,
        "records": n,
        "dimensions": dimensions,
        "mean_vector_norm": mean_vector_norm,
        "mean_pairwise_cosine": mean_similarity,
        "median_pairwise_cosine": median_similarity,
        "std_pairwise_cosine": np.std(similarities),
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank_ratio,
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
    # Save Pairwise Similarities
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
    # Save Singular Values
    # --------------------------------------

    pd.DataFrame({
        "component": np.arange(
            1, len(singular_values) + 1
        ),
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
print("ELIGIBLE PILOT GEOMETRY ANALYSIS COMPLETED")
print("=" * 55)

print(report.to_string(index=False))

print("\nResults saved in:", OUTPUT_DIR)
