
import os
import numpy as np
import pandas as pd

from itertools import combinations

# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_1000_results_farasa"

OUTPUT_DIR = os.path.join(
    FOLDER,
    "cosine_similarity"
)

MODELS = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Load Metadata
# ==========================================

metadata = pd.read_csv(
    os.path.join(FOLDER, "pilot_metadata.csv"),
    encoding="utf-8-sig"
).reset_index(drop=True)

required_columns = [
    "source_dataset",
    "emotion"
]

for column in required_columns:
    if column not in metadata.columns:
        raise ValueError(f"Missing column: {column}")

if metadata[required_columns].isna().any().any():
    raise ValueError("Missing category labels.")

print("Number of records:", len(metadata))
print("Number of emotions:", metadata["emotion"].nunique())
print("Number of datasets:", metadata["source_dataset"].nunique())

# ==========================================
# 3. Helper Function
# ==========================================

def calculate_statistics(values):

    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {
            "pairs": 0,
            "mean_similarity": np.nan,
            "median_similarity": np.nan,
            "std_similarity": np.nan
        }

    return {
        "pairs": len(values),
        "mean_similarity": float(np.mean(values)),
        "median_similarity": float(np.median(values)),
        "std_similarity": float(np.std(values))
    }


# ==========================================
# 4. Results Containers
# ==========================================

overall_results = []
emotion_pair_results = []
dataset_pair_results = []

# ==========================================
# 5. Process Both Models
# ==========================================

for model_name, filename in MODELS.items():

    print("\n" + "=" * 60)
    print("Analyzing:", model_name)
    print("=" * 60)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    ).astype(np.float64)

    # Verify dimensions and alignment

    if embeddings.shape != (len(metadata), 768):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("Embeddings are not L2-normalized.")

    print("Embedding shape:", embeddings.shape)
    print("All vectors normalized:", True)

    # ======================================
    # 6. Pairwise Cosine Similarity
    # ======================================

    similarity_matrix = embeddings @ embeddings.T

    upper_i, upper_j = np.triu_indices(
        len(metadata),
        k=1
    )

    similarities = similarity_matrix[upper_i, upper_j]

    emotion_i = metadata["emotion"].to_numpy()[upper_i]
    emotion_j = metadata["emotion"].to_numpy()[upper_j]

    dataset_i = metadata["source_dataset"].to_numpy()[upper_i]
    dataset_j = metadata["source_dataset"].to_numpy()[upper_j]

    same_emotion_mask = emotion_i == emotion_j
    same_dataset_mask = dataset_i == dataset_j

    # ======================================
    # 7. Overall Similarity Analysis
    # ======================================

    groups = {
        "Same Emotion": similarities[same_emotion_mask],
        "Different Emotion": similarities[~same_emotion_mask],
        "Same Dataset": similarities[same_dataset_mask],
        "Different Dataset": similarities[~same_dataset_mask]
    }

    print("\nOverall Similarity Results:")

    model_summary = {}

    for group_name, values in groups.items():

        stats = calculate_statistics(values)

        overall_results.append({
            "model": model_name,
            "comparison": group_name,
            **stats
        })

        model_summary[group_name] = stats["mean_similarity"]

        print(
            group_name,
            "| Pairs:", stats["pairs"],
            "| Mean:", round(stats["mean_similarity"], 6),
            "| Median:", round(stats["median_similarity"], 6),
            "| STD:", round(stats["std_similarity"], 6)
        )

    # Emotion separability gap

    emotion_gap = (
        model_summary["Same Emotion"]
        - model_summary["Different Emotion"]
    )

    # Dataset similarity gap

    dataset_gap = (
        model_summary["Same Dataset"]
        - model_summary["Different Dataset"]
    )

    print("\nEmotion Separability Gap:", round(emotion_gap, 6))
    print("Dataset Similarity Gap:", round(dataset_gap, 6))

    # ======================================
    # 8. Emotion Pair Analysis
    # ======================================

    emotion_groups = {}

    for emotion_1, emotion_2 in combinations(
        sorted(metadata["emotion"].unique()),
        2
    ):

        mask = (
            ((emotion_i == emotion_1) & (emotion_j == emotion_2))
            |
            ((emotion_i == emotion_2) & (emotion_j == emotion_1))
        )

        emotion_groups[(emotion_1, emotion_2)] = similarities[mask]

    # Also calculate within-emotion similarities

    for emotion in sorted(metadata["emotion"].unique()):

        mask = (emotion_i == emotion) & (emotion_j == emotion)

        emotion_groups[(emotion, emotion)] = similarities[mask]

    for (emotion_1, emotion_2), values in emotion_groups.items():

        stats = calculate_statistics(values)

        emotion_pair_results.append({
            "model": model_name,
            "emotion_1": emotion_1,
            "emotion_2": emotion_2,
            "comparison_type": (
                "Intra-emotion"
                if emotion_1 == emotion_2
                else "Inter-emotion"
            ),
            **stats
        })

    # ======================================
    # 9. Dataset Pair Analysis
    # ======================================

    dataset_groups = {}

    for dataset_1, dataset_2 in combinations(
        sorted(metadata["source_dataset"].unique()),
        2
    ):

        mask = (
            ((dataset_i == dataset_1) & (dataset_j == dataset_2))
            |
            ((dataset_i == dataset_2) & (dataset_j == dataset_1))
        )

        dataset_groups[(dataset_1, dataset_2)] = similarities[mask]

    # Within-dataset similarities

    for dataset in sorted(metadata["source_dataset"].unique()):

        mask = (dataset_i == dataset) & (dataset_j == dataset)

        dataset_groups[(dataset, dataset)] = similarities[mask]

    for (dataset_1, dataset_2), values in dataset_groups.items():

        stats = calculate_statistics(values)

        dataset_pair_results.append({
            "model": model_name,
            "dataset_1": dataset_1,
            "dataset_2": dataset_2,
            "comparison_type": (
                "Within-dataset"
                if dataset_1 == dataset_2
                else "Between-dataset"
            ),
            **stats
        })

# ==========================================
# 10. Save Reports
# ==========================================

overall_report = pd.DataFrame(overall_results)

emotion_report = pd.DataFrame(emotion_pair_results)

dataset_report = pd.DataFrame(dataset_pair_results)

overall_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_similarity_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

emotion_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "emotion_pair_similarity.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

dataset_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_pair_similarity.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 11. Final Verification
# ==========================================

print("\n" + "=" * 60)
print("COSINE SIMILARITY ANALYSIS COMPLETED")
print("=" * 60)

print("\nOverall Results:")
print(overall_report.to_string(index=False))

print("\nEmotion Pair Results:", len(emotion_report))
print("Dataset Pair Results:", len(dataset_report))

print("\nAll reports saved in:", OUTPUT_DIR)
