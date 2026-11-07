
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
    "controlled_emotion_separability"
)

MODELS = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

MIN_DATASET_SIZE = 20
MIN_SAMPLES_PER_EMOTION = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Load Metadata
# ==========================================

metadata = pd.read_csv(
    os.path.join(FOLDER, "pilot_metadata.csv"),
    encoding="utf-8-sig"
).reset_index(drop=True)

required_columns = ["source_dataset", "emotion"]

for column in required_columns:
    if column not in metadata.columns:
        raise ValueError(f"Missing column: {column}")

if metadata[required_columns].isna().any().any():
    raise ValueError("Missing dataset or emotion labels.")

print("Total records:", len(metadata))
print("Datasets:", metadata["source_dataset"].nunique())

# ==========================================
# 3. Result Containers
# ==========================================

dataset_results = []
emotion_pair_results = []
model_summary_results = []

# ==========================================
# 4. Process Each Model
# ==========================================

for model_name, filename in MODELS.items():

    print("\n" + "=" * 65)
    print("Processing:", model_name)
    print("=" * 65)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    ).astype(np.float64)

    if len(embeddings) != len(metadata):
        raise ValueError("Metadata and embeddings mismatch.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms < 1e-12):
        raise ValueError("Zero-length embedding detected.")

    embeddings = embeddings / norms[:, None]

    model_dataset_gaps = []

    pooled_same = []
    pooled_different = []

    # ======================================
    # 5. Analyze Each Dataset Independently
    # ======================================

    for dataset_name, indices in metadata.groupby(
        "source_dataset",
        sort=True
    ).groups.items():

        indices = np.array(list(indices))

        dataset_embeddings = embeddings[indices]

        labels = (
            metadata.loc[indices, "emotion"]
            .astype(str)
            .to_numpy()
        )

        n = len(indices)

        emotion_counts = pd.Series(labels).value_counts()

        n_emotions = len(emotion_counts)

        # Pairwise cosine similarity
        similarity_matrix = (
            dataset_embeddings @ dataset_embeddings.T
        )

        upper_i, upper_j = np.triu_indices(n, k=1)

        similarities = similarity_matrix[upper_i, upper_j]

        labels_i = labels[upper_i]
        labels_j = labels[upper_j]

        same_mask = labels_i == labels_j
        different_mask = ~same_mask

        same_values = similarities[same_mask]
        different_values = similarities[different_mask]

        same_mean = (
            float(np.mean(same_values))
            if len(same_values) else np.nan
        )

        different_mean = (
            float(np.mean(different_values))
            if len(different_values) else np.nan
        )

        gap = (
            same_mean - different_mean
            if len(same_values) and len(different_values)
            else np.nan
        )

        # Eligibility for macro comparison
        sufficiently_represented_emotions = int(
            (emotion_counts >= MIN_SAMPLES_PER_EMOTION).sum()
        )

        eligible = (
            n >= MIN_DATASET_SIZE
            and sufficiently_represented_emotions >= 2
            and len(same_values) > 0
            and len(different_values) > 0
        )

        status = (
            "Eligible for macro comparison"
            if eligible
            else "Descriptive only - insufficient representation"
        )

        dataset_results.append({
            "model": model_name,
            "source_dataset": dataset_name,
            "number_of_texts": n,
            "number_of_emotions": n_emotions,
            "emotions_with_at_least_2_texts":
                sufficiently_represented_emotions,
            "same_emotion_pairs": len(same_values),
            "different_emotion_pairs": len(different_values),
            "same_emotion_mean": same_mean,
            "different_emotion_mean": different_mean,
            "emotion_separability_gap": gap,
            "evaluation_status": status
        })

        if eligible:
            model_dataset_gaps.append(gap)
            pooled_same.extend(same_values.tolist())
            pooled_different.extend(different_values.tolist())

        print("\nDataset:", dataset_name)
        print("Texts:", n)
        print("Emotions:", n_emotions)
        print("Same emotion mean:", same_mean)
        print("Different emotion mean:", different_mean)
        print("Separability gap:", gap)
        print("Status:", status)

        # ==================================
        # 6. Emotion Pair Analysis
        # ==================================

        unique_emotions = sorted(emotion_counts.index)

        for emotion_1, emotion_2 in combinations(
            unique_emotions, 2
        ):

            mask = (
                ((labels_i == emotion_1) & (labels_j == emotion_2))
                |
                ((labels_i == emotion_2) & (labels_j == emotion_1))
            )

            values = similarities[mask]

            emotion_pair_results.append({
                "model": model_name,
                "source_dataset": dataset_name,
                "emotion_1": emotion_1,
                "emotion_2": emotion_2,
                "number_of_pairs": len(values),
                "mean_similarity": (
                    float(np.mean(values))
                    if len(values) else np.nan
                ),
                "median_similarity": (
                    float(np.median(values))
                    if len(values) else np.nan
                ),
                "evaluation_status": (
                    "Evaluated" if len(values)
                    else "Insufficient pairs"
                )
            })

    # ======================================
    # 7. Model-Level Summary
    # ======================================

    pooled_same_mean = (
        float(np.mean(pooled_same))
        if pooled_same else np.nan
    )

    pooled_different_mean = (
        float(np.mean(pooled_different))
        if pooled_different else np.nan
    )

    pooled_gap = (
        pooled_same_mean - pooled_different_mean
        if pooled_same and pooled_different
        else np.nan
    )

    macro_gap = (
        float(np.mean(model_dataset_gaps))
        if model_dataset_gaps else np.nan
    )

    model_summary_results.append({
        "model": model_name,
        "eligible_datasets": len(model_dataset_gaps),
        "macro_mean_dataset_gap": macro_gap,
        "pooled_same_emotion_mean": pooled_same_mean,
        "pooled_different_emotion_mean": pooled_different_mean,
        "pooled_emotion_gap": pooled_gap
    })

# ==========================================
# 8. Save Reports
# ==========================================

dataset_report = pd.DataFrame(dataset_results)

emotion_pair_report = pd.DataFrame(emotion_pair_results)

model_summary = pd.DataFrame(model_summary_results)

dataset_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "controlled_dataset_separability.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

emotion_pair_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "controlled_emotion_pair_similarity.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

model_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "controlled_model_comparison.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 9. Final Results
# ==========================================

print("\n" + "=" * 65)
print("CONTROLLED EMOTION SEPARABILITY COMPLETED")
print("=" * 65)

print("\nModel Comparison:")
print(model_summary.to_string(index=False))

print("\nDataset Results:")
print(dataset_report.to_string(index=False))

print("\nEligible datasets per model:")
print(
    dataset_report[
        dataset_report["evaluation_status"]
        == "Eligible for macro comparison"
    ].groupby("model").size()
)

print("\nResults saved in:", OUTPUT_DIR)
