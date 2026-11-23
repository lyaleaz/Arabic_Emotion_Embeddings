
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
    "controlled_emotion_separability_v2"
)

MODELS = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

MIN_DATASET_SIZE = 20
MIN_SAMPLES_PER_EMOTION = 5

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
    raise ValueError("Missing dataset or emotion labels.")

print("Total records:", len(metadata))
print("Number of datasets:", metadata["source_dataset"].nunique())
print("Number of emotions:", metadata["emotion"].nunique())

# ==========================================
# 3. Result Containers
# ==========================================

dataset_results = []
emotion_results = []
emotion_pair_results = []
model_summary_results = []

# ==========================================
# 4. Process Both Models
# ==========================================

for model_name, filename in MODELS.items():

    print("\n" + "=" * 65)
    print("Analyzing:", model_name)
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

    dataset_gaps = []

    # ======================================
    # 5. Analyze Each Dataset
    # ======================================

    for dataset_name, indices in metadata.groupby(
        "source_dataset",
        sort=True
    ).groups.items():

        indices = np.array(list(indices))

        dataset_labels = (
            metadata.loc[indices, "emotion"]
            .astype(str)
            .to_numpy()
        )

        dataset_size = len(indices)

        emotion_counts = pd.Series(
            dataset_labels
        ).value_counts()

        # Keep only emotions with at least 5 texts
        eligible_emotions = sorted(
            emotion_counts[
                emotion_counts >= MIN_SAMPLES_PER_EMOTION
            ].index.tolist()
        )

        eligible_mask = np.isin(
            dataset_labels,
            eligible_emotions
        )

        # Use exactly the same eligible texts
        # for both intra and inter comparisons

        selected_indices = indices[eligible_mask]

        selected_labels = dataset_labels[eligible_mask]

        number_of_selected_texts = len(selected_indices)

        number_of_eligible_emotions = len(
            eligible_emotions
        )

        status = "Eligible"

        if (
            dataset_size < MIN_DATASET_SIZE
            or number_of_eligible_emotions < 2
        ):

            status = "Insufficient representation"

            dataset_results.append({
                "model": model_name,
                "source_dataset": dataset_name,
                "original_texts": dataset_size,
                "selected_texts": number_of_selected_texts,
                "eligible_emotions": number_of_eligible_emotions,
                "eligible_emotion_names": "; ".join(
                    eligible_emotions
                ),
                "intra_emotion_mean": np.nan,
                "inter_emotion_mean": np.nan,
                "emotion_separability_gap": np.nan,
                "evaluation_status": status
            })

            continue

        selected_embeddings = embeddings[selected_indices]

        similarity_matrix = (
            selected_embeddings @ selected_embeddings.T
        )

        upper_i, upper_j = np.triu_indices(
            number_of_selected_texts,
            k=1
        )

        similarities = similarity_matrix[upper_i, upper_j]

        labels_i = selected_labels[upper_i]
        labels_j = selected_labels[upper_j]

        # ==================================
        # 6. Intra-emotion Similarity
        # ==================================

        intra_emotion_means = []

        for emotion in eligible_emotions:

            mask = (
                (labels_i == emotion)
                &
                (labels_j == emotion)
            )

            values = similarities[mask]

            if len(values) == 0:
                continue

            mean_value = float(np.mean(values))

            intra_emotion_means.append(mean_value)

            emotion_results.append({
                "model": model_name,
                "source_dataset": dataset_name,
                "emotion": emotion,
                "number_of_texts": int(
                    emotion_counts[emotion]
                ),
                "number_of_pairs": len(values),
                "mean_similarity": mean_value
            })

        # Equal weight for each eligible emotion
        intra_mean = float(
            np.mean(intra_emotion_means)
        )

        # ==================================
        # 7. Inter-emotion Similarity
        # ==================================

        inter_emotion_means = []

        for emotion_1, emotion_2 in combinations(
            eligible_emotions,
            2
        ):

            mask = (
                ((labels_i == emotion_1) &
                 (labels_j == emotion_2))
                |
                ((labels_i == emotion_2) &
                 (labels_j == emotion_1))
            )

            values = similarities[mask]

            if len(values) == 0:
                continue

            mean_value = float(np.mean(values))

            inter_emotion_means.append(mean_value)

            emotion_pair_results.append({
                "model": model_name,
                "source_dataset": dataset_name,
                "emotion_1": emotion_1,
                "emotion_2": emotion_2,
                "number_of_pairs": len(values),
                "mean_similarity": mean_value
            })

        # Equal weight for each emotion pair
        inter_mean = float(
            np.mean(inter_emotion_means)
        )

        # ==================================
        # 8. Emotion Separability Gap
        # ==================================

        gap = intra_mean - inter_mean

        dataset_gaps.append(gap)

        dataset_results.append({
            "model": model_name,
            "source_dataset": dataset_name,
            "original_texts": dataset_size,
            "selected_texts": number_of_selected_texts,
            "eligible_emotions": number_of_eligible_emotions,
            "eligible_emotion_names": "; ".join(
                eligible_emotions
            ),
            "intra_emotion_mean": intra_mean,
            "inter_emotion_mean": inter_mean,
            "emotion_separability_gap": gap,
            "evaluation_status": status
        })

        print("\nDataset:", dataset_name)
        print("Original texts:", dataset_size)
        print("Selected texts:", number_of_selected_texts)
        print("Eligible emotions:", eligible_emotions)
        print("Intra-emotion mean:", round(intra_mean, 6))
        print("Inter-emotion mean:", round(inter_mean, 6))
        print("Separability gap:", round(gap, 6))

    # ======================================
    # 9. Model-Level Summary
    # ======================================

    macro_gap = (
        float(np.mean(dataset_gaps))
        if dataset_gaps else np.nan
    )

    model_summary_results.append({
        "model": model_name,
        "eligible_datasets": len(dataset_gaps),
        "macro_mean_dataset_gap": macro_gap
    })

# ==========================================
# 10. Save Reports
# ==========================================

dataset_report = pd.DataFrame(dataset_results)

emotion_report = pd.DataFrame(emotion_results)

emotion_pair_report = pd.DataFrame(
    emotion_pair_results
)

model_summary = pd.DataFrame(
    model_summary_results
)

dataset_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "controlled_dataset_separability_v2.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

emotion_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "controlled_emotion_similarity_v2.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

emotion_pair_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "controlled_emotion_pair_similarity_v2.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

model_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "controlled_model_comparison_v2.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 11. Final Verification
# ==========================================

print("\n" + "=" * 65)
print("CONTROLLED ANALYSIS V2 COMPLETED")
print("=" * 65)

print("\nModel Comparison:")
print(model_summary.to_string(index=False))

print("\nDataset Results:")
print(dataset_report.to_string(index=False))

print("\nEligible datasets:")
print(
    dataset_report[
        dataset_report["evaluation_status"] == "Eligible"
    ].groupby("model").size()
)

print("\nAll results saved in:", OUTPUT_DIR)
