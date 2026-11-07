
import os
import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.metrics import silhouette_samples

# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_results_farasa"

OUTPUT_DIR = os.path.join(
    FOLDER,
    "within_dataset_analysis"
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
    os.path.join(FOLDER, "pilot_metadata.csv")
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
# 3. Analysis Functions
# ==========================================

def calculate_pairwise_similarity(embeddings, labels, dataset_name, model_name):

    groups = {
        "Same Emotion": [],
        "Different Emotion": []
    }

    similarity_matrix = embeddings @ embeddings.T

    for i, j in combinations(range(len(labels)), 2):

        score = float(similarity_matrix[i, j])

        if labels[i] == labels[j]:
            groups["Same Emotion"].append(score)
        else:
            groups["Different Emotion"].append(score)

    same = groups["Same Emotion"]
    different = groups["Different Emotion"]

    same_mean = np.mean(same) if same else np.nan
    different_mean = np.mean(different) if different else np.nan

    gap = (
        same_mean - different_mean
        if same and different
        else np.nan
    )

    return {
        "model": model_name,
        "source_dataset": dataset_name,
        "same_emotion_pairs": len(same),
        "different_emotion_pairs": len(different),
        "same_emotion_mean": same_mean,
        "different_emotion_mean": different_mean,
        "emotion_separability_gap": gap
    }


# ==========================================
# 4. Process Each Model
# ==========================================

silhouette_results = []
emotion_results = []
similarity_results = []

for model_name, filename in MODELS.items():

    print("\n" + "=" * 60)
    print("Processing:", model_name)
    print("=" * 60)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    ).astype(np.float64)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Metadata and embeddings mismatch: {model_name}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values.")

    # Ensure unit-length vectors
    norms = np.linalg.norm(embeddings, axis=1)

    if np.any(norms < 1e-12):
        raise ValueError("Zero-length embedding detected.")

    embeddings = embeddings / norms[:, None]

    print("Embedding shape:", embeddings.shape)

    # ======================================
    # 5. Analyze Each Dataset Separately
    # ======================================

    for dataset_name, indices in metadata.groupby(
        "source_dataset",
        sort=True
    ).groups.items():

        indices = np.array(list(indices))

        dataset_embeddings = embeddings[indices]

        dataset_labels = (
            metadata.loc[indices, "emotion"]
            .astype(str)
            .to_numpy()
        )

        n = len(indices)
        emotion_counts = pd.Series(dataset_labels).value_counts()
        n_emotions = len(emotion_counts)

        print("\nDataset:", dataset_name)
        print("Number of texts:", n)
        print("Number of emotions:", n_emotions)

        # ----------------------------------
        # A. Pairwise Cosine Similarity
        # ----------------------------------

        similarity_result = calculate_pairwise_similarity(
            dataset_embeddings,
            dataset_labels,
            dataset_name,
            model_name
        )

        similarity_results.append(similarity_result)

        print(
            "Same emotion mean:",
            similarity_result["same_emotion_mean"]
        )

        print(
            "Different emotion mean:",
            similarity_result["different_emotion_mean"]
        )

        print(
            "Emotion separability gap:",
            similarity_result["emotion_separability_gap"]
        )

        # ----------------------------------
        # B. Silhouette Analysis
        # ----------------------------------

        if n_emotions < 2 or n_emotions >= n:

            status = "Skipped: insufficient emotion classes"

            silhouette_results.append({
                "model": model_name,
                "source_dataset": dataset_name,
                "number_of_texts": n,
                "number_of_emotions": n_emotions,
                "silhouette_score": np.nan,
                "evaluation_status": status
            })

            print(status)
            continue

        scores = silhouette_samples(
            dataset_embeddings,
            dataset_labels,
            metric="cosine"
        )

        overall_score = float(np.mean(scores))

        silhouette_results.append({
            "model": model_name,
            "source_dataset": dataset_name,
            "number_of_texts": n,
            "number_of_emotions": n_emotions,
            "silhouette_score": overall_score,
            "evaluation_status": "Evaluated"
        })

        print("Overall Silhouette:", round(overall_score, 4))

        # ----------------------------------
        # C. Silhouette Per Emotion
        # ----------------------------------

        for emotion in sorted(emotion_counts.index):

            mask = dataset_labels == emotion
            emotion_scores = scores[mask]
            count = int(mask.sum())

            emotion_results.append({
                "model": model_name,
                "source_dataset": dataset_name,
                "emotion": emotion,
                "number_of_texts": count,
                "mean_silhouette": float(np.mean(emotion_scores)),
                "median_silhouette": float(np.median(emotion_scores)),
                "std_silhouette": float(np.std(emotion_scores)),
                "evaluation_status": (
                    "Insufficient samples for reliable class evaluation"
                    if count < 2
                    else "Evaluated"
                )
            })

# ==========================================
# 6. Save Reports
# ==========================================

silhouette_report = pd.DataFrame(silhouette_results)

emotion_report = pd.DataFrame(emotion_results)

similarity_report = pd.DataFrame(similarity_results)

silhouette_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_silhouette_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

emotion_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_emotion_silhouette.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

similarity_report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_within_between_cosine.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 7. Final Verification
# ==========================================

print("\n" + "=" * 60)
print("WITHIN-DATASET EMOTION ANALYSIS COMPLETED")
print("=" * 60)

print("\nSilhouette Summary:")
print(silhouette_report.to_string(index=False))

print("\nCosine Similarity Summary:")
print(similarity_report.to_string(index=False))

print("\nPer-Emotion Results:")
print(emotion_report.to_string(index=False))

print("\nAll reports saved in:", OUTPUT_DIR)
