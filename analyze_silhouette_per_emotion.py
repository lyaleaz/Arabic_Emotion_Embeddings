
import os
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_samples

# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_results_farasa"

MODELS = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

OUTPUT_FILE = os.path.join(
    FOLDER,
    "silhouette_per_emotion.csv"
)

metadata = pd.read_csv(
    os.path.join(FOLDER, "pilot_metadata.csv")
)

labels = metadata["emotion"].astype(str).values

results = []

# ==========================================
# 2. Analyze Each Model
# ==========================================

for model_name, filename in MODELS.items():

    print("\n" + "=" * 55)
    print("Analyzing:", model_name)
    print("=" * 55)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    )

    if len(embeddings) != len(labels):
        raise ValueError("Metadata and embeddings mismatch.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values.")

    # Calculate silhouette score for every sentence
    sample_scores = silhouette_samples(
        embeddings,
        labels,
        metric="cosine"
    )

    model_results = []

    # ======================================
    # 3. Calculate Results Per Emotion
    # ======================================

    for emotion in sorted(np.unique(labels)):

        mask = labels == emotion

        emotion_scores = sample_scores[mask]

        count = int(mask.sum())

        row = {
            "model": model_name,
            "emotion": emotion,
            "number_of_texts": count,
            "mean_silhouette": np.mean(emotion_scores),
            "median_silhouette": np.median(emotion_scores),
            "std_silhouette": np.std(emotion_scores),
            "minimum_silhouette": np.min(emotion_scores),
            "maximum_silhouette": np.max(emotion_scores),
            "evaluation_status": (
                "Insufficient samples for reliable evaluation"
                if count < 2
                else "Evaluated"
            )
        }

        results.append(row)
        model_results.append(row)

        print("\nEmotion:", emotion)
        print("Number of texts:", count)
        print("Mean:", round(row["mean_silhouette"], 4))
        print("Median:", round(row["median_silhouette"], 4))
        print("STD:", round(row["std_silhouette"], 4))
        print("Status:", row["evaluation_status"])

# ==========================================
# 4. Save Results
# ==========================================

report = pd.DataFrame(results)

report.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 55)
print("PER-EMOTION SILHOUETTE ANALYSIS COMPLETED")
print("=" * 55)

print(report.to_string(index=False))

print("\nSaved:", OUTPUT_FILE)
