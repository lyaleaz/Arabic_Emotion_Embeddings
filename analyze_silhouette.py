
import os
import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score

# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_results_farasa"

MODELS = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

metadata = pd.read_csv(
    os.path.join(FOLDER, "pilot_metadata.csv")
)

labels = metadata["emotion"].values

results = []

# ==========================================
# 2. Analyze Each Model
# ==========================================

for model_name, filename in MODELS.items():

    print("\n" + "=" * 50)
    print("Analyzing:", model_name)
    print("=" * 50)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    )

    if len(embeddings) != len(labels):
        raise ValueError("Metadata and embeddings mismatch.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values.")

    unique_labels = np.unique(labels)

    print("Number of records:", len(labels))
    print("Number of emotion classes:", len(unique_labels))

    # Cosine distance = 1 - cosine similarity
    score = silhouette_score(
        embeddings,
        labels,
        metric="cosine"
    )

    results.append({
        "model": model_name,
        "records": len(labels),
        "emotion_classes": len(unique_labels),
        "silhouette_score": score
    })

    print("Silhouette Score:", round(score, 6))

# ==========================================
# 3. Save Results
# ==========================================

report = pd.DataFrame(results)

output = os.path.join(
    FOLDER,
    "silhouette_comparison.csv"
)

report.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 50)
print("SILHOUETTE ANALYSIS COMPLETED")
print("=" * 50)

print(report.to_string(index=False))
print("\nSaved:", output)
