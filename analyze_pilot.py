
import numpy as np
import pandas as pd
import os

from itertools import combinations

folder = "pilot_results"

metadata = pd.read_csv(
    os.path.join(folder, "pilot_metadata.csv")
)

models = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

results = []

for model_name, filename in models.items():

    print("\n" + "=" * 50)
    print("Analyzing:", model_name)
    print("=" * 50)

    embeddings = np.load(
        os.path.join(folder, filename)
    )

    print("Shape:", embeddings.shape)
    print("All values finite:", np.isfinite(embeddings).all())

    norms = np.linalg.norm(embeddings, axis=1)

    print("Minimum vector norm:", norms.min())
    print("Maximum vector norm:", norms.max())

    # Embeddings are already L2-normalized
    similarity_matrix = embeddings @ embeddings.T

    same_emotion = []
    different_emotion = []

    same_dataset = []
    different_dataset = []

    for i, j in combinations(range(len(metadata)), 2):

        score = float(similarity_matrix[i, j])

        if metadata.loc[i, "emotion"] == metadata.loc[j, "emotion"]:
            same_emotion.append(score)
        else:
            different_emotion.append(score)

        if metadata.loc[i, "source_dataset"] == metadata.loc[j, "source_dataset"]:
            same_dataset.append(score)
        else:
            different_dataset.append(score)

    groups = {
        "Same Emotion": same_emotion,
        "Different Emotion": different_emotion,
        "Same Dataset": same_dataset,
        "Different Dataset": different_dataset
    }

    for group_name, values in groups.items():

        if len(values) == 0:
            continue

        row = {
            "model": model_name,
            "comparison": group_name,
            "pairs": len(values),
            "mean_similarity": np.mean(values),
            "median_similarity": np.median(values),
            "std_similarity": np.std(values)
        }

        results.append(row)

        print("\n", group_name)
        print("Number of pairs:", len(values))
        print("Mean:", round(np.mean(values), 4))
        print("Median:", round(np.median(values), 4))
        print("Standard deviation:", round(np.std(values), 4))


report = pd.DataFrame(results)

output = os.path.join(
    folder,
    "pilot_similarity_report.csv"
)

report.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 50)
print("PILOT ANALYSIS COMPLETED")
print("=" * 50)

print(report.to_string(index=False))
print("\nSaved:", output)
