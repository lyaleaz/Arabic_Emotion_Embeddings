
import os
import numpy as np
import pandas as pd

from itertools import combinations
from collections import defaultdict


# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_results_farasa"
OUTPUT_DIR = os.path.join(FOLDER, "category_similarity")

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
)

metadata = metadata.reset_index(drop=True)

print("Number of records:", len(metadata))
print("Number of emotions:", metadata["emotion"].nunique())
print("Number of datasets:", metadata["source_dataset"].nunique())


# ==========================================
# 3. Analysis Function
# ==========================================

def analyze_categories(embeddings, column, model_name):

    groups = defaultdict(list)

    similarity_matrix = embeddings @ embeddings.T

    for i, j in combinations(range(len(metadata)), 2):

        category_a = str(metadata.loc[i, column])
        category_b = str(metadata.loc[j, column])

        if column == "emotion":
            group_type = (
                "In-category"
                if category_a == category_b
                else "Cross-category"
            )
        else:
            group_type = (
                "In-category"
                if category_a == category_b
                else "Cross-category"
            )

        categories = tuple(sorted([category_a, category_b]))

        key = (group_type, categories)

        groups[key].append(float(similarity_matrix[i, j]))

    rows = []

    for (group_type, categories), values in groups.items():

        rows.append({
            "model": model_name,
            "category_type": column,
            "comparison_type": group_type,
            "category_1": categories[0],
            "category_2": categories[1],
            "number_of_pairs": len(values),
            "mean_similarity": np.mean(values),
            "median_similarity": np.median(values),
            "std_similarity": np.std(values)
        })

    return pd.DataFrame(rows)


# ==========================================
# 4. Process Both Models
# ==========================================

all_results = []

for model_name, filename in MODELS.items():

    print("\n" + "=" * 55)
    print("Analyzing:", model_name)
    print("=" * 55)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Metadata and embeddings mismatch for {model_name}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    norms = np.linalg.norm(embeddings, axis=1)

    print("Embedding shape:", embeddings.shape)
    print("Minimum norm:", norms.min())
    print("Maximum norm:", norms.max())

    # Emotion analysis
    emotion_results = analyze_categories(
        embeddings,
        "emotion",
        model_name
    )

    # Dataset analysis
    dataset_results = analyze_categories(
        embeddings,
        "source_dataset",
        model_name
    )

    model_results = pd.concat(
        [emotion_results, dataset_results],
        ignore_index=True
    )

    all_results.append(model_results)

    print("\nEmotion comparisons:")
    print(
        emotion_results.to_string(index=False)
    )

    print("\nDataset comparisons:")
    print(
        dataset_results.to_string(index=False)
    )


# ==========================================
# 5. Save Results
# ==========================================

final_results = pd.concat(
    all_results,
    ignore_index=True
)

emotion_output = os.path.join(
    OUTPUT_DIR,
    "emotion_category_similarity.csv"
)

dataset_output = os.path.join(
    OUTPUT_DIR,
    "dataset_category_similarity.csv"
)

final_results[
    final_results["category_type"] == "emotion"
].to_csv(
    emotion_output,
    index=False,
    encoding="utf-8-sig"
)

final_results[
    final_results["category_type"] == "source_dataset"
].to_csv(
    dataset_output,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 6. Final Verification
# ==========================================

print("\n" + "=" * 55)
print("CATEGORY SIMILARITY ANALYSIS COMPLETED")
print("=" * 55)

print("Total comparisons:", len(final_results))
print("Emotion report:", emotion_output)
print("Dataset report:", dataset_output)

print("\nAll results saved successfully!")
