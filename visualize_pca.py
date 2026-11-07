
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_results_farasa"

OUTPUT_DIR = os.path.join(
    FOLDER,
    "PCA_visualizations"
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
)

metadata = metadata.reset_index(drop=True)

print("Number of records:", len(metadata))


# ==========================================
# 3. PCA Visualization Function
# ==========================================

def create_pca_plot(coordinates, labels, model_name,
                    category_type, explained_variance):

    plt.figure(figsize=(12, 8))

    unique_labels = sorted(
        pd.Series(labels).dropna().unique()
    )

    for label in unique_labels:

        mask = np.array(labels) == label

        plt.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=str(label),
            alpha=0.75,
            s=55
        )

    plt.xlabel(
        f"PC1 ({explained_variance[0] * 100:.2f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained_variance[1] * 100:.2f}% variance)"
    )

    plt.title(
        f"{model_name} - PCA by {category_type}"
    )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8
    )

    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_PCA_by_{category_type}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file)


# ==========================================
# 4. Process Both Models
# ==========================================

summary_results = []

for model_name, filename in MODELS.items():

    print("\n" + "=" * 55)
    print("Processing:", model_name)
    print("=" * 55)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embedding and metadata row counts do not match."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values.")

    # PCA
    pca = PCA(n_components=2)

    coordinates = pca.fit_transform(embeddings)

    explained = pca.explained_variance_ratio_

    print("Original dimensions:", embeddings.shape[1])
    print("PCA dimensions:", coordinates.shape[1])

    print(
        "PC1 explained variance:",
        round(explained[0] * 100, 2), "%"
    )

    print(
        "PC2 explained variance:",
        round(explained[1] * 100, 2), "%"
    )

    print(
        "Total explained variance:",
        round(explained.sum() * 100, 2), "%"
    )

    # Save PCA coordinates
    coordinates_df = metadata.copy()

    coordinates_df["PC1"] = coordinates[:, 0]
    coordinates_df["PC2"] = coordinates[:, 1]

    coordinates_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_PCA_coordinates.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # Visualization by emotion
    create_pca_plot(
        coordinates,
        metadata["emotion"].values,
        model_name,
        "emotion",
        explained
    )

    # Visualization by dataset
    create_pca_plot(
        coordinates,
        metadata["source_dataset"].values,
        model_name,
        "dataset",
        explained
    )

    summary_results.append({
        "model": model_name,
        "PC1_variance": explained[0],
        "PC2_variance": explained[1],
        "total_explained_variance": explained.sum()
    })


# ==========================================
# 5. Save Comparison Report
# ==========================================

report = pd.DataFrame(summary_results)

report.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "PCA_comparison_report.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 55)
print("PCA ANALYSIS COMPLETED")
print("=" * 55)

print(report.to_string(index=False))

print("\nAll PCA results saved in:", OUTPUT_DIR)
