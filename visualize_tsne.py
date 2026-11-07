
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_results_farasa"

OUTPUT_DIR = os.path.join(
    FOLDER, "tSNE_visualizations"
)

MODELS = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

RANDOM_STATE = 42
PERPLEXITY = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Load Metadata
# ==========================================

metadata = pd.read_csv(
    os.path.join(FOLDER, "pilot_metadata.csv")
)

print("Number of records:", len(metadata))

# ==========================================
# 3. Plot Function
# ==========================================

def create_plot(coordinates, labels, title, filename):

    plt.figure(figsize=(14, 9))

    categories = sorted(labels.astype(str).unique())

    for category in categories:

        mask = labels.astype(str) == category

        plt.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=category,
            alpha=0.75,
            s=65
        )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


# ==========================================
# 4. Process Both Models
# ==========================================

for model_name, filename in MODELS.items():

    print("\n" + "=" * 50)
    print("Processing:", model_name)
    print("=" * 50)

    embeddings = np.load(
        os.path.join(FOLDER, filename)
    )

    print("Embedding shape:", embeddings.shape)

    if len(embeddings) != len(metadata):
        raise ValueError("Metadata and embeddings size mismatch.")

    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
        max_iter=1000
    )

    coordinates = tsne.fit_transform(embeddings)

    print("t-SNE completed.")
    print("Output shape:", coordinates.shape)

    # Save coordinates
    coordinates_df = pd.DataFrame({
        "tSNE_1": coordinates[:, 0],
        "tSNE_2": coordinates[:, 1]
    })

    coordinates_df = pd.concat(
        [metadata.reset_index(drop=True),
         coordinates_df],
        axis=1
    )

    coordinates_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{model_name}_tSNE_coordinates.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # By Emotion
    create_plot(
        coordinates,
        metadata["emotion"],
        f"{model_name} - t-SNE by Emotion",
        f"{model_name}_tSNE_by_emotion.png"
    )

    # By Dataset
    create_plot(
        coordinates,
        metadata["source_dataset"],
        f"{model_name} - t-SNE by Dataset",
        f"{model_name}_tSNE_by_dataset.png"
    )

print("\n" + "=" * 50)
print("t-SNE ANALYSIS COMPLETED")
print("=" * 50)

print("Results saved in:", OUTPUT_DIR)
