
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_results_farasa"

OUTPUT_DIR = os.path.join(
    FOLDER,
    "UMAP_visualizations"
)

MODELS = {
    "AraBERT_v2": "AraBERT_v2_pilot_embeddings.npy",
    "MARBERT": "MARBERT_pilot_embeddings.npy"
}

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. Load Metadata
# ==========================================

metadata = pd.read_csv(
    os.path.join(FOLDER, "pilot_metadata.csv")
)

print("Number of records:", len(metadata))

# ==========================================
# 3. UMAP Visualization
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
        raise ValueError("Embedding and metadata sizes do not match.")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE
    )

    reduced = reducer.fit_transform(embeddings)

    print("UMAP completed.")
    print("Output shape:", reduced.shape)

    # ======================================
    # 4. Plot by Emotion
    # ======================================

    plt.figure(figsize=(12, 8))

    for emotion in sorted(metadata["emotion"].unique()):

        mask = metadata["emotion"].values == emotion

        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=emotion,
            alpha=0.75,
            s=45
        )

    plt.title(f"{model_name} - UMAP by Emotion")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_UMAP_by_emotion.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)

    # ======================================
    # 5. Plot by Dataset
    # ======================================

    plt.figure(figsize=(12, 8))

    for dataset in sorted(metadata["source_dataset"].unique()):

        mask = metadata["source_dataset"].values == dataset

        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=dataset,
            alpha=0.75,
            s=45
        )

    plt.title(f"{model_name} - UMAP by Dataset")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_UMAP_by_dataset.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

    print("Saved:", output_file)

# ==========================================
# 6. Completion
# ==========================================

print("\n" + "=" * 50)
print("UMAP ANALYSIS COMPLETED")
print("=" * 50)

print("Results saved in:", OUTPUT_DIR)
