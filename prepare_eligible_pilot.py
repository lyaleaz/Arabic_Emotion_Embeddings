
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. Configuration
# ==========================================

INPUT_FOLDER = "pilot_1000_results_farasa"

OUTPUT_FOLDER = "pilot_984_eligible_farasa"

MIN_DATASET_SIZE = 20

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. Load Metadata
# ==========================================

metadata = pd.read_csv(
    os.path.join(INPUT_FOLDER, "pilot_metadata.csv")
)

print("Original pilot size:", len(metadata))

# ==========================================
# 3. Determine Eligible Datasets
# ==========================================

full_df = pd.read_excel(
    "Arabic_Emotion_Plutchik8_datasets.xlsx",
    sheet_name="Unified_Data"
)

eligible_counts = (
    full_df[
        full_df["emotion"].isin([
            "anger", "anticipation", "disgust",
            "fear", "joy", "sadness",
            "surprise", "trust"
        ])
    ]
    .groupby("source_dataset")
    .size()
)

eligible_datasets = set(
    eligible_counts[
        eligible_counts >= MIN_DATASET_SIZE
    ].index
)

# ==========================================
# 4. Create Consistent Filter
# ==========================================

mask = metadata["source_dataset"].isin(
    eligible_datasets
)

filtered_metadata = (
    metadata.loc[mask]
    .reset_index(drop=True)
)

print("Eligible pilot records:", len(filtered_metadata))

print("\nDataset distribution:")
print(
    filtered_metadata["source_dataset"]
    .value_counts()
    .to_string()
)

print("\nEmotion distribution:")
print(
    filtered_metadata["emotion"]
    .value_counts()
    .to_string()
)

# ==========================================
# 5. Filter Both Models
# ==========================================

for model_name in ["AraBERT_v2", "MARBERT"]:

    input_file = os.path.join(
        INPUT_FOLDER,
        f"{model_name}_pilot_embeddings.npy"
    )

    embeddings = np.load(input_file)

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Metadata mismatch: {model_name}"
        )

    filtered_embeddings = embeddings[mask.to_numpy()]

    if len(filtered_embeddings) != len(filtered_metadata):
        raise ValueError("Filtered alignment error.")

    if not np.isfinite(filtered_embeddings).all():
        raise ValueError("Invalid embedding values.")

    output_file = os.path.join(
        OUTPUT_FOLDER,
        f"{model_name}_pilot_embeddings.npy"
    )

    np.save(output_file, filtered_embeddings)

    print("\n", model_name)
    print("New shape:", filtered_embeddings.shape)

# ==========================================
# 6. Save Metadata
# ==========================================

filtered_metadata.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "pilot_metadata.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 7. Final Verification
# ==========================================

print("\n" + "=" * 55)
print("ELIGIBLE PILOT PREPARATION COMPLETED")
print("=" * 55)

print("Original records:", len(metadata))
print("Filtered records:", len(filtered_metadata))
print("Excluded records:", len(metadata) - len(filtered_metadata))
print("Eligible datasets:", filtered_metadata["source_dataset"].nunique())

for model_name in ["AraBERT_v2", "MARBERT"]:

    embeddings = np.load(
        os.path.join(
            OUTPUT_FOLDER,
            f"{model_name}_pilot_embeddings.npy"
        )
    )

    print(
        model_name,
        "Shape:", embeddings.shape,
        "Finite:", np.isfinite(embeddings).all()
    )

print("\nAll files saved successfully!")
print("Output folder:", OUTPUT_FOLDER)
