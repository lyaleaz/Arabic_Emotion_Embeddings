
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. Configuration
# ==========================================

INPUT_FILE = "Arabic_Emotion_Plutchik8_datasets.xlsx"
PILOT_FOLDER = "pilot_1000_results_farasa"

MIN_DATASET_SIZE = 20

PLUTCHIK_8 = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust"
]

# ==========================================
# 2. Load Original Dataset
# ==========================================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Unified_Data"
)

print("Original records:", len(df))

# Filter Plutchik 8 first
filtered_df = df[
    df["emotion"].isin(PLUTCHIK_8)
].copy()

print("Records after Plutchik 8 filtering:", len(filtered_df))

# ==========================================
# 3. Correct Eligibility Criterion
# ==========================================

dataset_counts = (
    filtered_df.groupby("source_dataset")
    .size()
    .sort_values(ascending=False)
)

eligible_datasets = dataset_counts[
    dataset_counts >= MIN_DATASET_SIZE
]

ineligible_datasets = dataset_counts[
    dataset_counts < MIN_DATASET_SIZE
]

print("\n" + "=" * 55)
print("CORRECTED DATASET ELIGIBILITY")
print("=" * 55)

print("\nEligible datasets:")
print(eligible_datasets.to_string())

print("\nIneligible datasets:")
print(ineligible_datasets.to_string())

print("\nNumber of eligible datasets:", len(eligible_datasets))
print("Number of ineligible datasets:", len(ineligible_datasets))

# ==========================================
# 4. Load Existing Pilot Metadata
# ==========================================

metadata_file = os.path.join(
    PILOT_FOLDER,
    "pilot_metadata.csv"
)

if not os.path.exists(metadata_file):
    raise FileNotFoundError(
        f"Pilot metadata not found: {metadata_file}"
    )

metadata = pd.read_csv(metadata_file)

print("\nExisting pilot records:", len(metadata))

# ==========================================
# 5. Compare Pilot with Eligibility
# ==========================================

eligible_names = set(eligible_datasets.index)

pilot_names = set(metadata["source_dataset"].dropna())

pilot_ineligible = pilot_names - eligible_names

pilot_eligible = pilot_names & eligible_names

print("\n" + "=" * 55)
print("PILOT ELIGIBILITY AUDIT")
print("=" * 55)

print("\nEligible datasets represented in pilot:")
print(sorted(pilot_eligible))

print("\nIneligible datasets represented in pilot:")
print(sorted(pilot_ineligible))

print("\nNumber of eligible datasets in pilot:", len(pilot_eligible))
print("Number of ineligible datasets in pilot:", len(pilot_ineligible))

# ==========================================
# 6. Verify Existing Embeddings
# ==========================================

for model in ["AraBERT_v2", "MARBERT"]:

    filename = os.path.join(
        PILOT_FOLDER,
        f"{model}_pilot_embeddings.npy"
    )

    if not os.path.exists(filename):
        print(f"\nMissing embeddings: {model}")
        continue

    embeddings = np.load(filename)

    print(f"\n{model}:")
    print("Shape:", embeddings.shape)
    print("Finite:", np.isfinite(embeddings).all())

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Metadata mismatch for {model}"
        )

# ==========================================
# 7. Save Audit Report
# ==========================================

audit = pd.DataFrame({
    "source_dataset": dataset_counts.index,
    "filtered_record_count": dataset_counts.values,
    "eligible": dataset_counts.values >= MIN_DATASET_SIZE
})

output_file = os.path.join(
    PILOT_FOLDER,
    "dataset_eligibility_audit.csv"
)

audit.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 55)
print("ELIGIBILITY AUDIT COMPLETED")
print("=" * 55)

print("Audit report saved:", output_file)
