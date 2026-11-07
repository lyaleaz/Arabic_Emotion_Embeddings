
import os
import numpy as np
import pandas as pd

# ==========================================
# 1. Configuration
# ==========================================

INPUT_FILE = "Arabic_Emotion_Plutchik8_datasets.xlsx"

INPUT_SHEET = "Unified_Data"

OUTPUT_DIR = "pilot_1000_results_farasa"

SAMPLE_SIZE = 1000

RANDOM_STATE = 42

GROUP_COLUMNS = [
    "source_dataset",
    "emotion"
]

REQUIRED_COLUMNS = [
    "source_dataset",
    "original_id",
    "text",
    "emotion"
]

# ==========================================
# 2. Load Dataset
# ==========================================

print("=" * 60)
print("LOADING ORIGINAL DATASET")
print("=" * 60)

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=INPUT_SHEET
)

print("Original records:", len(df))

# Verify required columns
for column in REQUIRED_COLUMNS:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

# Verify missing values
if df[REQUIRED_COLUMNS].isna().any().any():
    raise ValueError(
        "Missing values found in required columns."
    )

if len(df) < SAMPLE_SIZE:
    raise ValueError(
        "Dataset contains fewer than 1000 records."
    )

print("Number of datasets:",
      df["source_dataset"].nunique())

print("Number of emotions:",
      df["emotion"].nunique())

# ==========================================
# 3. Analyze Original Group Distribution
# ==========================================

print("\n" + "=" * 60)
print("ORIGINAL DATASET DISTRIBUTION")
print("=" * 60)

group_counts = (
    df.groupby(GROUP_COLUMNS, sort=True)
      .size()
      .reset_index(name="original_count")
)

print("\nNumber of dataset-emotion groups:",
      len(group_counts))

print("\nOriginal distribution:")
print(group_counts.to_string(index=False))

if len(group_counts) > SAMPLE_SIZE:
    raise ValueError(
        "There are more than 1000 non-empty groups. "
        "At least one sample per group is impossible."
    )

# ==========================================
# 4. Calculate Sample Allocation
# ==========================================

print("\n" + "=" * 60)
print("CALCULATING STRATIFIED SAMPLE ALLOCATION")
print("=" * 60)

group_counts["sample_count"] = 1

remaining = SAMPLE_SIZE - len(group_counts)

# Capacity after guaranteeing one sample per group
capacity = (
    group_counts["original_count"].to_numpy()
    - group_counts["sample_count"].to_numpy()
)

# Distribute remaining samples proportionally
while remaining > 0:

    available = capacity > 0

    if not available.any():
        raise ValueError(
            "Cannot allocate the requested sample size."
        )

    weights = np.where(available, capacity, 0)

    quotas = remaining * weights / weights.sum()

    additions = np.floor(quotas).astype(int)

    additions = np.minimum(additions, capacity)

    group_counts["sample_count"] += additions

    capacity -= additions

    remaining -= int(additions.sum())

    if remaining == 0:
        break

    # Distribute remaining samples by largest fractional remainder
    fractional = quotas - np.floor(quotas)

    candidates = np.where(capacity > 0)[0]

    if len(candidates) == 0:
        continue

    order = candidates[
        np.argsort(-fractional[candidates], kind="stable")
    ]

    for idx in order:

        if remaining == 0:
            break

        group_counts.loc[idx, "sample_count"] += 1
        capacity[idx] -= 1
        remaining -= 1

# Verify allocation
if group_counts["sample_count"].sum() != SAMPLE_SIZE:
    raise ValueError("Sample allocation does not equal 1000.")

if (
    group_counts["sample_count"]
    > group_counts["original_count"]
).any():
    raise ValueError("Allocation exceeds original group size.")

print("\nSample allocation:")
print(group_counts.to_string(index=False))

print("\nTotal allocated samples:",
      group_counts["sample_count"].sum())

# ==========================================
# 5. Select Pilot Records
# ==========================================

print("\n" + "=" * 60)
print("SELECTING PILOT SAMPLE")
print("=" * 60)

rng = np.random.default_rng(RANDOM_STATE)

selected_indices = []

for _, group in group_counts.iterrows():

    dataset_name = group["source_dataset"]
    emotion_name = group["emotion"]
    number_to_select = int(group["sample_count"])

    group_df = df[
        (df["source_dataset"] == dataset_name)
        & (df["emotion"] == emotion_name)
    ]

    selected = rng.choice(
        group_df.index.to_numpy(),
        size=number_to_select,
        replace=False
    )

    selected_indices.extend(selected.tolist())

# Create final pilot sample
pilot_df = (
    df.loc[selected_indices]
      .sample(frac=1, random_state=RANDOM_STATE)
      .reset_index(drop=True)
)

# ==========================================
# 6. Verify Pilot Sample
# ==========================================

print("\n" + "=" * 60)
print("VERIFYING PILOT SAMPLE")
print("=" * 60)

if len(pilot_df) != SAMPLE_SIZE:
    raise ValueError("Pilot sample size is incorrect.")

if pilot_df[REQUIRED_COLUMNS].isna().any().any():
    raise ValueError("Pilot contains missing required values.")

print("Pilot records:", len(pilot_df))

print("\nEmotion distribution:")
print(pilot_df["emotion"].value_counts())

print("\nDataset distribution:")
print(pilot_df["source_dataset"].value_counts())

# Verify each original group is represented
selected_counts = (
    pilot_df.groupby(GROUP_COLUMNS, sort=True)
            .size()
            .reset_index(name="selected_count")
)

verification = group_counts.merge(
    selected_counts,
    on=GROUP_COLUMNS,
    how="left"
)

verification["selected_count"] = (
    verification["selected_count"].fillna(0).astype(int)
)

if not (
    verification["sample_count"]
    == verification["selected_count"]
).all():
    raise ValueError("Group allocation verification failed.")

# ==========================================
# 7. Create Output Directory
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# 8. Save Pilot Metadata
# ==========================================

metadata_file = os.path.join(
    OUTPUT_DIR,
    "pilot_metadata.csv"
)

pilot_df.to_csv(
    metadata_file,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 9. Save Sampling Distribution
# ==========================================

distribution_file = os.path.join(
    OUTPUT_DIR,
    "pilot_sampling_distribution.csv"
)

verification.to_csv(
    distribution_file,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 10. Save Emotion Distribution
# ==========================================

emotion_distribution = (
    pilot_df["emotion"]
    .value_counts()
    .rename_axis("emotion")
    .reset_index(name="sample_count")
)

emotion_distribution.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pilot_emotion_distribution.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ==========================================
# 11. Final Summary
# ==========================================

print("\n" + "=" * 60)
print("PILOT SAMPLING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Original dataset records:", len(df))
print("Pilot sample records:", len(pilot_df))
print("Dataset-emotion groups:", len(group_counts))
print("Random state:", RANDOM_STATE)

print("\nSaved files:")
print(metadata_file)
print(distribution_file)
print(
    os.path.join(
        OUTPUT_DIR,
        "pilot_emotion_distribution.csv"
    )
)

print("\nAll original records remain unchanged.")
print("No original records were deleted.")
print("Sampling completed successfully!")
