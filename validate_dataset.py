
import pandas as pd

file_path = "Arabic_Emotion_Plutchik8_datasets.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="Unified_Data"
)

expected_emotions = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust"
]

print("=" * 50)
print("DATASET VALIDATION")
print("=" * 50)

print("\nTotal records:", len(df))

print("\nEmotion distribution:")
print(df["emotion"].value_counts(dropna=False))

print("\nMissing values:")
print(df[["text", "emotion", "source_dataset"]].isna().sum())

print("\nEmpty texts:",
      df["text"].fillna("").astype(str).str.strip().eq("").sum())

print("\nNumber of datasets:",
      df["source_dataset"].nunique())

print("\nUnique emotions:")
print(sorted(df["emotion"].dropna().unique()))

print("\nMissing expected emotions:")
print(set(expected_emotions) - set(df["emotion"].dropna().unique()))

print("\nDuplicate texts:",
      df["text"].duplicated().sum())

print("\nDuplicate source_dataset + original_id:")
print(df.duplicated(
    subset=["source_dataset", "original_id"]
).sum())

print("\nValidation completed.")
