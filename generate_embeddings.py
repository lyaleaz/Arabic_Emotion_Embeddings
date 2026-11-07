"""
Generate sentence embeddings for the FULL Arabic Plutchik-8 dataset.

This script follows the preprocessing and embedding approach used in
generate_embeddings_pilot.py, but processes every row in Unified_Data.

Outputs are written to:
    full_results_farasa/
        full_metadata.csv
        AraBERT_v2_full_embeddings.npy
        MARBERT_full_embeddings.npy
        embedding_generation_summary.txt

Important:
- Row order is preserved. Embedding row i corresponds to metadata row i.
- AraBERT v2: local Farasa segmentation + AraBERT preprocessing.
- MARBERT: original Arabic text + MARBERT tokenizer.
- Both models use masked mean pooling followed by L2 normalization.
- max_length=128 is retained from the pilot for methodological consistency.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor


# ============================================================
# 1. Configuration
# ============================================================

INPUT_FILE = "Arabic_Emotion_Plutchik8_datasets.xlsx"
SHEET_NAME = "Unified_Data"

OUTPUT_DIR = Path("full_results_farasa")

BATCH_SIZE = 16
MAX_LENGTH = 128

FARASA_JAR = Path(r"C:\Farasa\Farasa.jar")
FARASA_DATA = Path(r"C:\Farasa\FarasaData")
FARASA_LIB = Path(r"C:\Farasa\lib")

MODELS = {
    "AraBERT_v2": "aubmindlab/bert-base-arabertv2",
    "MARBERT": "UBC-NLP/MARBERT",
}

REQUIRED_COLUMNS = [
    "source_dataset",
    "original_id",
    "text",
    "emotion",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. Check environment and input
# ============================================================

print("Using device:", DEVICE)
print("Input file:", INPUT_FILE)
print("Output directory:", OUTPUT_DIR)

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(
        f"Input Excel file was not found: {Path(INPUT_FILE).resolve()}"
    )

for required_path in [FARASA_JAR, FARASA_DATA]:
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required Farasa path was not found: {required_path}"
        )

for required_jar in [
    FARASA_LIB / "commons-lang3-3.1.jar",
    FARASA_LIB / "mapdb-2.0.0-SNAPSHOT.jar",
]:
    if not required_jar.exists():
        raise FileNotFoundError(
            f"Required Farasa library was not found: {required_jar}"
        )

os.environ["FarasaDataDir"] = str(FARASA_DATA.resolve()) + os.sep

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Load the complete dataset
# ============================================================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name=SHEET_NAME,
)

missing_columns = [
    column for column in REQUIRED_COLUMNS if column not in df.columns
]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df.empty:
    raise ValueError("The input dataset is empty.")

if df[REQUIRED_COLUMNS].isna().any().any():
    missing_counts = df[REQUIRED_COLUMNS].isna().sum()
    raise ValueError(
        "Missing values were found in required columns:\n"
        + missing_counts[missing_counts > 0].to_string()
    )

# Preserve original row order and make the alignment explicit.
df = df.reset_index(drop=True).copy()
df.insert(0, "embedding_row", np.arange(len(df), dtype=np.int64))

original_texts = (
    df["text"]
    .astype(str)
    .str.replace("\r", " ", regex=False)
    .str.replace("\n", " ", regex=False)
    .str.strip()
    .tolist()
)

if any(not text for text in original_texts):
    raise ValueError("At least one text is empty after whitespace cleanup.")

print("\nFull dataset records:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts().to_string())
print("\nDataset source distribution:")
print(df["source_dataset"].value_counts().to_string())


# ============================================================
# 4. Local Farasa segmentation
# ============================================================

def farasa_segment_texts(texts):
    """
    Segment all texts in one Farasa process.

    Each source record is written as one input line. Newlines inside a
    record are replaced before writing, and the output row count is checked.
    """

    print("\nRunning local Farasa segmentation...")
    print("Input records:", len(texts))

    with tempfile.TemporaryDirectory(prefix="farasa_full_") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "farasa_input.txt"
        output_path = temp_dir / "farasa_output.txt"

        input_path.write_text(
            "\n".join(texts) + "\n",
            encoding="utf-8",
        )

        classpath = ";".join(
            [
                str(FARASA_JAR.resolve()),
                str((FARASA_LIB / "commons-lang3-3.1.jar").resolve()),
                str((FARASA_LIB / "mapdb-2.0.0-SNAPSHOT.jar").resolve()),
            ]
        )

        command = [
            "java",
            "-Dfile.encoding=UTF-8",
            "-cp",
            classpath,
            "farasa.TestCase",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.stdout.strip():
                print(result.stdout.strip())

        except subprocess.CalledProcessError as error:
            print("\nFARASA ERROR")
            print("Return code:", error.returncode)
            print("STDOUT:", error.stdout)
            print("STDERR:", error.stderr)
            raise

        if not output_path.exists():
            raise RuntimeError("Farasa did not create its output file.")

        segmented = output_path.read_text(
            encoding="utf-8"
        ).splitlines()

    if len(segmented) != len(texts):
        raise ValueError(
            "Farasa changed the number of records: "
            f"input={len(texts)}, output={len(segmented)}. "
            "Stop here; do not generate embeddings with misaligned rows."
        )

    empty_rows = [
        index for index, text in enumerate(segmented) if not text.strip()
    ]
    if empty_rows:
        raise ValueError(
            f"Farasa produced empty output for rows: {empty_rows[:20]}"
        )

    print("Farasa segmentation completed.")
    print("Output records:", len(segmented))

    print("\nSegmentation examples:")
    for index in range(min(3, len(texts))):
        print("Original:", texts[index][:500])
        print("Segmented:", segmented[index][:500])
        print()

    return segmented


# ============================================================
# 5. AraBERT preprocessing after local Farasa
# ============================================================

print("\nPreparing AraBERT preprocessing...")

segmented_texts = farasa_segment_texts(original_texts)


class IdentitySegmenter:
    """
    Farasa has already segmented the text locally.
    This prevents a second segmentation attempt by the preprocessor.
    """

    def segment(self, text):
        return text


arabert_preprocessor = ArabertPreprocessor(
    model_name="bert-base-arabertv2",
    apply_farasa_segmentation=False,
)

# Retain the same setup used in the successful pilot script.
arabert_preprocessor.farasa_segmenter = IdentitySegmenter()

arabert_texts = [
    arabert_preprocessor.preprocess(text)
    for text in segmented_texts
]

if len(arabert_texts) != len(df):
    raise ValueError(
        "AraBERT preprocessing changed the number of records."
    )

if any(not text.strip() for text in arabert_texts):
    raise ValueError("AraBERT preprocessing produced an empty text.")

print("AraBERT preprocessing completed.")
print("Processed texts:", len(arabert_texts))


# ============================================================
# 6. Save metadata before embedding generation
# ============================================================

metadata = df.copy()
metadata["farasa_segmented_text"] = segmented_texts
metadata["arabert_preprocessed_text"] = arabert_texts

metadata_path = OUTPUT_DIR / "full_metadata.csv"
metadata.to_csv(
    metadata_path,
    index=False,
    encoding="utf-8-sig",
)

print("\nMetadata saved:", metadata_path)


# ============================================================
# 7. Masked mean pooling
# ============================================================

def mean_pooling(hidden_states, attention_mask):
    """
    Average token representations while excluding padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ============================================================
# 8. Generate embeddings for all records
# ============================================================

def generate_model_embeddings(model_label, model_name, texts):
    print("\n" + "=" * 60)
    print("Processing:", model_label)
    print("Model:", model_name)
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)
    model.eval()

    batches = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode():
            output = model(**encoded)

            pooled = mean_pooling(
                output.last_hidden_state,
                encoded["attention_mask"],
            )

            normalized = F.normalize(
                pooled,
                p=2,
                dim=1,
            )

        batches.append(normalized.cpu().numpy())

        processed = min(start + len(batch_texts), len(texts))
        if processed % 500 < BATCH_SIZE or processed == len(texts):
            print(f"Processed {processed} / {len(texts)}")

    embeddings = np.concatenate(batches, axis=0)

    expected_shape = (len(df), 768)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"{model_label} has unexpected shape {embeddings.shape}; "
            f"expected {expected_shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"Non-finite values found in {model_label} embeddings."
        )

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            f"{model_label} embeddings are not all L2-normalized."
        )

    output_path = OUTPUT_DIR / f"{model_label}_full_embeddings.npy"
    np.save(output_path, embeddings)

    print("\nEmbedding shape:", embeddings.shape)
    print("All values finite:", np.isfinite(embeddings).all())
    print("Minimum vector norm:", float(norms.min()))
    print("Maximum vector norm:", float(norms.max()))
    print("Saved:", output_path)

    del model, tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return embeddings


# AraBERT v2 receives Farasa-segmented and AraBERT-preprocessed texts.
arabert_embeddings = generate_model_embeddings(
    "AraBERT_v2",
    MODELS["AraBERT_v2"],
    arabert_texts,
)

# MARBERT receives the original text, as in the pilot.
marbert_embeddings = generate_model_embeddings(
    "MARBERT",
    MODELS["MARBERT"],
    original_texts,
)


# ============================================================
# 9. Final alignment and output verification
# ============================================================

print("\n" + "=" * 60)
print("FULL EMBEDDING GENERATION COMPLETED")
print("=" * 60)

print("Dataset records:", len(df))
print("Metadata rows:", len(metadata))
print("AraBERT v2 shape:", arabert_embeddings.shape)
print("MARBERT shape:", marbert_embeddings.shape)

assert len(metadata) == len(df)
assert arabert_embeddings.shape[0] == len(metadata)
assert marbert_embeddings.shape[0] == len(metadata)

summary_path = OUTPUT_DIR / "embedding_generation_summary.txt"

with summary_path.open("w", encoding="utf-8") as summary:
    summary.write("FULL DATASET EMBEDDING GENERATION SUMMARY\n")
    summary.write("=" * 55 + "\n")
    summary.write(f"Input file: {INPUT_FILE}\n")
    summary.write(f"Sheet: {SHEET_NAME}\n")
    summary.write(f"Records: {len(df)}\n")
    summary.write(f"Device: {DEVICE}\n")
    summary.write(f"Batch size: {BATCH_SIZE}\n")
    summary.write(f"Maximum token length: {MAX_LENGTH}\n")
    summary.write("Pooling: attention-mask-aware mean pooling\n")
    summary.write("Normalization: L2 normalization\n")
    summary.write("AraBERT preprocessing: local Farasa + AraBERTPreprocessor\n")
    summary.write("MARBERT preprocessing: original text + MARBERT tokenizer\n")
    summary.write(f"AraBERT v2 shape: {arabert_embeddings.shape}\n")
    summary.write(f"MARBERT shape: {marbert_embeddings.shape}\n")
    summary.write("Metadata and embedding row order: preserved\n")

print("\nSummary saved:", summary_path)
print("\nAll full-dataset output files were saved successfully.")
print("IMPORTANT: Each embedding row matches the same embedding_row in full_metadata.csv.")
