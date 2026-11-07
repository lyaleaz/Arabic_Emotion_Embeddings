
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor


# ==========================================
# 1. Configuration
# ==========================================

FOLDER = "pilot_1000_results_farasa"

METADATA_FILE = os.path.join(
    FOLDER,
    "pilot_metadata.csv"
)

OUTPUT_DIR = FOLDER

BATCH_SIZE = 16
MAX_LENGTH = 128

FARASA_JAR = Path(r"C:\Farasa\Farasa.jar")
FARASA_DATA = Path(r"C:\Farasa\FarasaData")
FARASA_LIB = Path(r"C:\Farasa\lib")

MODELS = {
    "AraBERT_v2": "aubmindlab/bert-base-arabertv2",
    "MARBERT": "UBC-NLP/MARBERT"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# ==========================================
# 2. Verify Files
# ==========================================

if not os.path.exists(METADATA_FILE):
    raise FileNotFoundError(METADATA_FILE)

if not FARASA_JAR.exists():
    raise FileNotFoundError(FARASA_JAR)

if not FARASA_DATA.exists():
    raise FileNotFoundError(FARASA_DATA)

if not (FARASA_LIB / "commons-lang3-3.1.jar").exists():
    raise FileNotFoundError("commons-lang3-3.1.jar not found")

if not (FARASA_LIB / "mapdb-2.0.0-SNAPSHOT.jar").exists():
    raise FileNotFoundError("mapdb-2.0.0-SNAPSHOT.jar not found")

os.environ["FarasaDataDir"] = (
    str(FARASA_DATA.resolve()) + os.sep
)


# ==========================================
# 3. Load Pilot Metadata
# ==========================================

print("\nLoading pilot metadata...")

metadata = pd.read_csv(
    METADATA_FILE,
    encoding="utf-8-sig"
).reset_index(drop=True)

required_columns = [
    "source_dataset",
    "original_id",
    "text",
    "emotion"
]

for column in required_columns:
    if column not in metadata.columns:
        raise ValueError(f"Missing column: {column}")

if len(metadata) != 1000:
    raise ValueError(
        f"Expected 1000 records, found {len(metadata)}"
    )

if metadata[required_columns].isna().any().any():
    raise ValueError("Missing values in required columns.")

original_texts = (
    metadata["text"].astype(str).tolist()
)

print("Number of records:", len(metadata))


# ==========================================
# 4. Farasa Segmentation
# ==========================================

def farasa_segment_texts(texts):

    print("\nRunning local Farasa segmentation...")

    clean_texts = [
        str(text).replace("\r", " ").replace("\n", " ").strip()
        for text in texts
    ]

    with tempfile.TemporaryDirectory() as temp_dir:

        input_file = Path(temp_dir) / "input.txt"
        output_file = Path(temp_dir) / "output.txt"

        input_file.write_text(
            "\n".join(clean_texts) + "\n",
            encoding="utf-8"
        )

        command = [
            "java",
            "-Dfile.encoding=UTF-8",
            "-cp",
            (
                f"{FARASA_JAR};"
                f"{FARASA_LIB / 'commons-lang3-3.1.jar'};"
                f"{FARASA_LIB / 'mapdb-2.0.0-SNAPSHOT.jar'}"
            ),
            "farasa.TestCase",
            "-i",
            str(input_file),
            "-o",
            str(output_file)
        ]

        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        print(result.stdout)

        if not output_file.exists():
            raise RuntimeError(
                "Farasa did not create output file."
            )

        segmented = output_file.read_text(
            encoding="utf-8"
        ).splitlines()

    if len(segmented) != len(clean_texts):
        raise ValueError(
            f"Record mismatch: {len(clean_texts)} input, "
            f"{len(segmented)} output"
        )

    if any(not text.strip() for text in segmented):
        raise ValueError("Empty Farasa output detected.")

    print("Farasa completed successfully.")
    print("Input records:", len(clean_texts))
    print("Output records:", len(segmented))

    return segmented


# ==========================================
# 5. AraBERT Preprocessing
# ==========================================

print("\nPreparing AraBERT preprocessing...")

segmented_texts = farasa_segment_texts(original_texts)


class IdentitySegmenter:

    def segment(self, text):
        return text


arabert_preprocessor = ArabertPreprocessor(
    model_name="bert-base-arabertv2",
    apply_farasa_segmentation=False
)

# Prevent a second Farasa segmentation
arabert_preprocessor.farasa_segmenter = IdentitySegmenter()

arabert_texts = [
    arabert_preprocessor.preprocess(text)
    for text in segmented_texts
]

print("AraBERT preprocessing completed.")


# Save Farasa results for reproducibility

segmentation_df = metadata[
    ["source_dataset", "original_id", "text", "emotion"]
].copy()

segmentation_df["farasa_segmented_text"] = segmented_texts
segmentation_df["arabert_preprocessed_text"] = arabert_texts

segmentation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "pilot_1000_preprocessing_audit.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 6. Mean Pooling
# ==========================================

def mean_pooling(hidden_states, attention_mask):

    mask = attention_mask.unsqueeze(-1).float()

    summed = (hidden_states * mask).sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


# ==========================================
# 7. Generate Embeddings
# ==========================================

for model_label, model_name in MODELS.items():

    print("\n" + "=" * 55)
    print("Processing:", model_label)
    print("=" * 55)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name).to(device)

    model.eval()

    if model_label == "AraBERT_v2":
        texts = arabert_texts
    else:
        texts = original_texts

    all_embeddings = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.inference_mode():

            output = model(**encoded)

            embeddings = mean_pooling(
                output.last_hidden_state,
                encoded["attention_mask"]
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

        print(
            f"Processed {min(start + BATCH_SIZE, len(texts))}"
            f" / {len(texts)}"
        )

    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    if all_embeddings.shape != (1000, 768):
        raise ValueError(
            f"Unexpected embedding shape: {all_embeddings.shape}"
        )

    if not np.isfinite(all_embeddings).all():
        raise ValueError("Non-finite embedding values detected.")

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{model_label}_pilot_embeddings.npy"
    )

    np.save(output_file, all_embeddings)

    print("\nEmbedding shape:", all_embeddings.shape)
    print("Saved:", output_file)

    del model, tokenizer

    if device == "cuda":
        torch.cuda.empty_cache()


# ==========================================
# 8. Final Verification
# ==========================================

print("\n" + "=" * 55)
print("PILOT 1000 EMBEDDINGS COMPLETED")
print("=" * 55)

for model_label in MODELS:

    filename = os.path.join(
        OUTPUT_DIR,
        f"{model_label}_pilot_embeddings.npy"
    )

    embeddings = np.load(filename)

    norms = np.linalg.norm(embeddings, axis=1)

    print("\nModel:", model_label)
    print("Shape:", embeddings.shape)
    print("All finite:", np.isfinite(embeddings).all())
    print("Mean vector norm:", norms.mean())

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError("L2 normalization verification failed.")

print("\nAll files saved successfully!")
