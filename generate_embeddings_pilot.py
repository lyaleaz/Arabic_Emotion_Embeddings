
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
from sklearn.model_selection import train_test_split


# ==========================================
# 1. Configuration
# ==========================================

INPUT_FILE = "Arabic_Emotion_Plutchik8_datasets.xlsx"

OUTPUT_DIR = "pilot_results_farasa"

SAMPLE_SIZE = 100
BATCH_SIZE = 16
RANDOM_STATE = 42

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
# 2. Verify Farasa files
# ==========================================

if not FARASA_JAR.exists():
    raise FileNotFoundError(f"Farasa JAR not found: {FARASA_JAR}")

if not FARASA_DATA.exists():
    raise FileNotFoundError(f"Farasa data not found: {FARASA_DATA}")

os.environ["FarasaDataDir"] = str(FARASA_DATA.resolve()) + os.sep


# ==========================================
# 3. Local Farasa Segmentation
# ==========================================

def farasa_segment_texts(texts):

    print("\nRunning local Farasa segmentation...")

    # Replace line breaks so each original record
    # remains one line in the Farasa input file.
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

                
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8"
            )

            print(result.stdout)

        except subprocess.CalledProcessError as e:
            print("\nFARASA ERROR DETAILS:")
            print("Return code:", e.returncode)
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise


        if not output_file.exists():
            raise RuntimeError("Farasa did not create the output file.")

        segmented_texts = output_file.read_text(
            encoding="utf-8"
        ).splitlines()

    if len(segmented_texts) != len(clean_texts):
        raise ValueError(
            f"Farasa changed the number of records: "
            f"input={len(clean_texts)}, "
            f"output={len(segmented_texts)}"
        )

    if any(not text.strip() for text in segmented_texts):
        raise ValueError("Farasa produced an empty segmented text.")

    print("Farasa segmentation completed.")
    print("Input records:", len(clean_texts))
    print("Output records:", len(segmented_texts))

    print("\nSegmentation examples:")

    for i in range(min(3, len(clean_texts))):
        print("Original:", clean_texts[i])
        print("Segmented:", segmented_texts[i])
        print()

    return segmented_texts


# ==========================================
# 4. Load Dataset
# ==========================================

df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Unified_Data"
)

print("Original records:", len(df))

required_columns = [
    "source_dataset",
    "original_id",
    "text",
    "emotion"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

if df["text"].isna().any():
    raise ValueError("Dataset contains missing texts.")

if df["emotion"].isna().any():
    raise ValueError("Dataset contains missing emotions.")

pilot_df, _ = train_test_split(
    df,
    train_size=SAMPLE_SIZE,
    stratify=df["emotion"],
    random_state=RANDOM_STATE
)

pilot_df = pilot_df.reset_index(drop=True)

print("\nPilot sample size:", len(pilot_df))

print("\nPilot emotion distribution:")
print(pilot_df["emotion"].value_counts())


# ==========================================
# 5. Save Metadata
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = pilot_df.copy()

metadata.to_csv(
    os.path.join(OUTPUT_DIR, "pilot_metadata.csv"),
    index=False,
    encoding="utf-8-sig"
)

original_texts = pilot_df["text"].astype(str).tolist()


# ==========================================
# 6. AraBERT Preprocessing
# ==========================================

print("\nPreparing AraBERT preprocessing...")

segmented_texts = farasa_segment_texts(original_texts)


class IdentitySegmenter:
    """
    Farasa has already segmented the texts.
    This prevents AraBERTPreprocessor from
    attempting a second Farasa segmentation.
    """

    def segment(self, text):
        return text


arabert_preprocessor = ArabertPreprocessor(
    model_name="bert-base-arabertv2",
    apply_farasa_segmentation=False
)

# Supply the local segmentation result instead
# of calling the unavailable Farasa download service.
arabert_preprocessor.farasa_segmenter = IdentitySegmenter()

arabert_texts = [
    arabert_preprocessor.preprocess(text)
    for text in segmented_texts
]

print("AraBERT preprocessing completed.")
print("Number of processed texts:", len(arabert_texts))


# ==========================================
# 7. Mean Pooling
# ==========================================

def mean_pooling(hidden_states, attention_mask):

    mask = attention_mask.unsqueeze(-1).float()

    summed = (hidden_states * mask).sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


# ==========================================
# 8. Generate Embeddings
# ==========================================

for model_label, model_name in MODELS.items():

    print("\n" + "=" * 50)
    print("Processing:", model_label)
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name).to(device)

    model.eval()

    # AraBERT uses Farasa + AraBERT preprocessing.
    # MARBERT uses its original tokenizer directly.
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
            max_length=128,
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

    if not np.isfinite(all_embeddings).all():
        raise ValueError(f"Non-finite embeddings in {model_label}")

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
# 9. Final Verification
# ==========================================

print("\n" + "=" * 50)
print("PILOT COMPLETED")
print("=" * 50)

print("Number of records:", len(metadata))
print("Metadata rows:", len(metadata))

for model_label in MODELS:

    filename = f"{model_label}_pilot_embeddings.npy"

    embeddings = np.load(
        os.path.join(OUTPUT_DIR, filename)
    )

    print(
        model_label,
        "Shape:",
        embeddings.shape,
        "All finite:",
        np.isfinite(embeddings).all()
    )

print("\nAll pilot files saved successfully!")
