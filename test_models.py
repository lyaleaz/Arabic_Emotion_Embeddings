
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Three Arabic test sentences
sentences = [
    "أنا سعيد جدا بهذا الخبر",
    "أشعر بالحزن الشديد اليوم",
    "هذا الموقف يجعلني غاضبا"
]

models = {
    "AraBERT_v2": "aubmindlab/bert-base-arabertv2",
    "MARBERT": "UBC-NLP/MARBERT"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def mean_pooling(hidden_states, attention_mask):

    mask = attention_mask.unsqueeze(-1).float()

    summed = (hidden_states * mask).sum(dim=1)

    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


for name, model_id in models.items():

    print("\n==============================")
    print(f"Loading {name}...")
    print("==============================")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # Direct tokenization for this technical test
    encoded = tokenizer(
        sentences,
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

    similarity = embeddings @ embeddings.T

    print("Number of sentences:", len(sentences))

    print("Embedding shape:", tuple(embeddings.shape))

    print("\nCosine Similarity Matrix:")

    print(similarity.cpu().numpy().round(4))

    del model, tokenizer, embeddings, output

print("\nBoth models finished successfully!")
