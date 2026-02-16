"""
Train a small neural network on book metadata and save model + book index.
Loads data/books_processed.csv, encodes features, trains an autoencoder for embeddings.
"""
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
PROCESSED_CSV = DATA_DIR / "books_processed.csv"
PROCESSED_JSON = DATA_DIR / "books_processed.json"
MODEL_PATH = MODELS_DIR / "encoder.pt"
BOOKS_JSON = MODELS_DIR / "books.json"
EMBEDDINGS_NPY = MODELS_DIR / "embeddings.npy"
PREPROCESSOR_PKL = MODELS_DIR / "preprocessor.pkl"
TRAINING_INFO_JSON = MODELS_DIR / "training_info.json"
CONFIG_JSON = MODELS_DIR / "config.json"

# Model and preprocessing constants
MAX_AUTHORS = 300
MAX_SUBJECTS = 500
EMBED_DIM = 128
HIDDEN_DIMS = (256, 128)  # encoder hidden layers
EPOCHS = 120
BATCH_SIZE = 64
LR = 8e-4


def _str_field(val):
    """Normalize a field to str; handle None and pandas NaN from CSV."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def _float_field(val, default=0.0):
    """Convert to float; handle None and NaN from CSV."""
    try:
        x = float(val)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default


def load_books():
    """Load processed books from CSV or JSON."""
    if PROCESSED_CSV.exists():
        df = pd.read_csv(PROCESSED_CSV)
        return df.to_dict("records")
    if PROCESSED_JSON.exists():
        with open(PROCESSED_JSON, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"Run fetch_data.py first. Expected {PROCESSED_CSV} or {PROCESSED_JSON}"
    )


def build_preprocessor(books):
    """Build author encoder, subject vocab, and scaler from book list."""
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    authors = [_str_field(b.get("author")) for b in books]
    le = LabelEncoder()
    le.fit(authors)
    # Keep only top MAX_AUTHORS by frequency
    from collections import Counter
    top_authors = [a for a, _ in Counter(authors).most_common(MAX_AUTHORS)]
    author_to_idx = {a: i for i, a in enumerate(top_authors)}

    subject_counts = Counter()
    for b in books:
        subs = _str_field(b.get("subjects")).split(",")
        for s in subs:
            s = s.strip().lower()
            if s and len(s) > 1:
                subject_counts[s] += 1
    subject_vocab = [s for s, _ in subject_counts.most_common(MAX_SUBJECTS)]
    subject_to_idx = {s: i for i, s in enumerate(subject_vocab)}

    numeric = []
    for b in books:
        numeric.append([
            _float_field(b.get("publication_year")),
            _float_field(b.get("downloads")),
            _float_field(b.get("rank")),
            _float_field(b.get("flesch_reading_ease")),
        ])
    X_num = np.array(numeric, dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(X_num)

    return {
        "author_to_idx": author_to_idx,
        "subject_to_idx": subject_to_idx,
        "scaler": scaler,
        "num_authors": len(author_to_idx) + 1,
        "num_subjects": len(subject_to_idx),
        "numeric_dim": 4,
        "embed_dim": EMBED_DIM,
        "hidden_dims": HIDDEN_DIMS,
    }


def book_to_vector(book, preproc):
    """Convert one book dict to a fixed-size feature vector."""
    author_to_idx = preproc["author_to_idx"]
    subject_to_idx = preproc["subject_to_idx"]
    scaler = preproc["scaler"]
    num_authors = preproc["num_authors"]
    num_subjects = preproc["num_subjects"]

    author_vec = np.zeros(num_authors, dtype=np.float32)
    aidx = author_to_idx.get(_str_field(book.get("author")), None)
    if aidx is not None:
        author_vec[aidx] = 1.0
    else:
        author_vec[-1] = 1.0

    subject_vec = np.zeros(num_subjects, dtype=np.float32)
    for s in _str_field(book.get("subjects")).split(","):
        s = s.strip().lower()
        if s in subject_to_idx:
            subject_vec[subject_to_idx[s]] = 1.0

    num_vec = scaler.transform([[
        _float_field(book.get("publication_year")),
        _float_field(book.get("downloads")),
        _float_field(book.get("rank")),
        _float_field(book.get("flesch_reading_ease")),
    ]]).astype(np.float32).flatten()

    return np.concatenate([author_vec, subject_vec, num_vec])


def build_dataset(books, preproc):
    """Build (N, input_dim) float32 array."""
    vectors = [book_to_vector(b, preproc) for b in books]
    return np.stack(vectors)


class EncoderBlock(nn.Module):
    """Encoder: input -> hidden_layers -> embed_dim. Same structure in recommend.py."""
    def __init__(self, input_dim, embed_dim, hidden_dims):
        super().__init__()
        dims = [input_dim] + list(hidden_dims) + [embed_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, embed_dim=EMBED_DIM, hidden_dims=HIDDEN_DIMS):
        super().__init__()
        self.encoder = EncoderBlock(input_dim, embed_dim, hidden_dims)
        dec_dims = [embed_dim] + list(reversed(hidden_dims)) + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


def main():
    print("Loading books...")
    books = load_books()
    print(f"  {len(books)} books")

    print("Building preprocessor...")
    preproc = build_preprocessor(books)
    input_dim = preproc["num_authors"] + preproc["num_subjects"] + preproc["numeric_dim"]
    print(f"  Input dim: {input_dim}")

    X = build_dataset(books, preproc)
    X_t = torch.from_numpy(X)

    model = Autoencoder(input_dim, embed_dim=EMBED_DIM, hidden_dims=HIDDEN_DIMS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    model.train()
    n = len(X_t)
    final_loss = None
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        total_loss = 0.0
        count = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            batch = X_t[idx]
            opt.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            count += 1
        final_loss = total_loss / max(count, 1)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{EPOCHS} loss={final_loss:.4f}")

    model.eval()
    with torch.no_grad():
        _, embeddings = model(X_t)
        embeddings_np = embeddings.numpy()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), MODEL_PATH)
    books_clean = [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in b.items()}
        for b in books
    ]
    with open(BOOKS_JSON, "w", encoding="utf-8") as f:
        json.dump(books_clean, f, ensure_ascii=False, indent=0)
    np.save(EMBEDDINGS_NPY, embeddings_np)
    with open(PREPROCESSOR_PKL, "wb") as f:
        pickle.dump(preproc, f)

    config = {
        "embed_dim": EMBED_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "input_dim": input_dim,
        "max_authors": MAX_AUTHORS,
        "max_subjects": MAX_SUBJECTS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    }
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    training_info = {
        "num_books": len(books),
        "final_loss": round(final_loss, 6),
        "embed_dim": EMBED_DIM,
        "input_dim": input_dim,
    }
    with open(TRAINING_INFO_JSON, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved {len(books)} books to {BOOKS_JSON}")
    print(f"Saved embeddings to {EMBEDDINGS_NPY}")
    print(f"Saved preprocessor to {PREPROCESSOR_PKL}")
    print(f"Saved config to {CONFIG_JSON}")
    print(f"Saved training info to {TRAINING_INFO_JSON}")
    print("Done.")


if __name__ == "__main__":
    main()
