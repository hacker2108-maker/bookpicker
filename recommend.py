"""
Recommend a book from the trained catalog. Premium terminal output.
Remembers what it already recommended so it won't suggest the same book twice
(use --allow-repeat to ignore history).
Usage:
  python recommend.py                    # pick one or more (diversity-aware)
  python recommend.py --similar "Title"  # recommend similar to given title
"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

MODELS_DIR = Path(__file__).resolve().parent / "models"
DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = MODELS_DIR / "encoder.pt"
BOOKS_JSON = MODELS_DIR / "books.json"
EMBEDDINGS_NPY = MODELS_DIR / "embeddings.npy"
PREPROCESSOR_PKL = MODELS_DIR / "preprocessor.pkl"
HISTORY_PATH = DATA_DIR / "recommendation_history.json"
MAX_HISTORY_ENTRIES = 500

# Fallback for preprocessors saved before we added embed_dim / hidden_dims
DEFAULT_EMBED_DIM = 64
DEFAULT_HIDDEN_DIMS = (128,)


def _str_field(val):
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def load_artifacts():
    if not BOOKS_JSON.exists():
        raise FileNotFoundError(f"Run train.py first. Expected {BOOKS_JSON}")
    with open(BOOKS_JSON, encoding="utf-8") as f:
        books = json.load(f)
    embeddings = np.load(EMBEDDINGS_NPY)
    with open(PREPROCESSOR_PKL, "rb") as f:
        preproc = pickle.load(f)
    input_dim = preproc["num_authors"] + preproc["num_subjects"] + preproc["numeric_dim"]
    embed_dim = preproc.get("embed_dim", DEFAULT_EMBED_DIM)
    hidden_dims = preproc.get("hidden_dims", DEFAULT_HIDDEN_DIMS)

    class EncoderBlock(torch.nn.Module):
        def __init__(self, in_dim, out_dim, h_dims):
            super().__init__()
            dims = [in_dim] + list(h_dims) + [out_dim]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(torch.nn.ReLU())
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if "net.0.weight" in state:
        encoder = EncoderBlock(input_dim, embed_dim, hidden_dims)
    else:
        # Old save format: l1, l2 (single hidden layer)
        class OldEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                h = state["l1.weight"].shape[0]
                self.l1 = torch.nn.Linear(input_dim, h)
                self.l2 = torch.nn.Linear(h, state["l2.weight"].shape[0])

            def forward(self, x):
                return self.l2(torch.relu(self.l1(x)))
        encoder = OldEncoder()
    encoder.load_state_dict(state)
    encoder.eval()
    return books, embeddings, preproc, encoder, input_dim


def book_to_vector(book, preproc):
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
        float(book.get("publication_year") or 0),
        float(book.get("downloads") or 0),
        float(book.get("rank") or 0),
        float(book.get("flesch_reading_ease") or 0),
    ]]).astype(np.float32).flatten()
    return np.concatenate([author_vec, subject_vec, num_vec])


def find_book_by_title(books, query):
    q = _str_field(query).lower()
    if not q:
        return None
    for i, b in enumerate(books):
        if q in _str_field(b.get("title")).lower():
            return i
    return None


def load_history():
    """Load set of (title_lower, author_lower) already recommended."""
    if not HISTORY_PATH.exists():
        return set()
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("recommended", data) if isinstance(data, dict) else data
        return {(e.get("title", "").strip().lower(), e.get("author", "").strip().lower()) for e in entries if isinstance(e, dict)}
    except Exception:
        return set()


def save_history(new_entries):
    """Append new (title, author) to history and trim to MAX_HISTORY_ENTRIES."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, encoding="utf-8") as f:
                data = json.load(f)
            existing = data.get("recommended", data) if isinstance(data, dict) else data
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.extend(new_entries)
    if len(existing) > MAX_HISTORY_ENTRIES:
        existing = existing[-MAX_HISTORY_ENTRIES:]
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump({"recommended": existing}, f, ensure_ascii=False, indent=0)


def indices_already_recommended(books, history_set):
    """Return set of book indices whose (title, author) is in history."""
    return {i for i, b in enumerate(books) if (_str_field(b.get("title")).lower(), _str_field(b.get("author")).lower()) in history_set}


def nearest_indices(embeddings, query_embedding, k=5, exclude_index=None, exclude_indices=None):
    diff = embeddings - query_embedding
    dists = np.linalg.norm(diff, axis=1)
    if exclude_index is not None:
        dists[exclude_index] = np.inf
    if exclude_indices:
        for i in exclude_indices:
            dists[i] = np.inf
    return np.argsort(dists)[:k]


def pick_diverse(embeddings, n, weights=None, exclude_indices=None):
    """Pick n indices with diversity in embedding space; exclude_indices are not chosen."""
    exclude_indices = exclude_indices or set()
    available = [i for i in range(len(embeddings)) if i not in exclude_indices]
    if not available:
        available = list(range(len(embeddings)))
    if n >= len(available):
        return np.array(available[:n])
    if weights is not None:
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        weights_av = np.array([w[i] for i in available], dtype=float)
        weights_av /= weights_av.sum()
    else:
        weights_av = None
    rng = np.random.default_rng()
    first_av = int(rng.choice(len(available), p=weights_av)) if weights_av is not None else rng.integers(0, len(available))
    chosen_global = [available[first_av]]
    for _ in range(n - 1):
        centroid = np.mean(embeddings[chosen_global], axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        for c in chosen_global:
            dists[c] = -np.inf
        for c in exclude_indices:
            dists[c] = -np.inf
        if weights is not None:
            dists = np.maximum(dists, 0) * weights
        chosen_global.append(int(np.argmax(dists)))
    return np.array(chosen_global)


# ---- Premium output ----
def supports_color():
    try:
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    except Exception:
        return False


USE_COLOR = supports_color()

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
TITLE_C = "\033[96m"   # cyan
AUTHOR_C = "\033[93m"  # yellow
META_C = "\033[90m"    # gray
BOX_C = "\033[38;5;245m"  # light gray
ACCENT = "\033[38;5;213m"  # soft purple


def _t(s, color=None):
    return f"{color}{s}{RESET}" if USE_COLOR and color else s


def _box_line(char, width):
    return BOX_C + char * width + RESET if USE_COLOR else char * width


def print_header(title):
    w = 58
    print()
    print(_box_line("╔", w) if USE_COLOR else "=" * (w + 2))
    print((BOX_C + "║ " + RESET if USE_COLOR else "| ") + _t(title.center(w - 2), ACCENT if USE_COLOR else None))
    print(_box_line("╚", w) if USE_COLOR else "=" * (w + 2))
    print()


def print_book_card(b, index=None):
    title = _str_field(b.get("title")) or "Unknown"
    author = _str_field(b.get("author")) or "Unknown"
    year = b.get("publication_year") or None
    subjects = _str_field(b.get("subjects"))
    if subjects:
        subjects = subjects[:120] + ("..." if len(subjects) > 120 else "")

    pre = f"  {index}. " if index is not None else "  "
    print(_t("┌─────────────────────────────────────────────────────────────", BOX_C))
    print(f"  {_t(title, TITLE_C)}")
    print(f"  {_t('by ', DIM)}{_t(author, AUTHOR_C)}")
    if year:
        print(f"  {_t(f'Published {year}', META_C)}")
    if subjects:
        print(f"  {_t(subjects, META_C)}")
    print(_t("└─────────────────────────────────────────────────────────────", BOX_C))
    print()


def main():
    parser = argparse.ArgumentParser(description="Recommend a book from the trained catalog")
    parser.add_argument("--similar", type=str, default=None, metavar="TITLE", help="Recommend books similar to this title")
    parser.add_argument("-n", type=int, default=1, help="Number of recommendations (default 1)")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--allow-repeat", action="store_true", help="Ignore history; allow recommending the same book again")
    args = parser.parse_args()

    if args.no_color:
        global USE_COLOR
        USE_COLOR = False

    books, embeddings, preproc, encoder, input_dim = load_artifacts()

    history_set = set() if args.allow_repeat else load_history()
    exclude_indices = indices_already_recommended(books, history_set) if history_set else set()

    if args.similar is not None:
        idx = find_book_by_title(books, args.similar)
        if idx is None:
            print(f'No book found matching "{args.similar}". Try a partial title.')
            return
        book_vec = book_to_vector(books[idx], preproc)
        x = torch.from_numpy(book_vec.reshape(1, -1))
        with torch.no_grad():
            q_embed = encoder(x).numpy().flatten()
        nearest = nearest_indices(
            embeddings, q_embed,
            k=min(args.n + 10, len(books)),
            exclude_index=idx,
            exclude_indices=exclude_indices,
        )
        # Take first n that are not excluded (nearest_indices already excluded, but we asked for n+10)
        shown = [j for j in nearest[: args.n]]
        print_header(f'Books similar to "{books[idx]["title"]}"')
        for i, j in enumerate(shown):
            print_book_card(books[j], index=i + 1)
        if not args.allow_repeat and shown:
            save_history([{"title": books[j]["title"], "author": books[j].get("author") or ""} for j in shown])
        return

    n = min(args.n, len(books))
    weights = None
    if books and (books[0].get("rank") is not None or books[0].get("downloads") is not None):
        ranks = np.array([b.get("rank") or 1000 for b in books], dtype=float)
        weights = 1.0 / (ranks + 1)
        weights /= weights.sum()
    indices = pick_diverse(embeddings, n, weights, exclude_indices=exclude_indices)
    print_header("Your next read")
    for i, j in enumerate(indices):
        print_book_card(books[j], index=i + 1)
    if not args.allow_repeat and len(indices) > 0:
        save_history([{"title": books[j]["title"], "author": books[j].get("author") or ""} for j in indices])


if __name__ == "__main__":
    main()
