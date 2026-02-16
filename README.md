# Book ML Recommendation System

A **local, production-style** book recommendation system: it fetches or loads book data, trains a neural network on metadata, and recommends your next read—all from the terminal, no APIs at inference time. Designed for Windows and suitable for public use.

---

## Table of contents

- [What it does](#what-it-does)
- [How it works (architecture)](#how-it-works-architecture)
- [Quick start (Windows)](#quick-start-windows)
- [Additive data (fetch adds to your dataset)](#additive-data-fetch-adds-to-your-dataset)
- [Running with Python](#running-with-python)
- [Fetch options (full reference)](#fetch-options-full-reference)
- [Train and recommend](#train-and-recommend)
- [Project layout and files](#project-layout-and-files)
- [Data sources](#data-sources)
- [Model and training (technical)](#model-and-training-technical)
- [Recommendation logic](#recommendation-logic)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## License

This project is licensed under the **MIT License**. You may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software as you want, subject to including the copyright notice and license text. See the [LICENSE](LICENSE) file in the repo for the full text.

---

## What it does

1. **Fetch** – Downloads classic books from the internet (CORGIS, optional Open Library) and/or loads your own CSV/JSON. **Each run adds to the existing dataset** so the catalog grows over time.
2. **Train** – Builds a neural network (autoencoder) on book metadata (author, subjects, year, etc.), learns a compact representation (embedding) for each book, and saves the model and preprocessor.
3. **Recommend** – Picks a book at random (with diversity) or finds books similar to a title you give. **Remembers what it already recommended** so it won’t suggest the same book twice (history is stored in `data/recommendation_history.json`). Use `--allow-repeat` to ignore history.

Everything runs **locally** after the initial data fetch; no cloud or API is needed for training or recommendations.

---

## How it works (architecture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  FETCH (fetch_data.py)                                                   │
│  • Loads existing data/books_processed.csv if present (additive)         │
│  • Downloads CORGIS classics CSV                                          │
│  • Fetches Gutendex (Project Gutenberg public API, no key)               │
│  • Optionally Open Library by subject (--open-library)                    │
│  • Optionally your --file (CSV/JSON)                                     │
│  • All APIs used are free public APIs (no API keys)                       │
│  • Merges all, dedupes, normalizes schema → data/books_processed.*       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  TRAIN (train.py)                                                        │
│  • Reads data/books_processed.csv (or .json)                             │
│  • Builds preprocessor: author vocab (top 300), subject vocab (top 500),  │
│    numeric scaler (year, downloads, rank, readability)                    │
│  • Encodes each book as a fixed-size feature vector                      │
│  • Trains autoencoder: input → encoder → embedding (128-d) → decoder →   │
│    reconstruction; loss = MSE(input, reconstruction)                     │
│  • Saves: models/encoder.pt, books.json, embeddings.npy, preprocessor.pkl │
│  • Also saves: config.json (hyperparameters), training_info.json (stats)  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RECOMMEND (recommend.py)                                                 │
│  • Loads model, books, embeddings, preprocessor, recommendation history  │
│  • “Pick one”: diversity-aware sampling; excludes already-recommended    │
│    books; appends this run’s picks to history                             │
│  • “Similar to X”: k nearest in embedding space; excludes history;        │
│    appends to history so those titles aren’t suggested again as “pick”    │
│  • History: data/recommendation_history.json (last 500 entries)           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick start (Windows)

1. **One-time setup**
   ```cmd
   setup.bat
   ```
   Creates a virtual environment and installs dependencies (PyTorch, pandas, etc.). If the download times out, run `setup.bat` again.

2. **Full pipeline (fetch → train → recommend)**
   ```cmd
   run_all.bat
   ```

3. **Get more recommendations later**
   ```cmd
   recommend.bat
   recommend.bat -n 3
   recommend.bat --similar "Crime and Punishment" -n 5
   ```

---

## Additive data (fetch adds to your dataset)

By default, **fetch does not replace your data**; it **adds** to it.

- On each run, `fetch_data.py` **loads existing** `data/books_processed.csv` (or `.json`) if it exists.
- Then it fetches CORGIS (and optionally Open Library) and/or loads your `--file`.
- All sources are **merged** and **deduplicated** by (title, author). Duplicates keep one row.
- The combined set is saved back to `data/books_processed.csv` and `.json`.

So:

- First run: no existing file → you get CORGIS (+ optional Open Library + optional file).
- Second run: existing ~1000 books → you add CORGIS again (mostly duplicates, deduped), plus any new Open Library or new `--file` data. **Total book count can only grow or stay the same.**

To **start from scratch** (ignore existing data and rebuild only from fetch + `--file`), use:

```cmd
fetch.bat --reset
```

Or to only add your file to what you already have:

```cmd
fetch.bat --file "C:\path\to\more_books.csv"
```

Then run `train.bat` so the model is retrained on the updated catalog.

---

## Running with Python

From the project folder, activate the venv, then run the scripts.

**Command Prompt**
```cmd
venv\Scripts\activate.bat
python fetch_data.py
python train.py
python recommend.py
```

**PowerShell**
```powershell
.\venv\Scripts\Activate.ps1
python fetch_data.py
python train.py
python recommend.py
```

**Examples**
```cmd
python fetch_data.py --open-library
python fetch_data.py --file "D:\books.csv"
python fetch_data.py --reset --open-library
python recommend.py --similar "Pride and Prejudice" -n 5
python recommend.py --no-color
```

---

## Fetch options (full reference)

| Option | Description |
|--------|-------------|
| (none) | Load existing (if any), fetch CORGIS + Gutendex, merge and save. |
| **--file PATH** | Load your CSV or JSON and merge into the dataset (additive). |
| **--no-fetch** | Do not fetch from internet. Only merge existing + `--file` (or keep existing only if no `--file`). |
| **--no-gutendex** | Skip Gutendex; only fetch CORGIS (faster, fewer books). |
| **--gutendex-pages N** | Max Gutendex pages (default 20). |
| **--open-library** | Also fetch from Open Library (classics, fiction, sci-fi, mystery, romance, historical fiction) with rate limiting. |
| **--open-library-delay N** | Seconds between Open Library API requests (default: 1.0). |
| **--reset** | Ignore existing `books_processed`; build only from this run’s fetch and `--file`. |

All data sources are **free public APIs**; no API keys are required.

**Your CSV/JSON format**

- CSV: columns such as **Title**, **Author**, **Subjects** (or **Genre**), **Year** (optional). Names are flexible (e.g. "Book Title", "Authors", "Publication Year").
- JSON: array of objects with the same field names, or `{"books": [...]}`.

---

## Train and recommend

**Train**

- Reads `data/books_processed.csv` (or `.json`).
- Builds preprocessor and model, trains the autoencoder, saves everything under `models/`.
- Writes `models/config.json` (hyperparameters) and `models/training_info.json` (e.g. number of books, final loss).

**Recommend**

- **Pick one or more:** `recommend.bat` or `recommend.bat -n 3`. Uses diversity in embedding space. **Books already recommended are excluded** so you don’t get the same suggestion twice (history is saved in `data/recommendation_history.json`).
- **Similar to a title:** `recommend.bat --similar "Crime and Punishment" -n 5`. Finds nearest neighbors; also skips books in history and adds this run’s suggestions to history.
- **Allow repeats:** `recommend.bat --allow-repeat` ignores history and does not append to it.
- **Plain output:** `recommend.bat --no-color` disables ANSI colors.

---

## Project layout and files

| Path | Purpose |
|------|---------|
| **setup.bat** | One-time: create venv, install dependencies. |
| **run_all.bat** | Run fetch → train → recommend. |
| **fetch.bat** | Run fetch_data.py (passes arguments through). |
| **train.bat** | Run train.py. |
| **recommend.bat** | Run recommend.py (passes arguments through). |
| **fetch_data.py** | Fetch and/or load data; merge into existing; save processed books. |
| **train.py** | Load processed books, train autoencoder, save model and artifacts. |
| **recommend.py** | Load model and books; print recommendations (random or similar). |
| **requirements.txt** | Python dependencies (torch, pandas, numpy, requests, scikit-learn). |
| **data/** | Fetched and processed data. |
| **data/raw/** | Raw downloads (e.g. CORGIS classics.csv). |
| **data/books_processed.csv** | Unified book table (additive across runs). |
| **data/books_processed.json** | Same data in JSON. |
| **data/recommendation_history.json** | Books already recommended (so they aren’t suggested again). |
| **models/** | Trained model and artifacts. |
| **models/encoder.pt** | Encoder weights (PyTorch). |
| **models/books.json** | Book list used for recommendations. |
| **models/embeddings.npy** | Precomputed book embeddings. |
| **models/preprocessor.pkl** | Author/subject vocabs and scaler. |
| **models/config.json** | Model hyperparameters. |
| **models/training_info.json** | Last training run (e.g. book count, final loss). |

---

## Data sources (all free public APIs, no keys)

- **CORGIS classics** – Single CSV of ~1,000 classic books (Project Gutenberg). One HTTP GET per run.
- **Gutendex** – [Project Gutenberg](https://gutendex.com/) free public JSON API; no key. Fetched by default (many more English classics). Use `--no-gutendex` to skip.
- **Open Library** – Free [Subjects API](https://openlibrary.org/developers/api). Used when you pass `--open-library`. Subjects: classics, fiction, science_fiction, mystery, romance, historical_fiction. Rate-limited.
- **Your file** – Any CSV or JSON with title, author, subjects/genre, optional year. Merged additively.

---

## Model and training (technical)

- **Input per book:** one-hot author (top 300), multi-hot subjects (top 500), 4 numeric features (publication year, downloads, rank, Flesch reading ease), all normalized.
- **Architecture:** autoencoder — input → 256 → 128 → 128-d embedding → 128 → 256 → input. ReLU between layers; no activation on embedding or output.
- **Training:** MSE reconstruction loss, Adam optimizer, 120 epochs, batch size 64, learning rate 8e-4. No validation split; single train on full processed set.
- **Output:** encoder saved for inference; book list and precomputed embeddings saved so recommendations don’t need to run the encoder for the full catalog.

---

## Recommendation logic

- **History:** Each time you run recommend (pick one or similar), the suggested books are appended to `data/recommendation_history.json` (up to the last 500 entries). Future runs **exclude** any book in that history so the same book is not recommended again. Use `--allow-repeat` to disable this.
- **“Pick one” (or `-n N`):** Excludes books in history, then weights by inverse rank (when available) and uses **diversity sampling**: iteratively pick the next book farthest from the centroid of already chosen books in embedding space. If every book was already recommended, it falls back to allowing repeats for that run.
- **“Similar to Title”:** Embeds the matching book, finds k nearest by L2 distance (excluding the query book and books in history), prints formatted cards, and adds those suggestions to history.

---

## Troubleshooting

- **Fetch: “No data”** – Run without `--no-fetch`, or provide `--file PATH`, or remove `--reset` if you expected to use existing data.
- **Train: “Run fetch_data.py first”** – Ensure `data/books_processed.csv` or `data/books_processed.json` exists (run `fetch.bat` or `fetch_data.py` at least once).
- **Recommend: “Run train.py first”** – Ensure `models/books.json` and other artifacts exist (run `train.bat` or `train.py` after fetch).
- **pip timeout** – Use `pip install --timeout 600 -r requirements.txt` (or run `setup.bat`, which uses a long timeout).
- **Open Library rate limit** – Increase `--open-library-delay` (e.g. 2.0).

---

## License and public use

This project is **MIT licensed** so you can modify and use it as you want. The `data/` and `models/` folders are typically not committed (see `.gitignore`) so the repo stays small; users run `setup.bat` then fetch/train/recommend to generate their own data and models. All book data is fetched from free public APIs (CORGIS, Gutendex, Open Library); no API keys are required.
