"""
Fetch classic book data from the internet and/or load your own spreadsheet.
Uses CORGIS classics CSV, optional Open Library API (rate-limited), and optional --file for your CSV/JSON.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import requests

# URLs and paths
CORGIS_URL = "https://corgis-edu.github.io/corgis/datasets/csv/classics/classics.csv"
OPEN_LIBRARY_BASE = "https://openlibrary.org"
USER_AGENT = "BookMLRecommender/1.0 (local learning project)"
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_CSV = DATA_DIR / "books_processed.csv"
PROCESSED_JSON = DATA_DIR / "books_processed.json"

# Open Library subjects for richer catalog (rate-limited)
OPEN_LIBRARY_SUBJECTS = [
    "classics",
    "fiction",
    "science_fiction",
    "mystery_and_detective_stories",
    "romance",
    "historical_fiction",
]
WORKS_PER_SUBJECT = 120
REQUEST_DELAY = 1.0
MAX_RETRIES = 3


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _http_get(url, timeout=30, retries=MAX_RETRIES, **kwargs):
    """GET with retries and backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except (requests.RequestException, OSError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return None


def download_corgis():
    path = RAW_DIR / "classics.csv"
    print("  Downloading CORGIS classics CSV...")
    resp = _http_get(CORGIS_URL)
    path.write_text(resp.text, encoding="utf-8")
    print(f"  Saved to {path}")
    return path


def _get_col(row, *candidates):
    for k in candidates:
        if k in row:
            return row[k]
    for col in row:
        cnorm = col.replace(".", "_").replace(" ", "_").lower()
        for k in candidates:
            if cnorm == k.replace(".", "_").replace(" ", "_").lower():
                return row[col]
    for col in row:
        if "title" in col.lower() and "bibliography" in col.lower():
            return row[col]
    return ""


def normalize_corgis_row(row):
    title = (_get_col(row, "bibliography.title", "bibliography_title") or "").strip()
    author = (_get_col(row, "bibliography.author.name", "bibliography_author_name") or "").strip()
    subjects = (_get_col(row, "bibliography.subjects", "bibliography_subjects") or "").strip()[:2000]
    pub = _get_col(row, "bibliography.publication.year", "bibliography_publication_year")
    try:
        publication_year = int(pub) if pub else 0
    except (ValueError, TypeError):
        publication_year = 0
    downloads = _get_col(row, "metadata.downloads", "metadata_downloads")
    try:
        downloads = int(downloads) if downloads else 0
    except (ValueError, TypeError):
        downloads = 0
    rank = _get_col(row, "metadata.rank", "metadata_rank")
    try:
        rank = int(rank) if rank else 0
    except (ValueError, TypeError):
        rank = 0
    flesch = ""
    for col in row:
        if "flesch" in col.lower() and "ease" in col.lower():
            flesch = row[col]
            break
    try:
        flesch_reading_ease = float(flesch) if flesch else 0.0
    except (ValueError, TypeError):
        flesch_reading_ease = 0.0
    return {
        "title": str(title).strip(),
        "author": str(author).strip(),
        "subjects": str(subjects).strip()[:2000],
        "publication_year": publication_year,
        "downloads": downloads,
        "rank": rank,
        "flesch_reading_ease": flesch_reading_ease,
        "source": "corgis",
    }


def load_corgis_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = normalize_corgis_row(row)
            if norm["title"]:
                yield norm


# ---- User file: flexible column names (spreadsheet-style) ----
def _norm_str(v):
    if v is None or (isinstance(v, float) and str(v) == "nan"):
        return ""
    return str(v).strip()


def _find_column(row, *names):
    """Case-insensitive match for column name; also try stripped."""
    keys_lower = {k.strip().lower(): k for k in row}
    for n in names:
        nlo = n.strip().lower()
        if nlo in keys_lower:
            return row.get(keys_lower[nlo])
        for k in keys_lower:
            if nlo in k or k in nlo:
                return row.get(keys_lower[k])
    return ""


def normalize_user_row(row, source="user_file"):
    """Map user CSV/JSON row to common schema. Accepts Title, Author, Subjects, Year, etc."""
    title = _norm_str(_find_column(row, "title", "book title", "name", "book"))
    author = _norm_str(_find_column(row, "author", "authors", "writer", "creator"))
    subjects = _norm_str(_find_column(row, "subjects", "subject", "genre", "genres", "keywords", "tags", "categories"))[:2000]
    year_val = _find_column(row, "year", "publication year", "date", "publication_year", "published")
    try:
        publication_year = int(float(year_val)) if year_val not in (None, "") else 0
    except (ValueError, TypeError):
        publication_year = 0
    return {
        "title": title or "Unknown",
        "author": author or "Unknown",
        "subjects": subjects,
        "publication_year": publication_year,
        "downloads": 0,
        "rank": 0,
        "flesch_reading_ease": 0.0,
        "source": source,
    }


def load_user_file(path):
    """Load user CSV or JSON and yield normalized rows."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    if ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("books", data.get("items", [data]))
        for row in rows:
            if isinstance(row, dict):
                norm = normalize_user_row(row, "user_file")
                if norm["title"] and norm["title"] != "Unknown":
                    yield norm
        return
    # CSV (or .xlsx not supported here; user can export to CSV)
    with open(path, encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = normalize_user_row(row, "user_file")
            if norm["title"] and norm["title"] != "Unknown":
                yield norm


def fetch_open_library_subject(subject, limit=50, offset=0):
    url = f"{OPEN_LIBRARY_BASE}/subjects/{subject}.json"
    params = {"limit": limit, "offset": offset}
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def normalize_open_library_work(work):
    title = work.get("title") or ""
    authors = work.get("authors") or []
    author = authors[0].get("name", "") if authors else ""
    subs = work.get("subject") or []
    subjects = ", ".join(subs[:30])[:2000]
    publication_year = work.get("first_publish_year") or 0
    return {
        "title": str(title).strip(),
        "author": str(author).strip(),
        "subjects": subjects,
        "publication_year": int(publication_year) if publication_year else 0,
        "downloads": 0,
        "rank": 0,
        "flesch_reading_ease": 0.0,
        "source": "open_library",
    }


def fetch_open_library_subjects(subjects, works_per_subject=WORKS_PER_SUBJECT, delay_sec=REQUEST_DELAY):
    seen = set()
    rows = []
    for subj in subjects:
        for offset in range(0, works_per_subject, 50):
            time.sleep(delay_sec)
            try:
                data = fetch_open_library_subject(subj, limit=50, offset=offset)
            except Exception as e:
                print(f"  Warning: Open Library '{subj}' offset {offset}: {e}")
                continue
            works = data.get("works") or []
            if not works:
                break
            for w in works:
                norm = normalize_open_library_work(w)
                if not norm["title"]:
                    continue
                key = (norm["title"].lower().strip(), norm["author"].lower().strip())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(norm)
    return rows


def merge_and_dedupe(*row_lists):
    by_key = {}
    for rows in row_lists:
        for r in rows:
            key = (r["title"].lower().strip(), r["author"].lower().strip())
            by_key[key] = r
    return list(by_key.values())


def load_existing_processed():
    """Load existing processed books if present. Used for additive updates."""
    if PROCESSED_CSV.exists():
        try:
            with open(PROCESSED_CSV, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                return rows
        except Exception:
            pass
    if PROCESSED_JSON.exists():
        try:
            with open(PROCESSED_JSON, encoding="utf-8") as f:
                rows = json.load(f)
            if isinstance(rows, list) and rows:
                return rows
        except Exception:
            pass
    return []


def _safe_int(v, default=0):
    try:
        if v is None or v == "" or (isinstance(v, float) and (v != v or v == float("nan"))):
            return default
        if isinstance(v, str) and v.strip().lower() in ("nan", "none", "."):
            return default
        x = float(v)
        if x != x or x == float("nan"):
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=0.0):
    try:
        if v is None or v == "" or (isinstance(v, float) and (v != v or v == float("nan"))):
            return default
        if isinstance(v, str) and v.strip().lower() in ("nan", "none", "."):
            return default
        x = float(v)
        if x != x or x == float("nan"):
            return default
        return x
    except (TypeError, ValueError):
        return default


def normalize_book_record(r):
    """Ensure every book has the same schema and clean strings."""
    return {
        "title": str(r.get("title") or "").strip()[:500],
        "author": str(r.get("author") or "").strip()[:300],
        "subjects": str(r.get("subjects") or "").strip()[:2000],
        "publication_year": _safe_int(r.get("publication_year")),
        "downloads": _safe_int(r.get("downloads")),
        "rank": _safe_int(r.get("rank")),
        "flesch_reading_ease": _safe_float(r.get("flesch_reading_ease")),
        "source": str(r.get("source") or "merged").strip()[:50],
    }


def save_processed(rows):
    if not rows:
        raise ValueError("No rows to save")
    rows = [normalize_book_record(r) for r in rows]
    keys = list(rows[0].keys())
    with open(PROCESSED_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=0)
    print(f"  Saved {len(rows)} books to data/books_processed.csv and .json")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch book data and/or load your own spreadsheet. Adds to existing dataset each run (additive)."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to your own books CSV or JSON. Merged into existing data (additive).",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch from internet; only merge --file with existing (or use existing only if no --file).",
    )
    parser.add_argument(
        "--open-library",
        action="store_true",
        help="Also fetch from Open Library (classics, fiction, sci-fi, etc.) with rate limiting.",
    )
    parser.add_argument(
        "--open-library-delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Seconds between Open Library requests (default {REQUEST_DELAY})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore existing data and build from scratch (fetch + --file only).",
    )
    args = parser.parse_args()

    ensure_dirs()
    all_rows = []

    # Additive: load existing processed data first (unless --reset)
    if not args.reset:
        existing = load_existing_processed()
        if existing:
            all_rows.append(existing)
            print(f"  Loaded {len(existing)} existing books from data/books_processed.csv")

    if not args.no_fetch:
        path = download_corgis()
        corgis_rows = list(load_corgis_csv(path))
        print(f"  CORGIS: {len(corgis_rows)} books")
        all_rows.append(corgis_rows)

        if args.open_library:
            print("  Fetching Open Library (classics, fiction, sci-fi, mystery, romance, historical)...")
            ol_rows = fetch_open_library_subjects(
                OPEN_LIBRARY_SUBJECTS,
                works_per_subject=WORKS_PER_SUBJECT,
                delay_sec=args.open_library_delay,
            )
            print(f"  Open Library: {len(ol_rows)} works")
            all_rows.append(ol_rows)

    if args.file:
        user_rows = list(load_user_file(args.file))
        print(f"  Your file: {len(user_rows)} books")
        all_rows.append(user_rows)

    if not all_rows:
        print("No data: use --file PATH, or run without --no-fetch to fetch from internet, or remove --reset.")
        return

    merged = merge_and_dedupe(*all_rows)
    save_processed(merged)
    print("Done.")


if __name__ == "__main__":
    main()
