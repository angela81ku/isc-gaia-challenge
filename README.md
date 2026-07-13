# Gaia DR3 Variable Star Finder
### InterSystems Employee Programming Challenge #1

Identifies every astronomical source in the Gaia DR3 epoch-photometry archive whose Blue Photometer (BP) or Red Photometer (RP) flux changed by more than 100% across all valid observations.

**Result: 57,099 variable sources** across 20 input files, processed in ~4 seconds.

---

## How It Works

### Pipeline Overview

```
20x EpochPhotometry_*.csv.gz
        |
        | ThreadPoolExecutor (one thread per file)
        v
  _process_file() — Polars: decompress gzip, parse CSV,
                    extract bp_flux / rp_flux columns,
                    compute per-source min/max
        |
        | pl.concat() — merge all 20 DataFrames
        v
  filter: (max - min) / min > 1  →  compute pct_change
        |
        | Google Gemini via langchain-google-genai
        v
  ai_summary.txt — scientific interpretation of top 5 sources
        |
        v
  data/out/result.csv — 57,099 rows
```

---

### Formula: Why `ratio > 2` Instead of `pct > 100`

The requirement says: find sources where `(max - min) / min * 100 > 100`.

We rewrote this as:

```python
# Original — subtraction + division + multiply + compare
pct = (bp_max - bp_min) / bp_min * 100
if pct > 100: ...

# Optimized — single division, direct compare, no multiply
bp_ratio = bp_max / bp_min
rp_ratio = rp_max / rp_min
if max(bp_ratio, rp_ratio) > 2:
    pct = (max_ratio - 1) * 100   # only for the rows that pass
```

**Why this is equivalent:**
```
(max - min) / min > 1
= max/min - 1 > 1
= max/min > 2
```

Polars applies this as a vectorized column expression — no Python loop, no row-by-row iteration.

---

### Parallel Ingestion with Polars

```python
with ThreadPoolExecutor() as ex:
    parts = list(ex.map(_process_file, files))
```

Each thread handles one file independently: gzip decompress → CSV parse → min/max extraction. Polars uses Arrow-backed columnar storage — only the three relevant columns (`source_id`, `bp_flux`, `rp_flux`) are materialized. The per-row flux arrays arrive as JSON-like strings (`[1.2, 3.4, ...]`) and are parsed in a single Polars expression chain with no Python loop.

---

### AI Summary

The top 5 most variable sources are sent to Google Gemini for astronomical interpretation:

```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
response = llm.invoke(prompt)
```

The code also attempts the **InterSystems AI Hub** path first (`langchain_intersystems`). AI Hub is an Early Access SDK that manages LLM configs inside IRIS (`%ConfigStore.Configuration`) and provides a unified `init_chat_model()` interface. However, AI Hub's Python SDK currently requires an external DB-API connection to IRIS (`iris.IRISConnection`), which is not available inside the Embedded Python runtime. The AI Hub path fails gracefully and falls back to direct Gemini calls.

The AI step is skipped if `GEMINI_API_KEY` is not set.

---

## Performance Journey

We profiled the baseline to find the real bottleneck before optimizing:

### Bottleneck Profile (3-file sample, gzip + csv.reader baseline)

| Layer | Time | Share |
|-------|------|-------|
| gzip decompress | 0.711s | **41%** |
| csv.reader parsing | 0.913s | **53%** |
| json.loads (minmax) | 0.105s | 6% |

This told us that `json.loads` — the obvious target — was only 6% of runtime. The real cost was I/O and row parsing.

### What We Tried (20 files, 8 CPUs)

| Method | Time | Speedup | Why |
|--------|------|---------|-----|
| Baseline: gzip + csv.reader + json | 5.2s | 1.00x | — |
| A: isal (faster gzip decoder) | 5.0s | 1.01x | multiprocessing already parallelizes per-file; isal's multi-thread advantage is cancelled out |
| B: isal + pandas | 5.5s | 0.95x | DataFrame construction overhead > savings for single-pass workload |
| C: Parquet + json.loads | 2.9s | 1.82x | Columnar I/O — only 3 of 47 columns read |
| D: Parquet + numpy vectorized | 3.7s | 1.40x | Variable-length arrays require padding; genfromtxt overhead > gains |
| **E: Polars (current)** | **~4s** | **~1.3x** | Arrow-native columnar parse, no intermediate Python lists |

### Why Polars

- **Columnar parsing:** only `source_id`, `bp_flux`, `rp_flux` are decoded — the other 44 CSV columns are skipped.
- **No intermediate Python list:** the flux string `"[1.2, 3.4, ...]"` is parsed entirely in Polars' Rust expression engine (`str.split → cast Float64 → list.min/max`).
- **Single-pass:** `list.min()` and `list.max()` are computed in one scan; there is no separate Python `min()`/`max()` call.

### Why Numpy Vectorization Lost

Gaia flux arrays have variable length — from a handful to hundreds of values per row. To use numpy's matrix operations, every row must be padded to the same length. The padding itself requires a Python loop, and `np.genfromtxt` on a large padded block costs more than the vectorization saves. Numpy vectorization is optimal for **fixed-shape** data (e.g. image pixels), not variable-length arrays.

---

## Correctness Guarantee

Every optimization was verified against a fixed oracle:

```bash
python -m pytest test_runchallenge.py    # 27 unit tests — all pass
```

The 9 floating-point differences found between old and new formula are at the `2.7e-16` level — exactly machine epsilon for `float64`. These are rounding artefacts from operation-order differences at flux magnitudes of `~10^17`, not formula errors.

---

## Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [git](https://git-scm.com)
- Google Gemini API key — free at [aistudio.google.com](https://aistudio.google.com) (AI summary step only)

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | No | — | Enables AI summary step |
| `IRIS_USERNAME` | No | `_SYSTEM` | IRIS login (for local dev tools) |
| `IRIS_PASSWORD` | No | `SYS` | IRIS password (for local dev tools) |

Set on Windows:
```cmd
setx GEMINI_API_KEY "AIzaSy..."
```

### Run

```bash
git clone https://github.com/angela81ku/isc-gaia-challenge.git
cd isc-gaia-challenge
docker-compose up --build -d
docker-compose exec iris iris session IRIS -U USER
USER> do ^RunScript
```

### Output

| File | Contents |
|------|----------|
| `data/out/result.csv` | `source_id, bp_min_flux, bp_max_flux, rp_min_flux, rp_max_flux, percentage_change` — 57,099 rows |
| `data/out/ai_summary.txt` | Gemini interpretation of the top 5 most variable sources (only if `GEMINI_API_KEY` set) |

---

## Project Structure

```
├── src/
│   ├── gaia.py              # Core logic — parallel parse, Polars, Gemini AI summary
│   ├── register_llm.py      # Registers Gemini config in IRIS AI Hub at startup
│   └── RunScript.mac        # IRIS entry point  (do ^RunScript)
├── data/
│   ├── in/                  # 20 EpochPhotometry_*.csv.gz input files
│   └── out/result.csv       # Output — 57,099 variable sources
├── Dockerfile               # IRIS Community 2026.3.0AI + Polars + langchain
├── docker-compose.yml
└── startup.sh               # Compiles RunScript.mac + registers Gemini config
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Database & runtime | InterSystems IRIS Community 2026.3.0AI | Embedded Python runtime, AI Hub SDK |
| Data processing | Polars (Rust/Arrow) | Columnar CSV parse, vectorized min/max, no Python loops |
| Parallel I/O | Python `ThreadPoolExecutor` | One thread per file, parallel gzip decompress |
| AI interpretation | Google Gemini via langchain-google-genai | Free LLM, no local GPU needed |
| AI Hub | InterSystems AI Hub (EAP) + langchain_intersystems | LLM config managed inside IRIS |
| Testing | Python `unittest` — 27 tests | Formula equivalence, filter logic, AI mock |

---

## Data Attribution

Input: ESA Gaia DR3 Epoch Photometry.
Column definitions: [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_photometry/ssec_dm_epoch_photometry.html)
How to cite Gaia: [Credits](https://gea.esac.esa.int/archive/documentation/credits.html)
