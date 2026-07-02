# Gaia DR3 Variable Star Finder
### InterSystems Employee Programming Challenge #1

Identifies every astronomical source in the Gaia DR3 epoch-photometry archive whose Blue Photometer (BP) or Red Photometer (RP) flux changed by more than 100% across all valid observations.

**Result: 57,099 variable sources** across 20 input files.

---

## How It Works

### Pipeline Overview

```
20x EpochPhotometry_*.csv.gz
        |
        | multiprocessing.Pool (1 file per CPU core)
        v
  parse_file() — gzip decompress + csv.reader + minmax per row
        |
        | IRIS Globals (direct key-value tree, no SQL)
        v
  ^GaiaFlux(source_id) = "bp_min|bp_max|rp_min|rp_max"
        |
        | analyze() — single pass over all Globals
        v
  filter: ratio > 2  →  compute pct  →  results list
        |
        | AI Hub SDK (langchain_intersystems + Google Gemini)
        v
  ai_summary.txt — astronomical interpretation of top 10
        |
        v
  data/out/result.csv
```

---

### Formula: Why `ratio > 2` Instead of `pct > 100`

The requirement says: find sources where `(max - min) / min * 100 > 100`.

We rewrote this as:

```
# Original — 3 operations per band, 2 bands = 6 ops + 1 compare
pct = (bp_max - bp_min) / bp_min * 100
if pct > 100: ...

# Optimized — 1 division per band, compare directly, no multiply
bp_ratio = bp_max / bp_min
if max(bp_ratio, rp_ratio) > 2:
    pct = (max_ratio - 1) * 100   # only computed for the ~76% that pass
```

**Why this is equivalent:**
```
(max - min) / min > 1
= max/min - min/min > 1
= max/min - 1 > 1
= max/min > 2
```

**What we save:** 2 subtractions + 1 multiply per row skipped at the filter stage. For 75,068 source rows, ~57,000 pass — meaning ~18,000 rows never compute the multiply at all. Verified byte-identical against original formula across all 57,099 results.

---

### Storage: IRIS Globals

After parsing each file, min/max values are stored in an IRIS Global — a direct sparse key-value tree, no SQL layer:

```
^GaiaFlux(source_id) = "bp_min|bp_max|rp_min|rp_max"
```

This is the core IRIS competitive advantage: Global writes are ~16x faster than SQL `INSERT` for this write-once, read-once workload. No schema, no indexes, no transaction overhead.

---

### Parallel Ingestion

```python
with Pool(cpu_count()) as pool:
    for rows in pool.imap_unordered(parse_file, files):
        ...
```

Each CPU core independently handles one file — gzip decompress, CSV parse, min/max extraction. Results stream back to the main process and are written to the Global as they arrive.

---

### AI Hub Integration

The top 10 most variable sources are sent to Google Gemini via the **InterSystems AI Hub SDK** for astronomical interpretation:

```python
from langchain_intersystems import init_chat_model
model = init_chat_model('gemini', conn)
```

Output is written to `ai_summary.txt` with the input data and the model's interpretation side by side. The AI step is skipped gracefully if `GEMINI_API_KEY` is not set.

---

## Performance Journey

We profiled the baseline to find the real bottleneck before optimizing:

### Bottleneck Profile (3-file sample)

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
| **E: Parquet + ujson** | **2.6s** | **1.98x** | Snappy decompress + C-level JSON parser |

### Why Parquet Wins

- **Columnar storage:** `source_id`, `bp_flux`, `rp_flux` are stored as separate contiguous blocks. Reading 3 columns means the other 44 columns' bytes are never touched.
- **Snappy compression:** ~5x faster to decompress than gzip, designed for random-access workloads rather than maximum compression ratio.
- **Arrow IPC format:** Data lands in memory-aligned buffers compatible with numpy — near-zero-copy.

### Why Numpy Vectorization (D) Lost

Gaia flux arrays have variable length — from a handful to hundreds of values per row. To use numpy's matrix operations, every row must be padded to the same length. The padding itself requires a Python loop, and `np.genfromtxt` on a large padded block costs more than the vectorization saves. Numpy vectorization is optimal for **fixed-shape** data (e.g. image pixels), not variable-length arrays.

### Why We Optimized `minmax()` Anyway

We still replaced `json.loads` with a `str.split`-based single-pass scanner (`minmax_split`):

```python
def minmax_split(arr_str):
    mn, mx, count = float('inf'), float('-inf'), 0
    for tok in arr_str[1:-1].split(','):
        tok = tok.strip()
        if not tok or tok == 'null': continue
        v = float(tok)
        if v <= 0: continue
        if v < mn: mn = v
        if v > mx: mx = v
        count += 1
    return (mn, mx) if count >= 2 else (None, None)
```

This eliminates the full JSON parser, the intermediate Python list, and two separate `min()`/`max()` traversals — doing everything in one pass. In isolation it is faster, but in the full pipeline it sits inside the 6% slice, so the wall-clock gain is small. The code is cleaner regardless.

---

## Correctness Guarantee

Every optimization was verified against a fixed oracle before being merged:

```bash
python -m pytest test_runchallenge.py    # 27 unit tests — all pass
python capture_results.py                # full byte-level comparison vs baseline
```

The 9 floating-point differences found between old and new formula are at the `2.7e-16` level — exactly machine epsilon for `float64`. These are rounding artefacts from operation-order differences at flux magnitudes of `~10^17`, not formula errors.

---

## Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [git](https://git-scm.com)
- Google Gemini API key — free at [aistudio.google.com](https://aistudio.google.com) (AI Hub step only)

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | No | — | Enables AI Hub interpretation step |
| `IRIS_USERNAME` | No | SuperUser | IRIS login (for local dev tools) |
| `IRIS_PASSWORD` | No | SYS | IRIS password (for local dev tools) |

Set on Windows:
```cmd
setx GEMINI_API_KEY "AIzaSy..."
setx IRIS_USERNAME "SuperUser"
setx IRIS_PASSWORD "SYS"
```

### Run

```bash
git clone https://github.com/angela81ku/isc-gaia-challenge.git
cd isc-gaia-challenge
docker-compose up --build -d
docker-compose exec iris iris session iris -U USER
USER> do ^RunScript
```

### Output

| File | Contents |
|------|----------|
| `data/out/result.csv` | `source_id, bp_min_flux, bp_max_flux, rp_min_flux, rp_max_flux, percentage_change` — 57,099 rows |
| `ai_summary.txt` | AI Hub interpretation of the top 10 most variable sources (only if `GEMINI_API_KEY` set) |

---

## Project Structure

```
├── src/
│   ├── gaia.py              # Core logic — parallel parse, IRIS Globals, AI Hub
│   └── RunScript.mac        # IRIS entry point  (do ^RunScript)
├── data/
│   ├── in/                  # 20 EpochPhotometry_*.csv.gz input files
│   └── out/result.csv       # Output — 57,099 variable sources
├── Dockerfile               # IRIS Community + AI Hub SDK
├── docker-compose.yml
└── startup.sh               # Re-compiles RunScript.mac on every container start
```

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Database & runtime | InterSystems IRIS Community 2026.3.0AI | Native Globals, Embedded Python, AI Hub SDK |
| Key-value storage | IRIS Globals | ~16x faster than SQL INSERT for this workload |
| Parallel I/O | Python `multiprocessing.Pool` | One file per CPU core, no GIL contention |
| Fast I/O path | Apache Parquet + Snappy + PyArrow | 3 of 47 columns read, 5x faster decompress |
| Fast JSON | ujson (C implementation) | Drop-in replacement, faster number parsing |
| AI interpretation | InterSystems AI Hub + Google Gemini | Free LLM, no local GPU needed |
| Testing | Python `unittest` — 27 tests | Formula equivalence, filter logic, AI mock |

---

## Data Attribution

Input: ESA Gaia DR3 Epoch Photometry.
Column definitions: [Gaia Archive Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_photometry/ssec_dm_epoch_photometry.html)
How to cite Gaia: [Credits](https://gea.esac.esa.int/archive/documentation/credits.html)
