# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Project:** PlantBox — Open-source tools for plant bioinformatics
**GitHub:** https://github.com/iantsang779/PlantBox

## Commands

```bash
# Activate the project virtual environment (required — do not use system or miniforge Python)
source ~/python_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server (single worker required — cache is in-memory and not process-safe)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

# Start with auto-reload during development
uvicorn main:app --reload
```

## Git workflow

After completing any change to the codebase, stage the relevant files and **ask the user for permission before committing**. Once approved, commit with a concise message in this format:

```
<type>: <short imperative summary>

[optional body explaining why, not what]
```

Common types: `feat`, `fix`, `refactor`, `chore`, `docs`.

Then ask whether to push to `origin main`.

## Architecture

PlantBox is a single-page FastAPI dashboard that queries the **EMBL-EBI public MySQL server** (`mysql-eg-publicsql.ebi.ac.uk:4157`, user `ensro`, no password) for plant genomic variant, KASP marker, and protein domain data across multiple species.

### Three-database design

At startup, the app discovers and connects to three separate databases on the EBI server:

| Pool | Database pattern | Used by |
|---|---|---|
| `variation_pool` | `triticum_aestivum_variation_*` | Variant/transcript queries |
| `core_pool` | `triticum_aestivum_core_*` | KASP marker queries |
| `compara_pool` | `ensembl_compara_plants_*` | Homology queries |

All three pools are stored on `app.state`. Version selection picks the highest numeric suffix (e.g. `110_1 > 99_1`).

### Request flow

1. `POST /api/query` → validates inputs → `fetch_and_join(variation_pool, core_pool, ...)` in `database.py`
2. Inside `fetch_and_join`: variant query and marker query run **concurrently** via `asyncio.gather` when a `variant_id` is provided; marker query is skipped when only `transcript_id` is given
3. Results are pandas left-joined on `variant_name`, serialised to `list[dict]`, and stored in an `OrderedDict` cache under a UUID token
4. Download endpoints (`/api/download/csv/{token}`, `/api/download/excel/{token}`) do O(1) token lookup — no DB re-query

### In-memory cache

`_result_cache` in `main.py` — TTL 30 min, max 100 entries, FIFO eviction. **Not shared across processes**, hence `--workers 1` is mandatory.

### Frontend

`static/index.html` is a self-contained SPA loaded via `GET /`. It uses Tailwind CSS and Alpine.js from CDN (no build step). All state lives in the `dashboardApp()` Alpine component. Client-side sorting is computed in a `get sortedRows()` getter.

### Annotate Sequence tab

Endpoint: `POST /api/annotate` — accepts `{ gene_id: string }` (validated via `validate_input()`).

**Backend flow:**
1. Two concurrent `httpx` requests to the Ensembl Plants REST API (`rest.ensembl.org`):
   - `GET /lookup/id/{gene_id}?expand=1` — gene metadata + transcript/exon coordinates
   - `GET /sequence/id/{gene_id}?type=genomic` — full genomic sequence
2. For each transcript, `_compute_segments()` maps exon coordinates onto the genomic sequence to produce a list of `{type, seq_start, seq_end, number}` segments (exon or intron). Canonical transcript is sorted first.
3. Result (gene metadata + sequence + per-transcript segments) is stored in `_annotate_cache` (same TTL/eviction policy as `_result_cache`) and returned as an `AnnotateResponse`.

**Frontend behaviour:**
- Users enter an Ensembl gene ID (e.g. `TRAESCS3D02G273600`) and click **Annotate**.
- A transcript selector dropdown lets users switch between transcripts; the canonical transcript is pre-selected.
- The genomic sequence is rendered as colour-coded spans: odd exons (blue), even exons (sky), introns/flanking (grey).
- Hovering a span shows a tooltip with segment type, exon/intron number, and length.

### Annotate Protein tab

Endpoint: `POST /api/annotate_protein` — accepts `{ gene_id, species, domains[] }`.

**Backend flow:**
1. Validate `gene_id` via `validate_input()`; reject unknown `species` (see `SUPPORTED_SPECIES` in `database.py`) or empty `domains`.
2. `_get_species_core_pool(species)` lazily creates and caches a connection pool for the requested species core DB (pattern `{species}_core_*`). Pools are stored in `_species_core_pools` and closed on shutdown.
3. Two concurrent DB calls: `fetch_canonical_transcript_id()` and `fetch_protein_domains()` — both query the species core DB.
4. The canonical transcript ID is used to call `GET /sequence/id/{transcript_id}?type=protein` on the Ensembl REST API (using the transcript ID avoids Ensembl's "multiple sequences" error when querying by gene ID).
5. `fetch_protein_domains()` builds a combined REGEXP from `DOMAIN_PREFIXES` (e.g. `^(PF|PTHR)`) and queries `gene → transcript → translation → protein_feature`, filtering to the canonical transcript only.
6. Result is cached in `_prot_annotate_cache` and returned as `ProteinAnnotateResponse`.

**Frontend behaviour:**
- Users enter a gene ID, select a species, and tick one or more domain databases (Pfam, Panther, Prosite, Prints, Superfamily).
- Client-side `get paIntervals` decomposes the protein sequence at domain boundary breakpoints, priority-sorts overlapping domains (Pfam > Panther > Prosite > Prints > Superfamily), and renders colour-coded `<span>` elements.
- Hovering shows a tooltip with the top domain's type, hit name, description, positions, and a count of additional overlapping domains.

### Pull Homologous Sequences tab

Endpoint: `POST /api/homology` — accepts `{ gene_id: string, homology_type: string }`.

**Valid `homology_type` values** (defined in `HOMOLOGY_TYPES` in `database.py`):
- `"orthologues"` → `ortholog_one2one | ortholog_one2many | ortholog_many2many`
- `"homoeologues"` → `homoeolog_one2one | homoeolog_one2many | homoeolog_many2many`
- `"paralogues"` → `within_species_paralog | other_paralog | gene_split`

**Backend flow:**
1. Validate `gene_id` and `homology_type`; query `compara_pool` via `fetch_homologs()` in `database.py`.
2. `fetch_homologs()` builds a parameterised `OR`-chain of `h.description = %s` clauses from `_HOMOLOGY_SQL` and queries `homology → homology_member → gene_member / genome_db / seq_member / sequence` (joined twice — once for query, once for homologue).
3. Bytes in `query_sequence`, `homolog_sequence`, `query_cigar_line`, `homolog_cigar_line` are decoded UTF-8 if returned as `bytes` (latin1 charset edge case).
4. All 11 columns (including sequences and CIGAR lines) are cached under a UUID token via `_cache_store`; `/api/download/csv` and `/api/download/excel` are reused unchanged.
5. Response includes `display_columns` (7 cols, no sequences) and `all_columns` (11 cols).

**Frontend behaviour:**
- Users enter a gene ID and select a homology type from a dropdown, then click **Search**.
- Results table shows 7 display columns; rows are clickable to populate the alignment viewer.
- Alignment viewer reconstructs pairwise alignments client-side from CIGAR strings (`M` = consume residue, `D` = insert gap). Alignment is wrapped at 60 chars per line with `|` match markers between query (blue) and homologue (green).
- CSV/Excel downloads include all 11 columns (sequences + CIGARs).

### Input validation

All user inputs pass through `validate_input()` in `database.py` before query construction: max 100 chars, regex `^[\w.\-]+$`. SQL parameters are always passed via aiomysql parameterisation (never interpolated).
