"""
main.py — FastAPI application for PlantBox.

Endpoints:
  GET  /                           → serves static/index.html
  POST /api/query                  → runs DB queries, caches result, returns JSON
  GET  /api/download/csv/{token}   → streams CSV from in-memory cache
  GET  /api/download/excel/{token} → streams Excel from in-memory cache
"""

import asyncio
import io
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

from database import (
    _COMPARA_DB_PREFIX,
    _CORE_DB_PREFIX,
    _VARIATION_DB_PREFIX,
    DOMAIN_PREFIXES,
    HOMOLOGY_TYPES,
    SUPPORTED_SPECIES,
    create_pool,
    discover_latest_db,
    fetch_and_join,
    fetch_canonical_transcript_id,
    fetch_homologs,
    fetch_protein_domains,
    validate_input,
)


# ---------------------------------------------------------------------------
# In-memory result cache
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 30 * 60   # 30 minutes
_CACHE_MAX_ENTRIES = 100

ENSEMBL_REST_BASE     = "https://rest.ensembl.org"
_MAX_SEQUENCE_IDS     = 50
_ENSEMBL_TIMEOUT_SECS = 30.0
_VALID_SEQ_TYPES      = {"genomic", "cdna", "cds", "protein"}

async def _fetch_strandedness(transcript_ids: list[str]) -> dict[str, str]:
    """
    Batch-fetch strand info from the Ensembl REST POST /lookup/id endpoint.
    Returns a dict mapping transcript_id → "Forward" | "Reverse".
    Failures are silently swallowed; missing IDs will receive no Strandedness value.
    """
    if not transcript_ids:
        return {}
    try:
        async with httpx.AsyncClient(timeout=_ENSEMBL_TIMEOUT_SECS) as client:
            resp = await client.post(
                f"{ENSEMBL_REST_BASE}/lookup/id",
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                json={"ids": transcript_ids},
            )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        result: dict[str, str] = {}
        for tid, info in data.items():
            if isinstance(info, dict) and "strand" in info:
                result[tid] = "Forward" if info["strand"] == 1 else "Reverse"
        return result
    except Exception:
        return {}


# {token: {"data": list[dict], "columns": list[str], "ts": float}}
_result_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _cache_store(token: str, data: list[dict], columns: list[str], filename: str = "wheat_variants") -> None:
    """Insert a result into the cache, evicting the oldest entry if full."""
    if len(_result_cache) >= _CACHE_MAX_ENTRIES:
        _result_cache.popitem(last=False)
    _result_cache[token] = {"data": data, "columns": columns, "filename": filename, "ts": time.time()}


def _cache_get(token: str) -> dict[str, Any]:
    """
    Retrieve a cached result by token.
    Raises HTTPException(404) if not found or expired.
    """
    entry = _result_cache.get(token)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Result expired or not found. Please run the query again."
            },
        )
    age = time.time() - entry["ts"]
    if age > _CACHE_TTL_SECONDS:
        del _result_cache[token]
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Result expired (30-minute TTL). Please run the query again."
            },
        )
    return entry


# ---------------------------------------------------------------------------
# FASTA cache (separate from tabular result cache)
# ---------------------------------------------------------------------------

# {token: {"fasta": str, "gene_count": int, "ts": float}}
_fasta_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _fasta_cache_store(token: str, fasta: str, gene_count: int) -> None:
    """
    Insert a FASTA result into the FASTA cache, evicting the oldest entry if full.

    Args:
        token: UUID string used as the cache key.
        fasta: Raw FASTA-formatted string returned by the Ensembl sequence endpoint.
        gene_count: Number of gene IDs whose sequences are contained in *fasta*.
    """
    if len(_fasta_cache) >= _CACHE_MAX_ENTRIES:
        _fasta_cache.popitem(last=False)
    _fasta_cache[token] = {"fasta": fasta, "gene_count": gene_count, "ts": time.time()}


def _fasta_cache_get(token: str) -> dict[str, Any]:
    """
    Retrieve a FASTA result from the cache by token.

    Args:
        token: UUID string used as the cache key.

    Returns:
        The cached entry dict containing keys ``fasta``, ``gene_count``, and ``ts``.

    Raises:
        HTTPException(404): If the token is not present or the entry has exceeded the
            30-minute TTL.
    """
    entry = _fasta_cache.get(token)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "Result expired or not found. Please run the query again."},
        )
    if time.time() - entry["ts"] > _CACHE_TTL_SECONDS:
        del _fasta_cache[token]
        raise HTTPException(
            status_code=404,
            detail={"error": "Result expired (30-minute TTL). Please run the query again."},
        )
    return entry


# ---------------------------------------------------------------------------
# Annotate cache (separate from FASTA and tabular caches)
# ---------------------------------------------------------------------------

# {token: {"data": dict, "ts": float}}
_annotate_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _annotate_cache_store(token: str, data: dict) -> None:
    """
    Insert a gene annotation result into the annotate cache, evicting the oldest
    entry if the cache has reached its maximum size.

    Args:
        token: UUID string used as the cache key.
        data: Serialised ``AnnotateResponse`` dict (gene metadata + transcript segments).
    """
    if len(_annotate_cache) >= _CACHE_MAX_ENTRIES:
        _annotate_cache.popitem(last=False)
    _annotate_cache[token] = {"data": data, "ts": time.time()}


# ---------------------------------------------------------------------------
# Protein annotation cache
# ---------------------------------------------------------------------------

_prot_annotate_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


_promoter_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

def _promoter_cache_store(token: str, data: dict) -> None:
    """
    Insert a promoter annotation result into the promoter cache, evicting the
    oldest entry if the cache has reached its maximum size.

    Args:
        token: UUID string used as the cache key.
        data: Serialised ``PromoterResponse`` dict to cache.
    """
    if len(_promoter_cache) >= _CACHE_MAX_ENTRIES:
        _promoter_cache.popitem(last=False)
    _promoter_cache[token] = {"data": data, "ts": time.time()}


def _prot_cache_store(token: str, data: dict) -> None:
    """
    Insert a protein annotation result into the protein annotation cache, evicting
    the oldest entry if the cache has reached its maximum size.

    Args:
        token: UUID string used as the cache key.
        data: Serialised ``ProteinAnnotateResponse`` dict (protein sequence + domain features).
    """
    if len(_prot_annotate_cache) >= _CACHE_MAX_ENTRIES:
        _prot_annotate_cache.popitem(last=False)
    _prot_annotate_cache[token] = {"data": data, "ts": time.time()}


# ---------------------------------------------------------------------------
# Per-species core pool manager
# ---------------------------------------------------------------------------

_species_core_pools: dict[str, Any] = {}


async def _get_species_core_pool(species: str):
    """
    Return a connection pool for the core DB of the given species, creating it
    lazily on first access.

    Pools are stored in the module-level ``_species_core_pools`` dict and reused
    across requests.  The database name is resolved via ``discover_latest_db``
    using the pattern ``{species}_core_*``.

    Args:
        species: Ensembl species key, e.g. ``"oryza_sativa"``.  Must be a member
            of ``SUPPORTED_SPECIES``; callers are responsible for validating this
            before invoking the function.

    Returns:
        An ``aiomysql.Pool`` connected to the latest core DB for *species*.
    """
    if species not in _species_core_pools:
        db_name = await discover_latest_db(f"{species}_core_")
        _species_core_pools[species] = await create_pool(db_name)
    return _species_core_pools[species]


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Discover the latest variation + core + compara DBs and create three connection pools."""
    import asyncio as _asyncio
    variation_db, core_db, compara_db = await _asyncio.gather(
        discover_latest_db(_VARIATION_DB_PREFIX),
        discover_latest_db(_CORE_DB_PREFIX),
        discover_latest_db(_COMPARA_DB_PREFIX),
    )
    variation_pool, core_pool, compara_pool = await _asyncio.gather(
        create_pool(variation_db),
        create_pool(core_db),
        create_pool(compara_db),
    )
    app.state.variation_pool = variation_pool
    app.state.core_pool = core_pool
    app.state.compara_pool = compara_pool
    app.state.variation_db_name = variation_db
    app.state.core_db_name = core_db
    app.state.compara_db_name = compara_db
    print(f"[startup] variation DB : {variation_db}")
    print(f"[startup] core DB      : {core_db}")
    print(f"[startup] compara DB   : {compara_db}")
    yield
    variation_pool.close()
    core_pool.close()
    compara_pool.close()
    for pool in _species_core_pools.values():
        pool.close()
    await _asyncio.gather(
        variation_pool.wait_closed(),
        core_pool.wait_closed(),
        compara_pool.wait_closed(),
        *(_p.wait_closed() for _p in _species_core_pools.values()),
    )
    print("[shutdown] All connection pools closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Wheat Variation Explorer", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    variant_id: str | None = None
    transcript_id: str | None = None
    consequence_types: list[str] | None = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> "QueryRequest":
        """Ensure at least one of variant_id or transcript_id is supplied."""
        if not self.variant_id and not self.transcript_id:
            raise ValueError(
                "At least one search parameter (variant_id or transcript_id) must be provided."
            )
        return self


class QueryResponse(BaseModel):
    token: str
    columns: list[str]
    rows: list[dict]
    db_name: str
    row_count: int
    has_markers: bool


class SequenceRequest(BaseModel):
    gene_ids: list[str]
    sequence_type: str = "genomic"

    @model_validator(mode="after")
    def validate_seq_request(self) -> "SequenceRequest":
        """
        Validate that gene_ids is non-empty, does not exceed the batch limit, and
        that sequence_type is one of the accepted Ensembl sequence types.
        """
        if not self.gene_ids:
            raise ValueError("At least one gene ID must be provided.")
        if len(self.gene_ids) > _MAX_SEQUENCE_IDS:
            raise ValueError(f"Maximum {_MAX_SEQUENCE_IDS} gene IDs allowed.")
        if self.sequence_type not in _VALID_SEQ_TYPES:
            raise ValueError(
                f"Invalid sequence_type. Must be one of: {', '.join(sorted(_VALID_SEQ_TYPES))}."
            )
        return self


class SequenceResponse(BaseModel):
    token: str
    fasta: str
    gene_count: int


class AnnotateRequest(BaseModel):
    gene_id: str

    @model_validator(mode="after")
    def validate_annotate_request(self) -> "AnnotateRequest":
        """Run validate_input on gene_id to enforce character and length constraints."""
        validate_input(self.gene_id, "gene_id")
        return self


class SegmentInfo(BaseModel):
    type: str           # "exon" | "intron"
    number: int         # 1-indexed
    genomic_start: int  # absolute chromosome position (inclusive)
    genomic_end: int    # absolute chromosome position (inclusive)
    seq_start: int      # 0-indexed start in the sequence string
    seq_end: int        # 0-indexed exclusive end in the sequence string


class TranscriptAnnotation(BaseModel):
    transcript_id: str
    is_canonical: bool
    segments: list[SegmentInfo]


class AnnotateResponse(BaseModel):
    token: str
    gene_id: str
    display_name: str
    chromosome: str
    gene_start: int
    gene_end: int
    strand: int
    sequence: str
    transcripts: list[TranscriptAnnotation]


class ProteinAnnotateRequest(BaseModel):
    gene_id: str
    species: str
    domains: list[str]


class DomainFeature(BaseModel):
    domain_start: int
    domain_end:   int
    hit_name:     str
    domain_name:  str
    domain_type:  str


class ProteinAnnotateResponse(BaseModel):
    token:       str
    gene_id:     str
    species:     str
    protein_seq: str
    domains:     list[DomainFeature]


class HomologyRequest(BaseModel):
    gene_id:       str
    homology_type: str  # "orthologues" | "homoeologues" | "paralogues"


class HomologyResponse(BaseModel):
    token:           str
    display_columns: list[str]
    all_columns:     list[str]
    rows:            list[dict]
    row_count:       int
    db_name:         str


class PromoterRequest(BaseModel):
    """Request body for the promoter annotation endpoint."""

    gene_id: str
    upstream_bp: int = 2000


class PromoterHit(BaseModel):
    """A single cis-regulatory element hit within a promoter sequence."""

    name: str
    category: str
    function: str
    color_class: str
    start: int        # 0-indexed position in promoter sequence
    end: int          # exclusive
    matched_seq: str
    strand: str       # "+" or "-"


class PromoterResponse(BaseModel):
    """Response body for the promoter annotation endpoint."""

    token: str
    gene_id: str
    display_name: str
    chromosome: str
    gene_start: int
    gene_end: int
    strand: int
    upstream_bp: int
    sequence: str     # promoter sequence, always 5'->3' (distal->TSS)
    hits: list[PromoterHit]


class PrimerRequest(BaseModel):
    variant_name: str
    chromosome: str
    position: int
    allele_string: str
    flanking_bp: int = 200
    num_pairs: int = 5
    primer_type: str = "kasp"   # "kasp" | "pcr"


class PrimerPair(BaseModel):
    rank: int
    left_ref_seq: str
    left_alt_seq: str | None
    right_seq: str
    left_start: int
    left_length: int
    right_start: int
    right_length: int
    product_size: int
    left_tm: float
    right_tm: float
    left_gc: float
    right_gc: float
    left_hairpin_tm: float
    right_hairpin_tm: float
    left_self_any: float
    right_self_any: float
    pair_penalty: float
    left_alt_tm: float | None = None
    left_alt_gc: float | None = None
    left_alt_hairpin_tm: float | None = None
    left_alt_self_any: float | None = None


class PrimerResponse(BaseModel):
    variant_name: str
    chromosome: str
    position: int
    allele_string: str
    flanking_seq: str
    snp_offset: int
    primer_type: str
    primer_pairs: list[PrimerPair]


_HOMO_DISPLAY_COLS = [
    "query_gene_id", "query_species", "query_perc_id",
    "homology_type",
    "homolog_gene_id", "homolog_species", "homolog_perc_id",
]
_HOMO_ALL_COLS = [
    "query_gene_id", "query_species", "query_sequence",
    "query_perc_id", "query_cigar_line",
    "homology_type",
    "homolog_gene_id", "homolog_species",
    "homolog_perc_id", "homolog_cigar_line", "homolog_sequence",
]


_RC = str.maketrans("ACGTacgt", "TGCAtgca")

_IUPAC: dict[str, str] = {
    'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
    'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
    'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]',
}


def _iupac_to_regex(consensus: str) -> str:
    """
    Convert an IUPAC consensus string to a Python regex pattern.

    Replaces IUPAC ambiguity codes with the corresponding character class.
    Standard DNA bases (A, C, G, T) are passed through unchanged.

    Args:
        consensus: IUPAC nucleotide consensus string, e.g. ``"MACGTGYC"``.

    Returns:
        A regex pattern string suitable for ``re.compile()``.
    """
    return ''.join(_IUPAC.get(c.upper(), c) for c in consensus)

PROMOTER_MOTIFS: list[dict] = [
    # Core Promoter
    {"name": "TATA box",        "consensus": "TATAAA",     "category": "Core Promoter",  "function": "Transcription initiation",             "color_class": "bg-yellow-200"},
    {"name": "CAAT box",        "consensus": "CCAAT",      "category": "Core Promoter",  "function": "Enhancer/promoter activity",            "color_class": "bg-yellow-100"},
    # Light-responsive
    {"name": "G-box",           "consensus": "CACGTG",     "category": "Light",          "function": "Light response (HY5/bZIP binding)",     "color_class": "bg-sky-200"},
    {"name": "I-box",           "consensus": "GATAAG",     "category": "Light",          "function": "Light-regulated transcription",          "color_class": "bg-sky-100"},
    {"name": "GATA motif",      "consensus": "GATA",       "category": "Light",          "function": "Light-regulated gene expression",        "color_class": "bg-sky-300"},
    {"name": "Box-4",           "consensus": "ATTAAT",     "category": "Light",          "function": "Light-responsive element",              "color_class": "bg-cyan-200"},
    # ABA / Drought
    {"name": "ABRE",            "consensus": "MACGTGYC",   "category": "ABA/Drought",    "function": "ABA response, drought/osmotic stress",  "color_class": "bg-blue-200"},
    {"name": "DRE/CRT",         "consensus": "RCCGAC",     "category": "Cold/Drought",   "function": "DREB binding, cold/drought tolerance",  "color_class": "bg-indigo-200"},
    # Heat
    {"name": "HSE",             "consensus": "GAANNTTC",   "category": "Heat Stress",    "function": "Heat shock factor binding",             "color_class": "bg-orange-200"},
    # Defense
    {"name": "W-box",           "consensus": "TTGACY",     "category": "Defense",        "function": "WRKY TF binding, defense/stress",       "color_class": "bg-rose-200"},
    # Hormone
    {"name": "TGACG motif",     "consensus": "TGACG",      "category": "JA/SA",          "function": "MeJA and SA responsiveness",            "color_class": "bg-pink-200"},
    {"name": "TCA element",     "consensus": "TCATCTTCTT", "category": "Salicylic Acid", "function": "Salicylic acid response",               "color_class": "bg-fuchsia-200"},
    {"name": "AuxRE",           "consensus": "TGASTC",     "category": "Auxin",          "function": "Auxin response factor binding",         "color_class": "bg-violet-200"},
    {"name": "GCC box",         "consensus": "AGCCGCC",    "category": "Ethylene",       "function": "ERF/AP2 binding, ethylene response",    "color_class": "bg-purple-200"},
    # MYB / Circadian
    {"name": "MBS",             "consensus": "CAACTG",     "category": "MYB",            "function": "MYB TF binding, drought response",      "color_class": "bg-emerald-200"},
    {"name": "Evening element", "consensus": "AAATATCT",   "category": "Circadian",      "function": "Circadian clock regulation",            "color_class": "bg-teal-200"},
]

# Pre-compiled regex patterns for each motif; index-aligned with PROMOTER_MOTIFS.
_PROMOTER_PATTERNS: list[re.Pattern] = [
    re.compile(_iupac_to_regex(m["consensus"]), re.IGNORECASE) for m in PROMOTER_MOTIFS
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    """
    Serve the single-page application entry point.

    Returns:
        The contents of ``static/index.html`` as an HTML response.
    """
    index_path = _STATIC_DIR / "index.html"
    return FileResponse(str(index_path), media_type="text/html")


@app.post("/api/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """
    Run variant and KASP marker queries against the EBI MySQL databases and
    return the joined results.

    The variant query targets the variation DB (``variation_pool``) and the
    marker query targets the core DB (``core_pool``).  When a ``variant_id``
    is supplied both queries run concurrently via ``asyncio.gather``; when
    only ``transcript_id`` is given, the marker query is skipped.

    Strandedness for each unique transcript is fetched from the Ensembl REST
    ``POST /lookup/id`` endpoint and inserted as a column immediately after
    ``feature_stable_id``.

    The result is stored in the in-memory ``_result_cache`` under a UUID token
    that downstream download endpoints use for O(1) retrieval.

    Args:
        request: Validated ``QueryRequest`` containing optional ``variant_id``,
            ``transcript_id``, and ``consequence_types`` filter list.

    Returns:
        ``QueryResponse`` with token, column list, rows, DB name, row count,
        and a boolean indicating whether KASP marker data was found.

    Raises:
        HTTPException(400): If inputs fail validation or no records are found.
        HTTPException(500): On unexpected database errors.
    """
    # Validate and sanitise inputs
    try:
        variant_id = validate_input(request.variant_id, "variant_id")
        transcript_id = validate_input(request.transcript_id, "transcript_id")
        consequence_types: list[str] = []
        for ct in (request.consequence_types or []):
            validated = validate_input(ct, "consequence_types")
            if validated:
                consequence_types.append(validated)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})

    if not variant_id and not transcript_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "At least one search parameter must be provided."
            },
        )

    variation_pool = app.state.variation_pool
    core_pool = app.state.core_pool
    db_name: str = app.state.variation_db_name

    try:
        df, has_markers = await fetch_and_join(
            variation_pool, core_pool, variant_id, transcript_id,
            consequence_types or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": f"Database error: {exc}"},
        )

    # Fetch strandedness from Ensembl REST and insert after feature_stable_id
    unique_ids = df["feature_stable_id"].dropna().unique().tolist()
    strand_map = await _fetch_strandedness(unique_ids)
    insert_pos = df.columns.tolist().index("feature_stable_id") + 1
    df.insert(insert_pos, "Strandedness", df["feature_stable_id"].map(strand_map))

    # Serialise — replace NaN with None for JSON compatibility
    rows: list[dict] = df.where(pd.notna(df), other=None).to_dict(orient="records")
    columns: list[str] = list(df.columns)

    token = str(uuid.uuid4())
    _cache_store(token, rows, columns)

    return QueryResponse(
        token=token,
        columns=columns,
        rows=rows,
        db_name=db_name,
        row_count=len(rows),
        has_markers=has_markers,
    )


@app.get("/api/download/csv/{token}")
async def download_csv(token: str):
    """
    Stream a cached query result as a CSV file download.

    Args:
        token: UUID token previously returned by ``POST /api/query`` or
            ``POST /api/homology``.

    Returns:
        A ``StreamingResponse`` with ``Content-Disposition: attachment`` and
        a filename derived from the cached entry (e.g. ``wheat_variants.csv``
        or ``{gene_id}_homology.csv``).

    Raises:
        HTTPException(404): If the token is not found or has expired.
    """
    entry = _cache_get(token)
    df = pd.DataFrame(entry["data"], columns=entry["columns"])

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={entry['filename']}.csv"
        },
    )


@app.get("/api/download/excel/{token}")
async def download_excel(token: str):
    """
    Stream a cached query result as an Excel (.xlsx) file download.

    The DataFrame is written to an in-memory ``BytesIO`` buffer via
    ``openpyxl`` before streaming, so no temporary file is created on disk.

    Args:
        token: UUID token previously returned by ``POST /api/query`` or
            ``POST /api/homology``.

    Returns:
        A ``StreamingResponse`` with MIME type
        ``application/vnd.openxmlformats-officedocument.spreadsheetml.sheet``
        and a filename derived from the cached entry (e.g. ``wheat_variants.xlsx``).

    Raises:
        HTTPException(404): If the token is not found or has expired.
    """
    entry = _cache_get(token)
    df = pd.DataFrame(entry["data"], columns=entry["columns"])

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Variants")
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename={entry['filename']}.xlsx"
        },
    )


@app.post("/api/sequences", response_model=SequenceResponse)
async def fetch_sequences(request: SequenceRequest):
    """
    Fetch sequences for one or more Ensembl gene IDs from the Ensembl REST API
    and return the result as FASTA text.

    Each gene ID is validated individually via ``validate_input`` before the
    batch request is sent.  The FASTA string is stored in ``_fasta_cache`` under
    a UUID token for subsequent FASTA download requests.

    Args:
        request: Validated ``SequenceRequest`` containing a list of gene IDs
            (max 50) and a ``sequence_type`` (one of ``genomic``, ``cdna``,
            ``cds``, ``protein``).

    Returns:
        ``SequenceResponse`` with the UUID token, raw FASTA text, and the count
        of gene IDs successfully submitted.

    Raises:
        HTTPException(400): If no valid gene IDs remain after filtering, or if
            Ensembl rejects the request.
        HTTPException(404): If one or more gene IDs are not found in Ensembl.
        HTTPException(502): On unexpected Ensembl API errors.
        HTTPException(504): On Ensembl REST API timeout.
    """
    # Validate each gene ID individually
    validated_ids: list[str] = []
    for i, raw_id in enumerate(request.gene_ids):
        if not raw_id.strip():
            continue
        try:
            vid = validate_input(raw_id.strip(), f"gene_ids[{i}]")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)})
        if vid:
            validated_ids.append(vid)

    if not validated_ids:
        raise HTTPException(
            status_code=400,
            detail={"error": "No valid gene IDs provided after filtering blanks."},
        )

    sequence_type = request.sequence_type

    try:
        async with httpx.AsyncClient(timeout=_ENSEMBL_TIMEOUT_SECS) as client:
            resp = await client.post(
                f"{ENSEMBL_REST_BASE}/sequence/id",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/x-fasta",
                },
                json={"ids": validated_ids, "type": sequence_type},
            )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={"error": "Ensembl REST API timed out. Please try again."},
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": f"Could not reach Ensembl REST API: {exc}"},
        )

    if resp.status_code == 400:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Ensembl rejected the request: {resp.text[:300]}"},
        )
    if resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail={"error": "One or more gene IDs were not found in Ensembl."},
        )
    if not resp.is_success:
        raise HTTPException(
            status_code=502,
            detail={"error": f"Ensembl returned status {resp.status_code}: {resp.text[:300]}"},
        )

    fasta_text = resp.text
    token = str(uuid.uuid4())
    _fasta_cache_store(token, fasta_text, len(validated_ids))

    return SequenceResponse(token=token, fasta=fasta_text, gene_count=len(validated_ids))


@app.get("/api/download/fasta/{token}")
async def download_fasta(token: str):
    """
    Stream a cached FASTA result as a plain-text file download.

    Args:
        token: UUID token previously returned by ``POST /api/sequences``.

    Returns:
        A ``StreamingResponse`` with ``Content-Disposition: attachment`` and
        filename ``sequences.fasta``.

    Raises:
        HTTPException(404): If the token is not found or has expired.
    """
    entry = _fasta_cache_get(token)
    return StreamingResponse(
        iter([entry["fasta"]]),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=sequences.fasta"},
    )


# ---------------------------------------------------------------------------
# Annotate helpers
# ---------------------------------------------------------------------------

def _compute_segments(
    gene_start: int,
    gene_end: int,
    strand: int,
    exons: list[dict],
) -> list[SegmentInfo]:
    """
    Map a transcript's exon genomic coordinates onto the flat genomic sequence
    string, producing an ordered list of exon and intron ``SegmentInfo`` objects.

    Exons are sorted in transcript order (ascending for the forward strand,
    descending for the reverse strand).  Any gap between consecutive exons
    becomes an intron segment.  A leading gap before the first exon or a
    trailing gap after the last exon is treated as a flanking intron/UTR region.

    Sequence positions (``seq_start``, ``seq_end``) are 0-indexed, half-open
    intervals into the genomic sequence string returned by Ensembl.

    Args:
        gene_start: Absolute chromosomal start position of the gene (1-indexed,
            inclusive), as reported by the Ensembl lookup endpoint.
        gene_end: Absolute chromosomal end position of the gene (1-indexed,
            inclusive).
        strand: Strand indicator; ``1`` for forward, ``-1`` for reverse.
        exons: List of exon dicts, each containing ``start`` and ``end`` keys
            with absolute chromosomal positions.

    Returns:
        List of ``SegmentInfo`` objects in 5′→3′ transcript order, alternating
        between exon and intron entries as appropriate.
    """
    gene_len = gene_end - gene_start + 1
    # Transcript order: ascending for +1 strand, descending for -1 (rev-complement)
    ordered = sorted(exons, key=lambda e: e["start"], reverse=(strand == -1))

    def to_seq_range(gstart: int, gend: int) -> tuple[int, int]:
        if strand == 1:
            return gstart - gene_start, gend - gene_start + 1
        else:
            return gene_end - gend, gene_end - gstart + 1

    segments: list[SegmentInfo] = []
    seq_pos = 0
    intron_num = 0

    for i, exon in enumerate(ordered):
        s, e = to_seq_range(exon["start"], exon["end"])
        if s > seq_pos:   # gap before this exon → intron / UTR region
            intron_num += 1
            segments.append(SegmentInfo(
                type="intron",
                number=intron_num,
                genomic_start=gene_start + seq_pos if strand == 1 else gene_end - s + 1,
                genomic_end=gene_start + s - 1 if strand == 1 else gene_end - seq_pos,
                seq_start=seq_pos,
                seq_end=s,
            ))
        segments.append(SegmentInfo(
            type="exon",
            number=i + 1,
            genomic_start=exon["start"],
            genomic_end=exon["end"],
            seq_start=s,
            seq_end=e,
        ))
        seq_pos = e

    if seq_pos < gene_len:   # trailing region after last exon
        intron_num += 1
        segments.append(SegmentInfo(
            type="intron",
            number=intron_num,
            genomic_start=gene_start + seq_pos if strand == 1 else gene_end - gene_len + 1,
            genomic_end=gene_end if strand == 1 else gene_end - seq_pos,
            seq_start=seq_pos,
            seq_end=gene_len,
        ))

    return segments


@app.post("/api/annotate", response_model=AnnotateResponse)
async def annotate_gene(request: AnnotateRequest):
    """
    Fetch gene metadata and genomic sequence from the Ensembl REST API and
    compute exon/intron segment coordinates for every transcript.

    Two concurrent requests are made to Ensembl:
    - ``GET /lookup/id/{gene_id}?expand=1`` — gene metadata including all
      transcript and exon objects.
    - ``GET /sequence/id/{gene_id}?type=genomic`` — full genomic sequence.

    For each transcript, ``_compute_segments`` converts absolute exon
    coordinates into 0-indexed positions within the returned sequence string.
    The canonical transcript is sorted first in the response; remaining
    transcripts are ordered alphabetically by stable ID.

    The result is cached in ``_annotate_cache`` under a UUID token for
    potential future retrieval.

    Args:
        request: Validated ``AnnotateRequest`` containing the Ensembl gene ID
            (e.g. ``TRAESCS3D02G273600``).

    Returns:
        ``AnnotateResponse`` with gene metadata, the full genomic sequence, and
        a list of ``TranscriptAnnotation`` objects each containing ordered
        ``SegmentInfo`` entries.

    Raises:
        HTTPException(400): If Ensembl reports an invalid gene ID.
        HTTPException(404): If the gene ID or its sequence is not found in Ensembl.
        HTTPException(502): On unexpected Ensembl API errors.
        HTTPException(504): On Ensembl REST API timeout.
    """
    gene_id = request.gene_id.strip()

    try:
        async with httpx.AsyncClient(timeout=_ENSEMBL_TIMEOUT_SECS) as client:
            ann_resp, seq_resp = await asyncio.gather(
                client.get(
                    f"{ENSEMBL_REST_BASE}/lookup/id/{gene_id}",
                    params={"expand": "1"},
                    headers={"Accept": "application/json"},
                ),
                client.get(
                    f"{ENSEMBL_REST_BASE}/sequence/id/{gene_id}",
                    params={"type": "genomic"},
                    headers={"Accept": "application/json"},
                ),
            )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail={"error": "Ensembl REST API timed out. Please try again."},
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": f"Could not reach Ensembl REST API: {exc}"},
        )

    if ann_resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Gene ID '{gene_id}' not found in Ensembl."},
        )
    if ann_resp.status_code == 400:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Invalid gene ID: {ann_resp.text[:300]}"},
        )
    if not ann_resp.is_success:
        raise HTTPException(
            status_code=502,
            detail={"error": f"Ensembl annotation error {ann_resp.status_code}: {ann_resp.text[:300]}"},
        )
    if seq_resp.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Sequence for '{gene_id}' not found in Ensembl."},
        )
    if not seq_resp.is_success:
        raise HTTPException(
            status_code=502,
            detail={"error": f"Ensembl sequence error {seq_resp.status_code}: {seq_resp.text[:300]}"},
        )

    ann_data = ann_resp.json()
    seq_data = seq_resp.json()
    sequence: str = seq_data.get("seq", "")

    gene_start: int = ann_data["start"]
    gene_end: int = ann_data["end"]
    strand: int = ann_data["strand"]
    chromosome: str = ann_data.get("seq_region_name", "")
    display_name: str = ann_data.get("display_name", gene_id)

    transcript_annotations: list[TranscriptAnnotation] = []
    for tx in ann_data.get("Transcript", []):
        tx_id: str = tx.get("id", "")
        is_canonical: bool = bool(tx.get("is_canonical", 0))
        exons = [{"start": e["start"], "end": e["end"]} for e in tx.get("Exon", [])]
        if not exons:
            continue
        segments = _compute_segments(gene_start, gene_end, strand, exons)
        transcript_annotations.append(TranscriptAnnotation(
            transcript_id=tx_id,
            is_canonical=is_canonical,
            segments=segments,
        ))

    # Canonical transcript first, then alphabetical by ID
    transcript_annotations.sort(key=lambda t: (not t.is_canonical, t.transcript_id))

    token = str(uuid.uuid4())
    result_data = {
        "token": token,
        "gene_id": gene_id,
        "display_name": display_name,
        "chromosome": chromosome,
        "gene_start": gene_start,
        "gene_end": gene_end,
        "strand": strand,
        "sequence": sequence,
        "transcripts": [t.model_dump() for t in transcript_annotations],
    }
    _annotate_cache_store(token, result_data)

    return AnnotateResponse(**result_data)


@app.post("/api/annotate_protein", response_model=ProteinAnnotateResponse)
async def annotate_protein(request: ProteinAnnotateRequest):
    """
    Fetch protein domain features from the EBI core database and the protein
    sequence from the Ensembl REST API for a given gene.

    The workflow proceeds as follows:
    1. ``gene_id``, ``species``, and ``domains`` are validated; species must be
       a key of ``SUPPORTED_SPECIES`` and every domain name must be a key of
       ``DOMAIN_PREFIXES``.
    2. A species-specific core DB pool is obtained (or lazily created) via
       ``_get_species_core_pool``.
    3. Two concurrent DB calls fetch the canonical transcript stable ID and all
       matching protein domain features for the gene.
    4. The canonical transcript ID is used to request the protein sequence from
       ``GET /sequence/id/{transcript_id}?type=protein``.
    5. The result is cached in ``_prot_annotate_cache`` and returned.

    Args:
        request: ``ProteinAnnotateRequest`` containing ``gene_id``, ``species``
            (Ensembl species key), and ``domains`` (list of domain database
            names to include, e.g. ``["Pfam", "Panther"]``).

    Returns:
        ``ProteinAnnotateResponse`` with a UUID token, gene ID, species,
        protein sequence, and list of ``DomainFeature`` objects.

    Raises:
        HTTPException(400): If gene_id, species, or domain names fail validation.
        HTTPException(404): If the gene is not found in the core DB or Ensembl.
        HTTPException(500): On unexpected database errors.
        HTTPException(502): On unexpected Ensembl API errors.
        HTTPException(504): On Ensembl REST API timeout.
    """
    # Validate gene_id
    try:
        gene_id = validate_input(request.gene_id.strip(), "gene_id")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})
    if not gene_id:
        raise HTTPException(status_code=400, detail={"error": "gene_id is required."})

    # Validate species
    species = request.species
    if species not in SUPPORTED_SPECIES:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unsupported species '{species}'. Supported: {list(SUPPORTED_SPECIES.keys())}"},
        )

    # Validate domains
    if not request.domains:
        raise HTTPException(status_code=400, detail={"error": "At least one domain type must be selected."})
    unknown = [d for d in request.domains if d not in DOMAIN_PREFIXES]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail={"error": f"Unknown domain type(s): {unknown}. Valid: {list(DOMAIN_PREFIXES.keys())}"},
        )

    # Step 1: resolve canonical transcript ID + domain features from DB concurrently
    try:
        pool = await _get_species_core_pool(species)
        canonical_tx_id, domain_rows = await asyncio.gather(
            fetch_canonical_transcript_id(pool, gene_id),
            fetch_protein_domains(pool, gene_id, request.domains),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"Database error: {exc}"})

    if not canonical_tx_id:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Gene ID '{gene_id}' not found in the {SUPPORTED_SPECIES[species]} core database."},
        )

    # Step 2: fetch protein sequence for the canonical transcript from Ensembl
    try:
        async with httpx.AsyncClient(timeout=_ENSEMBL_TIMEOUT_SECS) as client:
            seq_resp = await client.get(
                f"{ENSEMBL_REST_BASE}/sequence/id/{canonical_tx_id}",
                params={"type": "protein"},
                headers={"Accept": "application/json"},
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail={"error": "Ensembl REST API timed out. Please try again."})
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail={"error": f"Could not reach Ensembl REST API: {exc}"})

    if seq_resp.status_code == 404:
        raise HTTPException(status_code=404, detail={"error": f"Gene ID '{gene_id}' not found in Ensembl."})
    if not seq_resp.is_success:
        raise HTTPException(
            status_code=502,
            detail={"error": f"Ensembl sequence error {seq_resp.status_code}: {seq_resp.text[:300]}"},
        )

    protein_seq: str = seq_resp.json().get("seq", "")

    domain_features = [
        DomainFeature(
            domain_start=row["domain_start"],
            domain_end=row["domain_end"],
            hit_name=row["hit_name"],
            domain_name=row["domain_name"],
            domain_type=row["domain_type"],
        )
        for row in domain_rows
    ]

    token = str(uuid.uuid4())
    result_data = {
        "token":       token,
        "gene_id":     gene_id,
        "species":     species,
        "protein_seq": protein_seq,
        "domains":     [d.model_dump() for d in domain_features],
    }
    _prot_cache_store(token, result_data)

    return ProteinAnnotateResponse(**result_data)


@app.post("/api/homology", response_model=HomologyResponse)
async def fetch_homology(request: HomologyRequest):
    """
    Query the Ensembl Compara database for homologous genes of the given type
    and return pairwise alignment data.

    ``homology_type`` must be one of ``"orthologues"``, ``"homoeologues"``, or
    ``"paralogues"`` (mapped to the underlying Compara ``description`` values in
    ``HOMOLOGY_TYPES``).

    Bytes values in sequence and CIGAR-line columns are decoded to UTF-8 strings
    to handle the latin1 charset edge case returned by the EBI MySQL server.

    The result (11 columns including sequences and CIGAR lines) is stored in
    ``_result_cache`` under a UUID token so the shared download endpoints can
    serve CSV and Excel exports without re-querying.

    Args:
        request: ``HomologyRequest`` containing ``gene_id`` and ``homology_type``.

    Returns:
        ``HomologyResponse`` with a UUID token, 7-column ``display_columns`` list
        for the frontend table, 11-column ``all_columns`` list for exports,
        the row data, row count, and the Compara DB name.

    Raises:
        HTTPException(400): If gene_id fails validation or homology_type is invalid.
        HTTPException(404): If no homologues are found for the given gene.
        HTTPException(500): On unexpected database errors.
    """
    try:
        gene_id = validate_input(request.gene_id.strip(), "gene_id")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})
    if not gene_id:
        raise HTTPException(status_code=400, detail={"error": "gene_id is required."})
    if request.homology_type not in HOMOLOGY_TYPES:
        raise HTTPException(status_code=400, detail={"error": "Invalid homology_type."})

    try:
        rows = await fetch_homologs(app.state.compara_pool, gene_id, request.homology_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"Database error: {exc}"})

    if not rows:
        raise HTTPException(
            status_code=404,
            detail={"error": f"No {request.homology_type} found for gene '{gene_id}'."},
        )

    # Decode bytes for TEXT/BLOB columns (latin1 charset edge case)
    for row in rows:
        for field in ("query_sequence", "homolog_sequence",
                      "query_cigar_line", "homolog_cigar_line"):
            if isinstance(row.get(field), (bytes, bytearray)):
                row[field] = row[field].decode("utf-8", errors="replace")

    token = str(uuid.uuid4())
    _cache_store(token, rows, _HOMO_ALL_COLS, filename=f"{gene_id}_homology")

    return HomologyResponse(
        token=token,
        display_columns=_HOMO_DISPLAY_COLS,
        all_columns=_HOMO_ALL_COLS,
        rows=rows,
        row_count=len(rows),
        db_name=app.state.compara_db_name,
    )


@app.post("/api/primers", response_model=PrimerResponse)
async def design_primers(req: PrimerRequest):
    """
    Design KASP or PCR primers flanking a given SNP using Primer3.

    The flanking genomic sequence is fetched from the Ensembl REST API for the
    region ``[position - flanking_bp, position + flanking_bp]``.  Primer3 is
    invoked via ``asyncio.to_thread`` to avoid blocking the event loop.

    **KASP mode** (``primer_type="kasp"``)
        ``SEQUENCE_FORCE_LEFT_END`` pins the 3′ end of the left primer to the
        SNP offset.  After design, the last base of the designed sequence is
        replaced by the REF and ALT alleles to produce ``left_ref_seq`` and
        ``left_alt_seq`` respectively.  Thermodynamic properties for the ALT
        primer are independently computed via ``primer3.bindings.calcTm``,
        ``calcHairpin``, and ``calcHomodimer``.

    **PCR mode** (``primer_type="pcr"``)
        ``SEQUENCE_TARGET`` requires that the designed primers flank the SNP
        position.  Only ``left_ref_seq`` is populated; ``left_alt_seq`` and its
        associated thermodynamic fields are ``None``.

    ``flanking_bp`` is clamped to the range [50, 500]; ``num_pairs`` is clamped
    to [1, 10].

    Args:
        req: ``PrimerRequest`` containing ``variant_name``, ``chromosome``,
            ``position`` (1-based), ``allele_string`` (e.g. ``"A/T"``),
            ``flanking_bp``, ``num_pairs``, and ``primer_type``.

    Returns:
        ``PrimerResponse`` with the flanking sequence, SNP offset, primer type,
        and a list of ``PrimerPair`` objects ranked by Primer3 penalty score.

    Raises:
        HTTPException(400): If variant_name, chromosome, or primer_type fails
            validation.
        HTTPException(422): If Primer3 cannot design any primers for the region.
        HTTPException(502): On unexpected Ensembl API errors.
        HTTPException(504): On Ensembl REST API timeout.
    """
    import primer3
    import warnings

    try:
        validate_input(req.variant_name, "variant_name")
        validate_input(req.chromosome, "chromosome")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)})

    if req.primer_type not in {"kasp", "pcr"}:
        raise HTTPException(status_code=400, detail={"error": "primer_type must be 'kasp' or 'pcr'"})

    flanking_bp = max(50, min(500, req.flanking_bp))
    num_pairs   = max(1, min(10, req.num_pairs))
    start = max(1, req.position - flanking_bp)
    end   = req.position + flanking_bp

    # Fetch flanking sequence from Ensembl REST
    region = f"{req.chromosome}:{start}..{end}"
    try:
        async with httpx.AsyncClient(timeout=_ENSEMBL_TIMEOUT_SECS) as client:
            resp = await client.get(
                f"{ENSEMBL_REST_BASE}/sequence/region/triticum_aestivum/{region}",
                headers={"Accept": "application/json"},
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail={"error": "Ensembl REST API timed out."})
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail={"error": f"Could not reach Ensembl REST API: {exc}"})

    if not resp.is_success:
        raise HTTPException(status_code=502, detail={"error": f"Failed to fetch flanking sequence from Ensembl (status {resp.status_code})"})

    flanking_seq = resp.json()["seq"].upper()
    snp_offset = req.position - start   # 0-indexed SNP position in flanking_seq

    # Run primer3 (synchronous — offload to thread)
    alleles = req.allele_string.split("/")
    ref = alleles[0]
    alt = alleles[1] if len(alleles) > 1 else alleles[0]

    if req.primer_type == "kasp":
        seq_args = {
            "SEQUENCE_ID": req.variant_name,
            "SEQUENCE_TEMPLATE": flanking_seq,
            "SEQUENCE_FORCE_LEFT_END": snp_offset,
        }
    else:
        seq_args = {
            "SEQUENCE_ID": req.variant_name,
            "SEQUENCE_TEMPLATE": flanking_seq,
            "SEQUENCE_TARGET": [[snp_offset, 1]],
        }

    global_args = {
        "PRIMER_OPT_SIZE": 20, "PRIMER_MIN_SIZE": 18, "PRIMER_MAX_SIZE": 25,
        "PRIMER_OPT_TM": 60.0, "PRIMER_MIN_TM": 57.0, "PRIMER_MAX_TM": 63.0,
        "PRIMER_MIN_GC": 20.0, "PRIMER_MAX_GC": 80.0,
        "PRIMER_PRODUCT_SIZE_RANGE": [[75, 300]],
        "PRIMER_NUM_RETURN": num_pairs,
        "PRIMER_THERMODYNAMIC_OLIGO_ALIGNMENT": 1,
    }

    p3 = await asyncio.to_thread(primer3.bindings.design_primers, seq_args, global_args)

    num_returned = p3.get("PRIMER_PAIR_NUM_RETURNED", 0)
    if num_returned == 0:
        raise HTTPException(status_code=422, detail={"error": "Primer3 could not design primers for this region."})

    pairs = []
    for i in range(num_returned):
        left_seq = p3[f"PRIMER_LEFT_{i}_SEQUENCE"]
        if req.primer_type == "kasp":
            left_ref = left_seq[:-1] + ref
            left_alt = left_seq[:-1] + alt
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _alt_tm = round(primer3.bindings.calcTm(left_alt), 2)
                _alt_gc = round((left_alt.upper().count('G') + left_alt.upper().count('C')) / len(left_alt) * 100, 1)
                _alt_hairpin_tm = round(max(primer3.bindings.calcHairpin(left_alt).tm, 0.0), 2)
                _alt_self_any = round(max(primer3.bindings.calcHomodimer(left_alt).tm, 0.0), 2)
        else:
            left_ref = left_seq
            left_alt = None
            _alt_tm = _alt_gc = _alt_hairpin_tm = _alt_self_any = None

        l_start, l_len = p3[f"PRIMER_LEFT_{i}"]
        r_start, r_len = p3[f"PRIMER_RIGHT_{i}"]

        pairs.append(PrimerPair(
            rank=i + 1,
            left_ref_seq=left_ref,
            left_alt_seq=left_alt,
            right_seq=p3[f"PRIMER_RIGHT_{i}_SEQUENCE"],
            left_start=l_start, left_length=l_len,
            right_start=r_start, right_length=r_len,
            product_size=p3[f"PRIMER_PAIR_{i}_PRODUCT_SIZE"],
            left_tm=round(p3[f"PRIMER_LEFT_{i}_TM"], 2),
            right_tm=round(p3[f"PRIMER_RIGHT_{i}_TM"], 2),
            left_gc=round(p3[f"PRIMER_LEFT_{i}_GC_PERCENT"], 1),
            right_gc=round(p3[f"PRIMER_RIGHT_{i}_GC_PERCENT"], 1),
            left_hairpin_tm=round(p3.get(f"PRIMER_LEFT_{i}_HAIRPIN_TH", 0.0), 2),
            right_hairpin_tm=round(p3.get(f"PRIMER_RIGHT_{i}_HAIRPIN_TH", 0.0), 2),
            left_self_any=round(p3.get(f"PRIMER_LEFT_{i}_SELF_ANY_TH", 0.0), 2),
            right_self_any=round(p3.get(f"PRIMER_RIGHT_{i}_SELF_ANY_TH", 0.0), 2),
            pair_penalty=round(p3[f"PRIMER_PAIR_{i}_PENALTY"], 4),
            left_alt_tm=_alt_tm,
            left_alt_gc=_alt_gc,
            left_alt_hairpin_tm=_alt_hairpin_tm,
            left_alt_self_any=_alt_self_any,
        ))

    return PrimerResponse(
        variant_name=req.variant_name,
        chromosome=req.chromosome,
        position=req.position,
        allele_string=req.allele_string,
        flanking_seq=flanking_seq,
        snp_offset=snp_offset,
        primer_type=req.primer_type,
        primer_pairs=pairs,
    )


def _scan_promoter_motifs(sequence: str) -> list[dict]:
    """
    Scan a promoter sequence for all known cis-regulatory elements on both strands.

    Searches both the forward strand and its reverse complement for each motif
    in ``PROMOTER_MOTIFS``, using compiled IUPAC regex patterns.  Reverse-strand
    hit coordinates are mapped back to the forward-strand coordinate space so
    that all ``start``/``end`` values are relative to the same sequence index 0.

    Args:
        sequence: Promoter nucleotide sequence (5'→3', distal to TSS).

    Returns:
        A list of hit dictionaries (keys: ``name``, ``category``, ``function``,
        ``color_class``, ``start``, ``end``, ``matched_seq``, ``strand``),
        sorted ascending by ``start`` position.
    """
    rev_comp = sequence[::-1].translate(_RC)
    hits = []
    for motif, pattern in zip(PROMOTER_MOTIFS, _PROMOTER_PATTERNS):
        for strand, seq in (("+", sequence), ("-", rev_comp)):
            for m in pattern.finditer(seq):
                start = m.start() if strand == "+" else len(sequence) - m.end()
                end   = m.end()   if strand == "+" else len(sequence) - m.start()
                hits.append({
                    "name": motif["name"], "category": motif["category"],
                    "function": motif["function"], "color_class": motif["color_class"],
                    "start": start, "end": end,
                    "matched_seq": m.group(), "strand": strand,
                })
    return sorted(hits, key=lambda h: h["start"])


@app.post("/api/promoter", response_model=PromoterResponse)
async def promoter_annotate(req: PromoterRequest):
    """
    Fetch a configurable upstream promoter window and scan it for cis-regulatory elements.

    Performs two sequential Ensembl REST calls: a gene lookup to obtain chromosomal
    coordinates and strand, followed by a sequence fetch for the upstream region.
    The promoter sequence is always returned 5'→3' (distal end at index 0, TSS at the
    last position) regardless of gene strand.  The sequence is then scanned on both
    strands for all motifs in ``PROMOTER_MOTIFS`` using ``_scan_promoter_motifs()``.

    Args:
        req: ``PromoterRequest`` containing the Ensembl gene ID and desired upstream
             window size in base pairs (clamped to 200–5000).

    Returns:
        A ``PromoterResponse`` with gene metadata, the promoter sequence, and all
        cis-regulatory element hits, stored in ``_promoter_cache`` under a UUID token.

    Raises:
        HTTPException: 400 if ``gene_id`` fails validation or is empty.
        HTTPException: 400 if the Ensembl REST lookup returns a non-2xx status for
                       the gene ID.
        HTTPException: 502 if the Ensembl REST sequence endpoint returns a non-2xx
                       status.
    """
    try:
        gene_id = validate_input(req.gene_id, "gene_id")
    except ValueError as exc:
        raise HTTPException(400, {"error": str(exc)})
    if not gene_id:
        raise HTTPException(400, {"error": "gene_id is required."})
    upstream_bp = max(200, min(5000, req.upstream_bp))

    async with httpx.AsyncClient(timeout=_ENSEMBL_TIMEOUT_SECS) as client:
        lookup_resp = await client.get(
            f"{ENSEMBL_REST_BASE}/lookup/id/{gene_id}",
            params={"expand": "0"},
            headers={"Accept": "application/json"},
        )
        if lookup_resp.status_code == 400 or lookup_resp.status_code == 404:
            raise HTTPException(400, {"error": f"Gene ID '{gene_id}' not found in Ensembl."})
        if lookup_resp.status_code != 200:
            raise HTTPException(502, {"error": "Ensembl REST lookup failed."})
        gene_data = lookup_resp.json()
        chrom     = gene_data["seq_region_name"]
        gstart    = gene_data["start"]
        gend      = gene_data["end"]
        strand    = gene_data["strand"]
        species   = gene_data["species"]
        dname     = gene_data.get("display_name") or gene_id

        if strand == 1:
            reg_start, reg_end = gstart - upstream_bp, gstart - 1
        else:
            reg_start, reg_end = gend + 1, gend + upstream_bp
        reg_start = max(1, reg_start)

        seq_resp = await client.get(
            f"{ENSEMBL_REST_BASE}/sequence/region/{species}/{chrom}:{reg_start}..{reg_end}",
            headers={"Accept": "application/json"},
        )
        if seq_resp.status_code != 200:
            raise HTTPException(502, {"error": "Ensembl REST sequence fetch failed."})
        sequence = seq_resp.json()["seq"]

        if strand == -1:
            sequence = sequence[::-1].translate(_RC)

    hits = _scan_promoter_motifs(sequence)
    token = str(uuid.uuid4())
    result = PromoterResponse(
        token=token, gene_id=gene_id, display_name=dname,
        chromosome=chrom, gene_start=gstart, gene_end=gend,
        strand=strand, upstream_bp=upstream_bp,
        sequence=sequence, hits=[PromoterHit(**h) for h in hits],
    )
    _promoter_cache_store(token, result.model_dump())
    return result
