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
    _CORE_DB_PREFIX,
    _VARIATION_DB_PREFIX,
    DOMAIN_PREFIXES,
    SUPPORTED_SPECIES,
    create_pool,
    discover_latest_db,
    fetch_and_join,
    fetch_canonical_transcript_id,
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

# {token: {"data": list[dict], "columns": list[str], "ts": float}}
_result_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _cache_store(token: str, data: list[dict], columns: list[str]) -> None:
    """Insert a result into the cache, evicting the oldest entry if full."""
    if len(_result_cache) >= _CACHE_MAX_ENTRIES:
        _result_cache.popitem(last=False)
    _result_cache[token] = {"data": data, "columns": columns, "ts": time.time()}


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
    if len(_fasta_cache) >= _CACHE_MAX_ENTRIES:
        _fasta_cache.popitem(last=False)
    _fasta_cache[token] = {"fasta": fasta, "gene_count": gene_count, "ts": time.time()}


def _fasta_cache_get(token: str) -> dict[str, Any]:
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
    if len(_annotate_cache) >= _CACHE_MAX_ENTRIES:
        _annotate_cache.popitem(last=False)
    _annotate_cache[token] = {"data": data, "ts": time.time()}


# ---------------------------------------------------------------------------
# Protein annotation cache
# ---------------------------------------------------------------------------

_prot_annotate_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _prot_cache_store(token: str, data: dict) -> None:
    if len(_prot_annotate_cache) >= _CACHE_MAX_ENTRIES:
        _prot_annotate_cache.popitem(last=False)
    _prot_annotate_cache[token] = {"data": data, "ts": time.time()}


# ---------------------------------------------------------------------------
# Per-species core pool manager
# ---------------------------------------------------------------------------

_species_core_pools: dict[str, Any] = {}


async def _get_species_core_pool(species: str):
    if species not in _species_core_pools:
        db_name = await discover_latest_db(f"{species}_core_")
        _species_core_pools[species] = await create_pool(db_name)
    return _species_core_pools[species]


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Discover the latest variation + core DBs and create two connection pools."""
    import asyncio as _asyncio
    variation_db, core_db = await _asyncio.gather(
        discover_latest_db(_VARIATION_DB_PREFIX),
        discover_latest_db(_CORE_DB_PREFIX),
    )
    variation_pool, core_pool = await _asyncio.gather(
        create_pool(variation_db),
        create_pool(core_db),
    )
    app.state.variation_pool = variation_pool
    app.state.core_pool = core_pool
    app.state.variation_db_name = variation_db
    app.state.core_db_name = core_db
    print(f"[startup] variation DB : {variation_db}")
    print(f"[startup] core DB      : {core_db}")
    yield
    variation_pool.close()
    core_pool.close()
    for pool in _species_core_pools.values():
        pool.close()
    await _asyncio.gather(
        variation_pool.wait_closed(),
        core_pool.wait_closed(),
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    index_path = _STATIC_DIR / "index.html"
    return FileResponse(str(index_path), media_type="text/html")


@app.post("/api/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
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
    entry = _cache_get(token)
    df = pd.DataFrame(entry["data"], columns=entry["columns"])

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=wheat_variants.csv"
        },
    )


@app.get("/api/download/excel/{token}")
async def download_excel(token: str):
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
            "Content-Disposition": "attachment; filename=wheat_variants.xlsx"
        },
    )


@app.post("/api/sequences", response_model=SequenceResponse)
async def fetch_sequences(request: SequenceRequest):
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
    """Map exon genomic coordinates onto a flat sequence string, inserting intron segments for gaps."""
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
