"""
main.py — FastAPI application for the Wheat Variation Explorer Dashboard.

Endpoints:
  GET  /                           → serves static/index.html
  POST /api/query                  → runs DB queries, caches result, returns JSON
  GET  /api/download/csv/{token}   → streams CSV from in-memory cache
  GET  /api/download/excel/{token} → streams Excel from in-memory cache
"""

import io
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, model_validator

from database import (
    _CORE_DB_PREFIX,
    _VARIATION_DB_PREFIX,
    create_pool,
    discover_latest_db,
    fetch_and_join,
    validate_input,
)


# ---------------------------------------------------------------------------
# In-memory result cache
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 30 * 60   # 30 minutes
_CACHE_MAX_ENTRIES = 100

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
    await _asyncio.gather(
        variation_pool.wait_closed(),
        core_pool.wait_closed(),
    )
    print("[shutdown] Both connection pools closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Wheat Variation Explorer", lifespan=lifespan)

_STATIC_DIR = Path(__file__).parent / "static"


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
