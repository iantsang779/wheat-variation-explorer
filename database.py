"""
database.py — DB discovery, connection pool, queries, and pandas join logic
for the Wheat Variation Explorer Dashboard.
"""

import asyncio
import re
import aiomysql
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EBI_HOST = "mysql-eg-publicsql.ebi.ac.uk"
EBI_PORT = 4157
EBI_USER = "ensro"
EBI_PASSWORD = ""

_VARIATION_DB_PREFIX = "triticum_aestivum_variation_"
_CORE_DB_PREFIX = "triticum_aestivum_core_"
_INPUT_RE = re.compile(r"^[\w.\-]+$")
_MAX_INPUT_LEN = 100


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input(value: str | None, field_name: str) -> str | None:
    """Strip whitespace, enforce length and character constraints."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > _MAX_INPUT_LEN:
        raise ValueError(
            f"{field_name} exceeds maximum length of {_MAX_INPUT_LEN} characters."
        )
    if not _INPUT_RE.match(value):
        raise ValueError(
            f"{field_name} contains invalid characters. "
            "Only alphanumeric characters, dots, hyphens, and underscores are allowed."
        )
    return value


# ---------------------------------------------------------------------------
# DB discovery
# ---------------------------------------------------------------------------

def _parse_version(db_name: str, prefix: str) -> tuple[int, ...]:
    """
    Extract numeric version tuple from a DB name, e.g.
    'triticum_aestivum_variation_110_1' → (110, 1).
    Handles arbitrary depth.
    """
    suffix = db_name[len(prefix):]
    parts = suffix.split("_")
    try:
        return tuple(int(p) for p in parts if p.isdigit())
    except ValueError:
        return (0,)


async def discover_latest_db(prefix: str) -> str:
    """
    Connect to EBI MySQL, list databases matching *prefix*, and return
    the name of the most recent one (highest numeric version suffix).
    """
    conn = await aiomysql.connect(
        host=EBI_HOST,
        port=EBI_PORT,
        user=EBI_USER,
        password=EBI_PASSWORD,
        autocommit=True,
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"SHOW DATABASES LIKE '{prefix}%'")
            rows = await cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise RuntimeError(
            f"No databases matching '{prefix}*' found on the EBI server."
        )

    db_names = [row[0] for row in rows]
    latest = max(db_names, key=lambda n: _parse_version(n, prefix))
    return latest


# ---------------------------------------------------------------------------
# Pool creation
# ---------------------------------------------------------------------------

async def create_pool(db_name: str) -> aiomysql.Pool:
    """Create and return an aiomysql connection pool for the given DB."""
    pool = await aiomysql.create_pool(
        host=EBI_HOST,
        port=EBI_PORT,
        user=EBI_USER,
        password=EBI_PASSWORD,
        db=db_name,
        minsize=2,
        maxsize=10,
        autocommit=True,
        pool_recycle=3600,
    )
    return pool


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

async def _run_query(pool: aiomysql.Pool, sql: str, params: tuple) -> list[dict]:
    """Execute a parameterised query and return rows as a list of dicts."""
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()


_VARIANT_SQL_BASE = """
SELECT
    a.feature_stable_id,
    c.name,
    a.consequence_types,
    a.codon_allele_string,
    a.pep_allele_string,
    a.sift_prediction,
    a.sift_score
FROM transcript_variation a
    JOIN variation_feature b USING (variation_feature_id)
    JOIN variation c USING (variation_id)
WHERE {where_clause}
LIMIT 300
"""

_MARKER_SQL = """
SELECT d.name, e.left_primer, e.right_primer
FROM marker_synonym d
    JOIN marker e USING (marker_id)
WHERE d.name LIKE %s
LIMIT 300
"""


async def run_variant_query(
    pool: aiomysql.Pool,
    variant_id: str | None,
    transcript_id: str | None,
    consequence_types: list[str] | None = None,
) -> list[dict]:
    """Query transcript_variation for the given inputs."""
    clauses: list[str] = []
    params: list[str] = []

    if variant_id:
        clauses.append("c.name LIKE %s")
        params.append(f"{variant_id}%")
    if transcript_id:
        clauses.append("a.feature_stable_id LIKE %s")
        params.append(f"{transcript_id}%")
    if consequence_types:
        # Match any row whose consequence_types SET contains at least one selected value
        or_parts = " OR ".join(
            "FIND_IN_SET(%s, a.consequence_types) > 0" for _ in consequence_types
        )
        clauses.append(f"({or_parts})")
        params.extend(consequence_types)

    where_clause = " AND ".join(clauses)
    sql = _VARIANT_SQL_BASE.format(where_clause=where_clause)
    return await _run_query(pool, sql, tuple(params))


async def run_marker_query(
    core_pool: aiomysql.Pool,
    variant_id: str,
) -> list[dict]:
    """Query marker_synonym/marker in the core DB for matching KASP markers."""
    return await _run_query(core_pool, _MARKER_SQL, (f"{variant_id}%",))


# ---------------------------------------------------------------------------
# Public interface: fetch_and_join
# ---------------------------------------------------------------------------

async def fetch_and_join(
    variation_pool: aiomysql.Pool,
    core_pool: aiomysql.Pool,
    variant_id: str | None,
    transcript_id: str | None,
    consequence_types: list[str] | None = None,
) -> tuple[pd.DataFrame, bool]:
    """
    Run queries concurrently (when possible) and return a merged DataFrame
    plus a boolean indicating whether any marker rows were found.

    Variant query runs against the variation DB (variation_pool).
    Marker query runs against the core DB (core_pool).

    Raises ValueError if no variant records are found.
    """
    if variant_id:
        # Run both queries concurrently against their respective DBs
        rows1, rows2 = await asyncio.gather(
            run_variant_query(variation_pool, variant_id, transcript_id, consequence_types),
            run_marker_query(core_pool, variant_id),
        )
    else:
        # transcript_id only — no markers to fetch
        rows1 = await run_variant_query(variation_pool, variant_id, transcript_id, consequence_types)
        rows2 = []

    if not rows1:
        raise ValueError(
            "No variant records found for the given search parameters. "
            "Please check your input and try again."
        )

    df1 = pd.DataFrame(rows1).rename(columns={"name": "variant_name"})

    has_markers = bool(rows2)

    if rows2:
        df2 = pd.DataFrame(rows2).rename(columns={"name": "variant_name"})
        merged = pd.merge(df1, df2, on="variant_name", how="left")
    else:
        merged = df1

    # Split variant_name (format: Variant.Chromosome.Location) into 3 columns
    parts = merged["variant_name"].str.split(".", n=2, expand=True)
    pos = merged.columns.tolist().index("variant_name")
    merged.insert(pos, "Variant", parts[0])
    merged.insert(merged.columns.tolist().index("variant_name"), "Chromosome",
                  parts[1] if parts.shape[1] > 1 else None)
    merged.insert(merged.columns.tolist().index("variant_name"), "Location",
                  parts[2] if parts.shape[1] > 2 else None)
    merged = merged.drop(columns=["variant_name"])

    return merged, has_markers
