"""Build the parquet fact tables from every exported profile in a collection.

Walks <collection>/<run>/<label>[_rank<n>].sqlite, runs extract.sql against each
one, concatenates the results and writes <collection>/facts/*.parquet.

Full rebuild every run. That is deliberate: the tables worth keeping are small
enough that rebuilding from scratch costs seconds, and it removes any question
of a stale fact table disagreeing with the profiles.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from importlib.resources import files
from pathlib import Path

import duckdb

# Order matters: v_profiles is cross-joined by the others.
FACTS = ("profiles", "nvtx", "kernels", "memops", "mpi", "gpu_metrics")

PROFILE_RE = re.compile(r"^(?P<label>.+?)(?:_rank(?P<rank>\d+))?$")

# Without these a file is not an nsys export at all.
REQUIRED_TABLES = ("META_DATA_EXPORT", "TARGET_INFO_SESSION_START_TIME", "StringIds")

# nsys writes a table only when the profile has something to put in it, so a
# profile taken without MPI tracing has no MPI tables, and a view naming a missing
# table fails when it is created. extract.sql reads each of these through a
# src_<table> view, which is the real table or an empty one with these columns.
OPTIONAL_TABLES = {
    "MPI_RANKS": ("rank",),
    "TARGET_INFO_GPU": ("name",),
    "TARGET_INFO_SYSTEM_ENV": ("name", "value"),
    "NVTX_EVENTS": ("start", "end", "text", "textId", "globalTid"),
    "CUPTI_ACTIVITY_KIND_KERNEL": (
        "start", "end", "shortName", "demangledName", "correlationId", "deviceId",
        "streamId", "graphNodeId", "gridX", "gridY", "gridZ", "blockX", "blockY",
        "blockZ", "registersPerThread", "staticSharedMemory", "dynamicSharedMemory"),
    "CUPTI_ACTIVITY_KIND_RUNTIME": ("start", "correlationId", "globalTid"),
    "CUPTI_ACTIVITY_KIND_MEMCPY": (
        "start", "end", "copyKind", "correlationId", "bytes", "deviceId", "streamId"),
    "CUPTI_ACTIVITY_KIND_MEMSET": (
        "start", "end", "correlationId", "bytes", "deviceId", "streamId"),
    "ENUM_CUDA_MEMCPY_OPER": ("id", "label"),
    "MPI_COLLECTIVES_EVENTS": ("start", "end", "textId", "size", "recvSize", "rootRank"),
    "MPI_OTHER_EVENTS": ("start", "end", "textId"),
}
VARCHAR_COLUMNS = {"name", "value", "text", "label"}


def build_facts(collection: Path) -> None:
    profiles = sorted(collection.glob("*/*.sqlite"))
    if not profiles:
        sys.exit(f"no profiles under {collection}: expected "
                 f"<run>/<label>_rank<n>.nsys-rep or .sqlite")

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")
    extract_sql = files("nvstitch").joinpath("sql/extract.sql").read_text()

    created = False
    missing_metrics: Counter[tuple[str, ...]] = Counter()
    for db in profiles:
        match = PROFILE_RE.match(db.stem)
        assert match is not None
        con.execute(f"ATTACH {sql_string(db)} AS s (TYPE sqlite, READ_ONLY)")
        try:
            tables = attached_tables(con)
            absent = [t for t in REQUIRED_TABLES if t not in tables]
            if absent:
                sys.exit(f"{db}: not an nsys SQLite export (no {', '.join(absent)})")
            point_optional_tables(con, tables)
            point_gpu_metrics_src(con, tables)

            from_name = None if match["rank"] is None else int(match["rank"])
            con.execute("SET VARIABLE run = ?", [db.parent.name])
            con.execute("SET VARIABLE label = ?", [match["label"]])
            con.execute("SET VARIABLE rank = ?", [resolve_rank(con, db, from_name)])
            con.execute(extract_sql)

            missing = missing_gpu_metrics(con, db)
            if missing:
                missing_metrics[missing] += 1
            for table in FACTS:
                verb = (f"INSERT INTO {table} BY NAME" if created
                        else f"CREATE TABLE {table} AS")
                con.execute(f"{verb} SELECT * FROM v_{table}")
        finally:
            con.execute("DETACH s")
        created = True

    check_schema_versions(con)

    for columns, count in missing_metrics.items():
        print(f"warning: {count} profile(s) have no samples for "
              f"{', '.join(columns)}", file=sys.stderr)

    out = collection / "facts"
    out.mkdir(exist_ok=True)
    for table in FACTS:
        path = out / f"{table}.parquet"
        con.execute(f"COPY {table} TO {sql_string(path)} (FORMAT parquet)")
        (rows,) = con.execute(f"SELECT count(*) FROM {table}").fetchone()
        print(f"{rows:>8,} rows  {path.name}")

    (runs,) = con.execute("SELECT count(DISTINCT run) FROM profiles").fetchone()
    print(f"\n{len(profiles)} profile(s) from {runs} run(s) -> {out}")


def sql_string(path: Path) -> str:
    """path as a SQL string literal, for the statements that take no binds."""
    return "'" + str(path).replace("'", "''") + "'"


def attached_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute("SELECT table_name FROM duckdb_tables() "
                       "WHERE database_name = 's'").fetchall()
    return {name for (name,) in rows}


def point_optional_tables(con: duckdb.DuckDBPyConnection, tables: set[str]) -> None:
    """Define the src_<table> views that extract.sql reads."""
    for table, columns in OPTIONAL_TABLES.items():
        if table in tables:
            src = f"SELECT * FROM s.{table}"
        else:
            typed = ", ".join(
                f'NULL::{"VARCHAR" if c in VARCHAR_COLUMNS else "BIGINT"} AS "{c}"'
                for c in columns)
            src = f"SELECT {typed} WHERE false"
        con.execute(f"CREATE OR REPLACE TEMP VIEW src_{table.lower()} AS {src}")


def point_gpu_metrics_src(con: duckdb.DuckDBPyConnection, tables: set[str]) -> None:
    """Define the gpu_metrics_src view that extract.sql's v_gpu_metrics reads.

    Separate from point_optional_tables because the real source is a join.
    """
    if "GPU_METRICS" in tables:
        src = ("SELECT g.timestamp AS ts_ns, g.typeId - t.sourceId AS device_id, "
               "       t.metricName AS metric, g.value "
               "FROM s.GPU_METRICS g "
               "JOIN s.TARGET_INFO_GPU_METRICS t USING (typeId, metricId)")
    else:
        src = ("SELECT NULL::BIGINT AS ts_ns, NULL::BIGINT AS device_id, "
               "       NULL::VARCHAR AS metric, NULL::DOUBLE AS value "
               "WHERE false")
    con.execute(f"CREATE OR REPLACE TEMP VIEW gpu_metrics_src AS {src}")


def resolve_rank(con: duckdb.DuckDBPyConnection, db: Path,
                 from_name: int | None) -> int:
    """The profile's rank, cross-checking MPI_RANKS against the filename.

    The filename is what nsys was told to write; the table is what the process
    reported itself as. A disagreement means every per-rank conclusion drawn from
    this profile is suspect, so fail rather than silently pick one.
    """
    row = con.execute("SELECT rank FROM src_mpi_ranks LIMIT 1").fetchone()
    from_db = None if row is None else row[0]
    if from_db is None:
        if from_name is not None:
            print(f"warning: {db.name} has no MPI_RANKS row, trusting filename rank",
                  file=sys.stderr)
        return from_name or 0
    if from_name is not None and from_db != from_name:
        sys.exit(f"{db}: filename says rank {from_name}, MPI_RANKS says {from_db}")
    return from_db


def missing_gpu_metrics(con: duckdb.DuckDBPyConnection, db: Path) -> tuple[str, ...]:
    """The v_gpu_metrics columns with no samples in this profile.

    v_gpu_metrics picks metrics out by their nsys display name, and the set a
    profile has depends on the GPU: C2C exists only on Grace Hopper, for example.
    So a few empty columns are expected and only worth a warning. All of them
    empty means the names no longer match this nsys release, which is an error.
    """
    rel = con.sql("SELECT count(*) AS n, count(COLUMNS(* EXCLUDE "
                  "(run, label, rank, device_id, ts_ns, abs_ts_ns))) "
                  "FROM v_gpu_metrics")
    n, *counts = rel.fetchone()
    if not n:
        return ()
    missing = tuple(col for col, c in zip(rel.columns[1:], counts) if c == 0)
    if len(missing) == len(counts):
        sys.exit(f"{db}: no samples for any GPU metric -- the nsys metric names in "
                 f"extract.sql's v_gpu_metrics no longer match "
                 f"TARGET_INFO_GPU_METRICS")
    return missing


def check_schema_versions(con: duckdb.DuckDBPyConnection) -> None:
    """Refuse to mix nsys export schema versions in one set of fact tables.

    Column names and table contents shift between nsys releases. Two versions
    concatenated into one table is the kind of thing that does not error, it just
    quietly makes a longitudinal comparison meaningless.
    """
    rows = con.execute(
        "SELECT schema_version, nsys_version, count(*) AS n, "
        "       min(run) AS first_run, max(run) AS last_run "
        "FROM profiles GROUP BY 1, 2 ORDER BY 3 DESC").fetchall()
    if len(rows) > 1:
        detail = "\n".join(
            f"  schema {s} (nsys {v}): {n} profile(s), runs {a}..{b}"
            for s, v, n, a, b in rows)
        sys.exit(f"profiles span {len(rows)} nsys export schema versions:\n"
                 f"{detail}\n"
                 f"Re-export the older runs with the current nsys, or build a "
                 f"collection holding a single version.")
