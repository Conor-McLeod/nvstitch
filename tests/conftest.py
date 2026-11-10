"""Synthetic stand-ins for nsys SQLite exports.

The schema here is only what extract.sql reads, so these tests exercise nvstitch's
own logic. They say nothing about whether a real nsys release matches that schema.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

EPOCH_NS = 1_000_000_000_000

METRICS = {1: "GPC Clock Frequency [MHz]", 2: "SYS Clock Frequency [MHz]"}

CORE = """
CREATE TABLE META_DATA_EXPORT (name TEXT, value TEXT);
CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER, utcTime TEXT);
CREATE TABLE TARGET_INFO_SYSTEM_ENV (name TEXT, value TEXT);
CREATE TABLE TARGET_INFO_GPU (name TEXT);
CREATE TABLE StringIds (id INTEGER, value TEXT);
"""

MPI = """
CREATE TABLE MPI_RANKS (rank INTEGER);
CREATE TABLE MPI_COLLECTIVES_EVENTS
  (start INTEGER, "end" INTEGER, textId INTEGER, size INTEGER, recvSize INTEGER,
   rootRank INTEGER);
CREATE TABLE MPI_OTHER_EVENTS (start INTEGER, "end" INTEGER, textId INTEGER);
INSERT INTO MPI_COLLECTIVES_EVENTS VALUES (100, 3000100, 3, 64, 64, 0);
INSERT INTO MPI_OTHER_EVENTS VALUES (100, 2000100, 3);
"""

ACTIVITY = """
CREATE TABLE NVTX_EVENTS
  (start INTEGER, "end" INTEGER, text TEXT, textId INTEGER, globalTid INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
  (start INTEGER, correlationId INTEGER, globalTid INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
  (start INTEGER, "end" INTEGER, shortName INTEGER, demangledName INTEGER,
   correlationId INTEGER, deviceId INTEGER, streamId INTEGER, graphNodeId INTEGER,
   gridX INTEGER, gridY INTEGER, gridZ INTEGER, blockX INTEGER, blockY INTEGER,
   blockZ INTEGER, registersPerThread INTEGER, staticSharedMemory INTEGER,
   dynamicSharedMemory INTEGER);
CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, label TEXT);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY
  (start INTEGER, "end" INTEGER, copyKind INTEGER, correlationId INTEGER,
   bytes INTEGER, deviceId INTEGER, streamId INTEGER);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET
  (start INTEGER, "end" INTEGER, correlationId INTEGER, bytes INTEGER,
   deviceId INTEGER, streamId INTEGER);
"""

ACTIVITY_ROWS = """
INSERT INTO NVTX_EVENTS VALUES (1000, 5001000, ':rep 0', NULL, 16777217);
INSERT INTO NVTX_EVENTS VALUES (2000, NULL, 'unfinished', NULL, 16777217);
INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (1500, 7, 16777218);
INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
  (2000, 4000, 1, 2, 7, 0, 1, NULL, 8, 1, 1, 64, 1, 1, 12, 0, 0);
INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1, 'DtoH');
INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (3000, 3100, 1, 8, 200, 0, 1);
INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (3200, 3300, 9, 100, 0, 1);
"""

GPU_METRICS = """
CREATE TABLE TARGET_INFO_GPU_METRICS
  (typeId INTEGER, sourceId INTEGER, metricId INTEGER, metricName TEXT);
CREATE TABLE GPU_METRICS
  (timestamp INTEGER, typeId INTEGER, metricId INTEGER, value REAL);
"""


def make_profile(path: Path, *, rank: int | None = 0, mpi: bool = True,
                 activity: bool = True, activity_rows: bool = True,
                 metrics: dict[int, str] | None = None, schema: str = "3.11",
                 epoch: int = EPOCH_NS, tables: str = "") -> Path:
    """Write a synthetic export. rank=None leaves MPI_RANKS present but empty.

    activity_rows=False creates the activity tables empty, for a caller that
    fills them through `tables`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(CORE)
    con.execute("INSERT INTO META_DATA_EXPORT VALUES "
                "('EXPORT_PRODUCT_VERSION', '2025.5'), "
                "('EXPORT_SCHEMA_VERSION', ?)", [schema])
    con.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?, '2026-11-09')",
                [epoch])
    con.execute("INSERT INTO TARGET_INFO_SYSTEM_ENV VALUES ('Hostname', 'node-1')")
    con.execute("INSERT INTO TARGET_INFO_GPU VALUES ('GPU A')")
    con.execute("INSERT INTO StringIds VALUES (1, 'kern'), (2, 'kern(int)'), "
                "(3, 'MPI_Allreduce')")
    if mpi:
        con.executescript(MPI)
        if rank is not None:
            con.execute("INSERT INTO MPI_RANKS VALUES (?)", [rank])
    if activity:
        con.executescript(ACTIVITY)
        if activity_rows:
            con.executescript(ACTIVITY_ROWS)
    if metrics is not None:
        con.executescript(GPU_METRICS)
        for metric_id, name in metrics.items():
            con.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (10, 10, ?, ?)",
                        [metric_id, name])
            con.execute("INSERT INTO GPU_METRICS VALUES (500, 10, ?, 1.98e9)",
                        [metric_id])
    if tables:
        con.executescript(tables)
    con.commit()
    con.close()
    return path


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    root = tmp_path / "collection"
    root.mkdir()
    return root


MS = 1_000_000
TID = 16777217     # globalTid whose low 24 bits, the thread id, are 1


def stage_profile(path: Path, *, rank: int = 0, epoch: int = EPOCH_NS,
                  stages: tuple[float, float] = (1.0, 2.0),
                  warmup: tuple[float, float] | None = None) -> Path:
    """A profile following the NVTX convention.

    Three 10 ms iterations (`warmup 0`, `rep 0`, `rep 1`), each holding a
    `my_gemm` range from 0.5 to 9 ms in and two stages back to back from 1 ms in:
    `1 scaling` and `2 work`, lasting `stages` ms (`warmup` ms in the warmup).
    `2 work` launches one 0.5 ms kernel 0.1 ms after it opens.
    """
    rows = []
    for i, (bucket, start) in enumerate((("warmup 0", 0), ("rep 0", 20 * MS),
                                          ("rep 1", 40 * MS))):
        s1, s2 = warmup if (warmup and bucket.startswith("warmup")) else stages
        a, b, c = start + MS, start + MS + int(s1 * MS), start + MS + int((s1 + s2) * MS)
        rows += [
            f"INSERT INTO NVTX_EVENTS VALUES ({start}, {start + 10 * MS}, '{bucket}', NULL, {TID});",
            f"INSERT INTO NVTX_EVENTS VALUES ({start + MS // 2}, {start + 9 * MS}, 'my_gemm', NULL, {TID});",
            f"INSERT INTO NVTX_EVENTS VALUES ({a}, {b}, '1 scaling', NULL, {TID});",
            f"INSERT INTO NVTX_EVENTS VALUES ({b}, {c}, '2 work', NULL, {TID});",
            f"INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES ({b + MS // 10}, {i + 1}, {TID});",
            f"INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({b + MS // 5}, {b + MS // 5 + MS // 2},"
            f" 1, 2, {i + 1}, 0, 1, NULL, 1, 1, 1, 1, 1, 1, 1, 0, 0);",
        ]
    return make_profile(path, rank=rank, epoch=epoch, activity_rows=False,
                        tables="\n".join(rows))
