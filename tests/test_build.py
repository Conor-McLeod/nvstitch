from __future__ import annotations

import sqlite3
from pathlib import Path

import duckdb
import pytest

from conftest import EPOCH_NS, METRICS, make_profile
from nvstitch.build import FACTS, build_facts


def rows(collection: Path, query: str) -> list[tuple]:
    """Run query with each fact table available under its own name."""
    con = duckdb.connect()
    for table in FACTS:
        path = collection / "facts" / f"{table}.parquet"
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path}')")
    return con.execute(query).fetchall()


def test_stitches_ranks_and_runs(collection: Path) -> None:
    for rank in range(2):
        make_profile(collection / "1001" / f"gemm_rank{rank}.sqlite", rank=rank)
    make_profile(collection / "1002" / "gemm_rank0.sqlite")

    build_facts(collection)

    assert rows(collection, "SELECT run, label, rank FROM profiles ORDER BY 1, 3") == [
        ("1001", "gemm", 0), ("1001", "gemm", 1), ("1002", "gemm", 0)]
    assert rows(collection, "SELECT count(*) FROM kernels") == [(3,)]


def test_derived_columns(collection: Path) -> None:
    make_profile(collection / "r" / "gemm_rank0.sqlite")

    build_facts(collection)

    # The unfinished range is dropped and the domain prefix is stripped.
    assert rows(collection, "SELECT name, dur_ms, abs_start_ns FROM nvtx") == [
        ("rep 0", 5.0, EPOCH_NS + 1000)]
    assert rows(collection, "SELECT name, launch_ns, launch_tid FROM kernels") == [
        ("kern", 1500, 2)]
    assert rows(collection, "SELECT kind, op, bytes FROM memops ORDER BY kind") == [
        ("memcpy", "DtoH", 200), ("memset", None, 100)]
    assert rows(collection, "SELECT kind, name, recv_size FROM mpi ORDER BY kind") == [
        ("collective", "MPI_Allreduce", 64), ("other", "MPI_Allreduce", None)]


def test_label_keeps_underscores_and_prefix(collection: Path) -> None:
    make_profile(collection / "r" / "run_par_ozaki_rank3.sqlite", rank=3)

    build_facts(collection)

    assert rows(collection, "SELECT label, rank FROM profiles") == [("run_par_ozaki", 3)]


def test_single_rank_without_suffix_or_optional_tables(collection: Path) -> None:
    make_profile(collection / "r" / "solo.sqlite", mpi=False, activity=False)

    build_facts(collection)

    assert rows(collection, "SELECT label, rank FROM profiles") == [("solo", 0)]
    for table in ("nvtx", "kernels", "memops", "mpi", "gpu_metrics"):
        assert rows(collection, f"SELECT count(*) FROM {table}") == [(0,)]


def test_sparse_profile_first_then_full_profile(collection: Path) -> None:
    """The first profile's empty views set the types the later ones insert into."""
    make_profile(collection / "a" / "solo.sqlite", mpi=False, activity=False)
    make_profile(collection / "b" / "gemm_rank0.sqlite", metrics=METRICS)

    build_facts(collection)

    assert rows(collection, "SELECT run FROM kernels") == [("b",)]
    assert rows(collection, "SELECT run, gpc_clock_mhz FROM gpu_metrics") == [("b", 1980.0)]


def test_filename_rank_used_when_mpi_ranks_is_empty(collection: Path, capsys) -> None:
    make_profile(collection / "r" / "gemm_rank2.sqlite", rank=None)

    build_facts(collection)

    assert rows(collection, "SELECT rank FROM profiles") == [(2,)]
    assert "trusting filename rank" in capsys.readouterr().err


def test_mpi_ranks_used_when_filename_has_no_rank(collection: Path) -> None:
    make_profile(collection / "r" / "gemm.sqlite", rank=5)

    build_facts(collection)

    assert rows(collection, "SELECT rank FROM profiles") == [(5,)]


def test_rank_mismatch_fails(collection: Path) -> None:
    make_profile(collection / "r" / "gemm_rank1.sqlite", rank=0)

    with pytest.raises(SystemExit, match="filename says rank 1, MPI_RANKS says 0"):
        build_facts(collection)


def test_mixed_schema_versions_fail(collection: Path) -> None:
    make_profile(collection / "a" / "gemm_rank0.sqlite", schema="3.11")
    make_profile(collection / "b" / "gemm_rank0.sqlite", schema="3.12")

    with pytest.raises(SystemExit, match="2 nsys export schema versions"):
        build_facts(collection)


def test_some_gpu_metrics_missing_warns_once(collection: Path, capsys) -> None:
    for rank in range(2):
        make_profile(collection / "r" / f"gemm_rank{rank}.sqlite", rank=rank,
                     metrics=METRICS)

    build_facts(collection)

    err = capsys.readouterr().err
    assert err.count("2 profile(s) have no samples for") == 1
    assert "c2c_rx_pct" in err
    assert "gpc_clock_mhz" not in err


def test_all_gpu_metrics_missing_fails(collection: Path) -> None:
    make_profile(collection / "r" / "gemm_rank0.sqlite", metrics={1: "Renamed Metric"})

    with pytest.raises(SystemExit, match="no samples for any GPU metric"):
        build_facts(collection)


def test_empty_collection_fails(collection: Path) -> None:
    with pytest.raises(SystemExit, match="no profiles under"):
        build_facts(collection)


def test_unrelated_sqlite_fails_clearly(collection: Path) -> None:
    path = collection / "r" / "notes.sqlite"
    path.parent.mkdir()
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (x INTEGER)")
    con.close()

    with pytest.raises(SystemExit, match="not an nsys SQLite export"):
        build_facts(collection)


def test_path_with_a_quote(tmp_path: Path) -> None:
    collection = tmp_path / "bob's runs"
    make_profile(collection / "r" / "gemm_rank0.sqlite")

    build_facts(collection)

    assert (collection / "facts" / "profiles.parquet").exists()
