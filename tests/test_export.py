from __future__ import annotations

import os
from pathlib import Path

import pytest

from nvstitch.export import export_profiles

FAKE_NSYS = """#!/bin/sh
echo "$@" >> "$NSYS_CALLS"
while [ $# -gt 0 ]; do
  if [ "$1" = -o ]; then out=$2; shift; fi
  shift
done
echo partial > "$out"
if [ -n "$NSYS_FAIL" ]; then echo "$NSYS_FAIL" >&2; exit 3; fi
"""


@pytest.fixture
def nsys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Install a fake nsys and return the file recording its invocations."""
    script = tmp_path / "fake-nsys"
    script.write_text(FAKE_NSYS)
    script.chmod(0o755)
    calls = tmp_path / "calls"
    calls.touch()
    monkeypatch.setenv("NSYS", str(script))
    monkeypatch.setenv("NSYS_CALLS", str(calls))
    return calls


def rep_at(path: Path, mtime: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, (mtime, mtime))
    return path


def test_exports_a_missing_database(collection: Path, nsys: Path) -> None:
    rep = rep_at(collection / "r" / "gemm_rank0.nsys-rep", 100)

    export_profiles(collection)

    assert rep.with_suffix(".sqlite").exists()
    assert len(nsys.read_text().splitlines()) == 1


def test_skips_a_newer_database(collection: Path, nsys: Path) -> None:
    rep = rep_at(collection / "r" / "gemm_rank0.nsys-rep", 100)
    rep_at(rep.with_suffix(".sqlite"), 200)

    export_profiles(collection)

    assert nsys.read_text() == ""


def test_reexports_a_database_not_newer_than_its_report(
        collection: Path, nsys: Path) -> None:
    rep = rep_at(collection / "r" / "gemm_rank0.nsys-rep", 200)
    rep_at(rep.with_suffix(".sqlite"), 100)

    export_profiles(collection)

    assert len(nsys.read_text().splitlines()) == 1


def test_parallel_export_covers_every_profile(collection: Path, nsys: Path) -> None:
    reps = [rep_at(collection / "r" / f"gemm_rank{i}.nsys-rep", 100) for i in range(4)]

    export_profiles(collection, jobs=2)

    assert all(rep.with_suffix(".sqlite").exists() for rep in reps)


def test_failure_reports_stderr_and_leaves_nothing_behind(
        collection: Path, nsys: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NSYS_FAIL", "license expired")
    rep = rep_at(collection / "r" / "gemm_rank0.nsys-rep", 100)

    with pytest.raises(SystemExit, match="license expired"):
        export_profiles(collection)

    assert [p.name for p in rep.parent.iterdir()] == [rep.name]


def test_missing_nsys_fails_when_an_export_is_needed(
        collection: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NSYS", "/no/such/nsys")
    rep_at(collection / "r" / "gemm_rank0.nsys-rep", 100)

    with pytest.raises(SystemExit, match="1 profile.s. need exporting"):
        export_profiles(collection)


def test_missing_nsys_is_fine_when_nothing_is_stale(
        collection: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NSYS", "/no/such/nsys")
    rep = rep_at(collection / "r" / "gemm_rank0.nsys-rep", 100)
    rep_at(rep.with_suffix(".sqlite"), 200)

    export_profiles(collection)
