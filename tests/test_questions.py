from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import EPOCH_NS, make_profile, stage_profile
from nvstitch import cli, connect, latest_run
from nvstitch.build import build_facts
from nvstitch.questions import north_star, stage_breakdown

DAY = 86_400 * 10**9


def two_rank_run(collection: Path, run: str, *, epoch: int = EPOCH_NS,
                 slow: tuple[float, float] = (1.5, 2.0)) -> None:
    """Rank 0 has stages of 1 and 2 ms, rank 1 is slower and gates the run."""
    stage_profile(collection / run / "gemm_rank0.sqlite", rank=0, epoch=epoch)
    stage_profile(collection / run / "gemm_rank1.sqlite", rank=1, epoch=epoch,
                  stages=slow)


def table(tables, i):
    return tables[i].df.to_dict("records")


# --------------------------------------------------------------- connect

def test_connect_requires_facts(collection: Path) -> None:
    with pytest.raises(SystemExit, match="run `nvstitch build` first"):
        connect(collection)


def test_runs_view_is_empty_without_a_runs_table(collection: Path) -> None:
    two_rank_run(collection, "r1")
    build_facts(collection)

    con = connect(collection)

    assert con.sql("select * from runs").columns == ["run", "label", "time_ms"]
    assert con.sql("select count(*) from runs").fetchone() == (0,)


def test_runs_table_reads_run_as_text(collection: Path) -> None:
    two_rank_run(collection, "001")
    build_facts(collection)
    (collection / "runs.csv").write_text("run,label,time_ms,knob\n001,gemm,3.5,7\n")

    rows = connect(collection).sql("select run, time_ms, knob from runs").fetchall()

    assert rows == [("001", 3.5, 7)]


@pytest.mark.parametrize("contents, message", [
    ("run,label\nr1,gemm\n", "missing required column.s. time_ms"),
    ("run,label,time_ms\nr1,gemm,1\nr1,gemm,2\n", "r1, gemm: 2 rows"),
    ("run,label,time_ms\nr9,gemm,1\n", "none of its 1 .run, label. pairs match"),
])
def test_runs_table_rules(collection: Path, contents: str, message: str) -> None:
    two_rank_run(collection, "r1")
    build_facts(collection)
    (collection / "runs.csv").write_text(contents)

    with pytest.raises(SystemExit, match=message):
        connect(collection)


def test_runs_table_may_cover_only_some_profiles(collection: Path) -> None:
    two_rank_run(collection, "r1")
    two_rank_run(collection, "r2", epoch=EPOCH_NS + DAY)
    build_facts(collection)
    (collection / "runs.csv").write_text("run,label,time_ms\nr1,gemm,\n")

    connect(collection)


def test_extra_sql_columns_reach_kernel_stages(collection: Path, tmp_path: Path) -> None:
    two_rank_run(collection, "r1")
    build_facts(collection)
    extra = tmp_path / "app.sql"
    extra.write_text("create or replace view kernels as "
                     "select *, upper(name) as shouting from read_parquet('facts/kernels.parquet');")

    con = connect(collection, extra_sql=(extra,))

    assert con.sql("select distinct shouting from kernel_stages").fetchall() == [("KERN",)]


# --------------------------------------------------------------- views

def test_run_order_follows_session_start_not_name(collection: Path) -> None:
    two_rank_run(collection, "b", epoch=EPOCH_NS)
    two_rank_run(collection, "a", epoch=EPOCH_NS + DAY)
    build_facts(collection)

    con = connect(collection)

    assert con.sql("select run from run_order order by run_index").fetchall() == [
        ("b",), ("a",)]
    assert latest_run(con) == "a"


def test_profiles_without_the_convention_have_no_stages(collection: Path) -> None:
    make_profile(collection / "r1" / "gemm_rank0.sqlite")
    build_facts(collection)

    con = connect(collection)

    assert con.sql("select count(*) from stages").fetchone() == (0,)
    with pytest.raises(SystemExit, match="no run in the collection has NVTX stage"):
        latest_run(con)


# --------------------------------------------------------------- questions

def test_stage_breakdown_shares_of_the_iteration(collection: Path) -> None:
    two_rank_run(collection, "r1")
    build_facts(collection)

    tables = stage_breakdown.answer(connect(collection), reps_only=True)

    assert tables[0].caption == "r1 / gemm / rank 0 -- wall clock"
    rep0 = [r for r in table(tables, 0) if r["bucket"] == "rep 0"]
    assert [(r["stage"], round(r["ms"], 6), round(r["pct"], 6)) for r in rep0] == [
        ("scaling", 1.0, 10.0), ("work", 2.0, 20.0),
        ("(unaccounted)", 7.0, 70.0), ("TOTAL (iteration)", 10.0, 100.0)]


def test_stage_breakdown_shares_of_a_named_call(collection: Path) -> None:
    two_rank_run(collection, "r1")
    build_facts(collection)

    tables = stage_breakdown.answer(connect(collection), call="%gemm", reps_only=True)

    rep0 = [r for r in table(tables, 0) if r["bucket"] == "rep 0"]
    assert [(r["stage"], round(r["ms"], 6)) for r in rep0] == [
        ("scaling", 1.0), ("work", 2.0), ("(unaccounted)", 5.5), ("TOTAL (%gemm)", 8.5)]


def test_stage_breakdown_keeps_the_warmup_unless_asked(collection: Path) -> None:
    two_rank_run(collection, "r1")
    build_facts(collection)

    tables = stage_breakdown.answer(connect(collection))

    assert {r["bucket"] for r in table(tables, 0)} == {"warmup 0", "rep 0", "rep 1"}


def test_stage_breakdown_charges_kernels_to_the_launching_stage(collection: Path) -> None:
    two_rank_run(collection, "r1")
    build_facts(collection)

    tables = stage_breakdown.answer(connect(collection), reps_only=True)

    assert table(tables, 1)[0] == {
        "bucket": "rep 0", "stage": "work", "launches": 1, "kernel_ms": 0.5,
        "busy_ms": 0.5, "memops": 0, "memops_ms": 0.0, "pct": 100.0}


def test_north_star_gating_rank_and_delta(collection: Path) -> None:
    two_rank_run(collection, "b", epoch=EPOCH_NS)
    two_rank_run(collection, "a", epoch=EPOCH_NS + DAY, slow=(2.5, 2.0))
    build_facts(collection)
    (collection / "runs.csv").write_text(
        "run,label,time_ms,knob\nb,gemm,3.1,0\na,gemm,4.6,1\n")

    tables = north_star.answer(connect(collection), meta=["knob"])

    [latest] = table(tables, 0)
    assert (latest["run"], latest["north_star"], latest["prev"], latest["delta"],
            latest["slowest_rank"], latest["knob"]) == ("a", 4.5, 3.5, 1.0, 1, 1)
    history = table(tables, 1)
    assert [(r["run"], r["imbalance"], r["gated_by"]) for r in history] == [
        ("a", 0.75, "1"), ("b", 0.25, "1")]
    moved = {r["stage"]: r["delta"] for r in table(tables, 2)}
    assert moved == {"scaling": 1.0, "work": 0.0}


def test_north_star_run_and_exclude(collection: Path) -> None:
    two_rank_run(collection, "b", epoch=EPOCH_NS)
    two_rank_run(collection, "a", epoch=EPOCH_NS + DAY)
    build_facts(collection)
    con = connect(collection)

    assert [r["run"] for r in table(north_star.answer(con, run="b"), 1)] == ["b"]
    assert [r["run"] for r in table(north_star.answer(con, exclude=["a"]), 1)] == ["b"]


def test_north_star_rejects_unknown_arguments(collection: Path) -> None:
    two_rank_run(collection, "r1")
    build_facts(collection)
    con = connect(collection)

    with pytest.raises(SystemExit, match="doesn't have: knob"):
        north_star.answer(con, meta=["knob"])
    with pytest.raises(SystemExit, match="no run 'nope'"):
        north_star.answer(con, run="nope")


# --------------------------------------------------------------- cli

def run_cli(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["nvstitch", *argv])
    cli.main()


def test_ask_lists_the_questions(monkeypatch, capsys) -> None:
    run_cli(monkeypatch, "ask")

    out = capsys.readouterr().out
    assert "north_star" in out and "stage_breakdown" in out


def test_ask_answers_for_a_run(collection: Path, monkeypatch, capsys) -> None:
    two_rank_run(collection, "b", epoch=EPOCH_NS)
    two_rank_run(collection, "a", epoch=EPOCH_NS + DAY)
    build_facts(collection)
    capsys.readouterr()

    run_cli(monkeypatch, "-C", str(collection), "ask", "stage_breakdown", "--run", "b")

    assert "## b / gemm / rank 0 -- wall clock" in capsys.readouterr().out


def test_ask_rejects_an_unknown_question(monkeypatch) -> None:
    with pytest.raises(SystemExit, match="no question 'nope'"):
        run_cli(monkeypatch, "ask", "nope")
