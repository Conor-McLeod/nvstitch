from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from conftest import EPOCH_NS, stage_profile
from nvstitch.build import build_facts
from nvstitch.cli import app

runner = CliRunner()
DAY = 86_400 * 10**9


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "build" in result.output and "ask" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("nvstitch ")


def test_build_needs_an_existing_collection(tmp_path: Path) -> None:
    result = runner.invoke(app, ["-C", str(tmp_path / "missing"), "build"])

    assert result.exit_code == 1
    assert "no collection at" in str(result.exception)


def test_build_rejects_zero_jobs(collection: Path) -> None:
    result = runner.invoke(app, ["-C", str(collection), "build", "-j", "0"])

    assert result.exit_code == 2


def test_build_writes_facts(collection: Path) -> None:
    stage_profile(collection / "r1" / "gemm_rank0.sqlite")

    result = runner.invoke(app, ["-C", str(collection), "build"])

    assert result.exit_code == 0, result.output
    assert (collection / "facts" / "profiles.parquet").exists()


def test_report_is_not_implemented() -> None:
    result = runner.invoke(app, ["report"])

    assert result.exit_code == 1
    assert "not implemented yet" in str(result.exception)


def test_ask_lists_the_questions() -> None:
    result = runner.invoke(app, ["ask"])

    assert result.exit_code == 0
    assert "north_star" in result.output and "stage_breakdown" in result.output


def test_ask_answers_for_a_run(collection: Path) -> None:
    stage_profile(collection / "b" / "gemm_rank0.sqlite", epoch=EPOCH_NS)
    stage_profile(collection / "a" / "gemm_rank0.sqlite", epoch=EPOCH_NS + DAY)
    build_facts(collection)

    result = runner.invoke(app, ["-C", str(collection), "ask", "stage_breakdown",
                                 "--run", "b", "--md"])

    assert result.exit_code == 0, result.output
    assert "## b / gemm / rank 0 -- wall clock" in result.output
    assert "|" in result.output


def test_ask_rejects_an_unknown_question() -> None:
    result = runner.invoke(app, ["ask", "nope"])

    assert result.exit_code == 2
    assert "no question 'nope'" in result.output
