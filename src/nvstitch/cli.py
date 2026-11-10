"""The nvstitch command line.

The commands are declared here so the interface can be read in one place. Those
not implemented yet exit non-zero and say so. See docs/design.md.
"""

import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer

from nvstitch.build import build_facts
from nvstitch.export import export_profiles
from nvstitch.query import connect, show
from nvstitch.questions import available, summary

app = typer.Typer(
    help="Stitch per-rank nsys profiles into queryable tables and reports.",
    no_args_is_help=True,
    # A traceback's locals can hold profile data; the frames are enough.
    pretty_exceptions_show_locals=False,
)


def show_version(value: bool) -> None:
    if value:
        print(f"nvstitch {version('nvstitch')}")
        raise typer.Exit()


def question_names(incomplete: str) -> list[str]:
    return [name for name in available() if name.startswith(incomplete)]


@app.callback()
def options(
    ctx: typer.Context,
    collection: Annotated[Path, typer.Option(
        "-C", "--collection", metavar="DIR",
        help="Profile collection to work on.")] = Path("."),
    _version: Annotated[bool, typer.Option(
        "--version", callback=show_version, is_eager=True,
        help="Show the version and exit.")] = False,
) -> None:
    ctx.obj = collection


@app.command()
def build(
    ctx: typer.Context,
    jobs: Annotated[int, typer.Option(
        "-j", "--jobs", metavar="N", min=1,
        help="Export up to N profiles at once.")] = 1,
) -> None:
    """Export profiles and build the fact tables."""
    collection: Path = ctx.obj
    if not collection.is_dir():
        sys.exit(f"no collection at {collection}")
    export_profiles(collection, jobs)
    build_facts(collection)


@app.command()
def report(
    run: Annotated[str | None, typer.Argument(
        help="Run to report on. Default: the newest.")] = None,
    vs: Annotated[str | None, typer.Option(
        "--vs", metavar="RUN",
        help="Run to compare against. Default: the one before.")] = None,
) -> None:
    """Summarise a run."""
    sys.exit("nvstitch report: not implemented yet")


@app.command()
def ask(
    ctx: typer.Context,
    question: Annotated[str | None, typer.Argument(
        metavar="QUESTION", help="Question to answer. Default: list them.",
        autocompletion=question_names)] = None,
    run: Annotated[str | None, typer.Option(
        "--run", metavar="RUN", help="Run to ask about. Default: the newest.")] = None,
    md: Annotated[bool, typer.Option(
        "--md", help="Print markdown tables.")] = False,
) -> None:
    """List the questions, or answer one."""
    questions = available()
    if question is None:
        width = max(map(len, questions))
        for name, module in questions.items():
            print(f"{name:<{width}}  {summary(module)}")
        return
    if question not in questions:
        raise typer.BadParameter(f"no question {question!r}; `nvstitch ask` lists them",
                                 param_hint="QUESTION")
    con = connect(ctx.obj)
    kwargs = {} if run is None else {"run": run}
    for table in questions[question].answer(con, **kwargs):
        show(table, markdown=md or None)


def main() -> None:
    app()
