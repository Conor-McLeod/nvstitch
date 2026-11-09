"""The nvstitch command line.

The commands are declared here so the interface can be read in one place. None
of them is implemented yet; each exits non-zero and says so. See docs/design.md.
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import version
from pathlib import Path


def not_implemented(args: argparse.Namespace) -> None:
    sys.exit(f"nvstitch {args.command}: not implemented yet")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="nvstitch",
        description="Stitch per-rank nsys profiles into queryable tables and reports.")
    ap.add_argument("--version", action="version",
                    version=f"%(prog)s {version('nvstitch')}")
    ap.add_argument("-C", "--collection", type=Path, default=Path("."),
                    metavar="DIR",
                    help="profile collection to work on (default: current directory)")
    commands = ap.add_subparsers(dest="command", required=True)

    commands.add_parser("build", help="export profiles and build the fact tables")

    report = commands.add_parser("report", help="summarise a run")
    report.add_argument("run", nargs="?", help="run to report on (default: newest)")
    report.add_argument("--vs", metavar="RUN",
                        help="run to compare against (default: the one before)")

    ask = commands.add_parser("ask", help="list the questions, or run one")
    ask.add_argument("question", nargs="?", help="question to run (default: list them)")
    ask.add_argument("--job", metavar="RUN", help="run to ask about (default: newest)")

    return ap


def main() -> None:
    args = parser().parse_args()
    not_implemented(args)
