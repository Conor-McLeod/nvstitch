"""The nvstitch command line.

The commands are declared here so the interface can be read in one place. Those
not implemented yet exit non-zero and say so. See docs/design.md.
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

from nvstitch.build import build_facts
from nvstitch.export import export_profiles


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

    build = commands.add_parser("build", help="export profiles and build the fact tables")
    build.add_argument("-j", "--jobs", type=int, default=1, metavar="N",
                       help="export up to N profiles at once (default: 1)")

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
    if args.command == "build":
        if not args.collection.is_dir():
            sys.exit(f"no collection at {args.collection}")
        export_profiles(args.collection, args.jobs)
        build_facts(args.collection)
    else:
        not_implemented(args)
