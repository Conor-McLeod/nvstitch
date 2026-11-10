"""Querying a built collection: the connection, and helpers for questions.

    con = connect(collection)
    con.sql("select * from stage_gpu where run = $run", params={"run": "2160148"})

These are the public API that application repositories build on, so their
behaviour should stay stable.
"""

from __future__ import annotations

import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import NamedTuple

import duckdb
import pandas as pd

from nvstitch.build import sql_string

RUNS_REQUIRED = ("run", "label", "time_ms")


class Table(NamedTuple):
    """One result of a question: a frame, its caption, and per-column formats."""
    caption: str
    df: pd.DataFrame
    fmt: dict[str, str]


def connect(collection: Path, extra_sql: tuple[Path, ...] = ()) -> duckdb.DuckDBPyConnection:
    """A connection with the views loaded over collection's fact tables.

    extra_sql files run after the views, in order, which is how an application
    adds views of its own or redefines one of these.
    """
    collection = Path(collection).resolve()
    if not (collection / "facts" / "profiles.parquet").exists():
        sys.exit(f"no fact tables in {collection} -- run `nvstitch build` first")
    con = duckdb.connect()
    con.execute("SET file_search_path = ?", [str(collection)])
    con.execute(files("nvstitch").joinpath("sql/views.sql").read_text())
    load_runs(con, collection / "runs.csv")
    for path in extra_sql:
        con.execute(Path(path).read_text())
    return con


def load_runs(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    """Define the `runs` view over the optional runs table, after checking it.

    Without the file, `runs` is an empty view with the required columns, so a
    question can left-join it either way. See docs/design.md for the rules.
    """
    if not path.exists():
        con.execute("create or replace view runs as select NULL::VARCHAR as run, "
                    "NULL::VARCHAR as label, NULL::DOUBLE as time_ms where false")
        return

    # Check the header before pinning types: naming a column that isn't there in
    # read_csv's types= fails with DuckDB's message rather than ours.
    columns = [c for (c, *_) in con.execute(
        f"describe select * from read_csv({sql_string(path)}, header = true)").fetchall()]
    missing = [c for c in RUNS_REQUIRED if c not in columns]
    if missing:
        sys.exit(f"{path}: missing required column(s) {', '.join(missing)}")
    con.execute(f"""create or replace view runs as select * from read_csv(
        {sql_string(path)}, header = true,
        types = {{'run': 'VARCHAR', 'label': 'VARCHAR', 'time_ms': 'DOUBLE'}})""")

    dupes = con.execute("select run, label, count(*) from runs group by all "
                        "having count(*) > 1 order by 1, 2").fetchall()
    if dupes:
        listed = "\n".join(f"  {run}, {label}: {n} rows" for run, label, n in dupes)
        sys.exit(f"{path}: more than one row per (run, label):\n{listed}")

    (rows, matched) = con.execute("""
        select count(*), count(*) filter (where exists (
            select 1 from profiles p where p.run = r.run and p.label = r.label))
        from runs r""").fetchone()
    if rows and not matched:
        sys.exit(f"{path}: none of its {rows} (run, label) pairs match a profile "
                 f"in the collection")


def latest_run(con: duckdb.DuckDBPyConnection) -> str:
    """The newest run, in session start order, that has stage ranges.

    Not just the newest run: one taken without the NVTX convention has nothing
    for the stage views to say.
    """
    row = con.sql("""select run from run_order
                     where run in (select run from stages)
                     order by run_index desc limit 1""").fetchone()
    if row is None:
        sys.exit("no run in the collection has NVTX stage ranges")
    return row[0]


def filters(**kw) -> tuple[str, dict]:
    """Predicates and bind params for the keyword arguments that aren't None.

    Lets a question say `RANK = None` for "every rank" without growing a branch:

        where, params = filters(run=RUN, label=LABEL, rank=RANK)
        con.sql(f"select ... where {where}", params=params)

    Values go through $name binds, never through the f-string.
    """
    given = {k: v for k, v in kw.items() if v is not None}
    return " and ".join(f"{k} = ${k}" for k in given) or "true", given


def in_reps(table: str, reps: list[int]) -> str:
    """A predicate keeping only the rows of `table` that start inside one of the
    listed reps.

        con.sql(f"select ... from kernels where {in_reps('kernels', [0, 1])}")

    Drops warmup and everything outside the timed loop. The join is on start_ns
    against the host-side `rep N` range, so a launch queued just before the range
    closes but run after it still counts.
    """
    if not reps:
        return "false"
    listed = ", ".join(str(int(r)) for r in reps)
    return f"""exists (
        select 1 from iterations i
        where not i.is_warmup and i.bucket_index in ({listed})
          and i.run = {table}.run and i.label = {table}.label
          and i.rank = {table}.rank
          and {table}.start_ns >= i.start_ns and {table}.start_ns < i.end_ns)"""


def show(table: Table, markdown: bool | None = None) -> None:
    """Print a question's table under its caption.

    markdown defaults to the NVSTITCH_MD environment variable: output format is a
    property of where the result is going, not of the question asked.
    """
    if markdown is None:
        markdown = bool(os.environ.get("NVSTITCH_MD"))
    df, fmt = table.df, table.fmt
    print(f"## {table.caption}")
    if markdown:
        print(df.to_markdown(index=False,
                             floatfmt=[fmt.get(c, "g") for c in df.columns]))
    else:
        # pandas right-justifies both headers and values, which reads badly for
        # a column of names. Pad those cells *and* their header out to the
        # column width instead: a full-width string is unaffected by
        # justification, so it lands flush left.
        #
        # justify="left" is not the fix -- it also left-aligns any numeric
        # column that has no format spec. Nor is `dtype == object`: pandas 3.0
        # gives string columns a `str` dtype, so that check matches nothing.
        out, specs = df.copy(), {c: ("{:" + s + "}").format for c, s in fmt.items()}
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            w = max(len(str(c)), *(len(str(v)) for v in df[c]))
            out[c] = [str(v).ljust(w) for v in df[c]]
            out = out.rename(columns={c: str(c).ljust(w)})
            specs.pop(c, None)
        print(out.to_string(index=False, formatters=specs))
    print()
