"""What one iteration is made of, wall-clock and on GPU.

Both tables cover every iteration in the profile, warmup included, because the
comparison between them is the point: a warmup can take a hundred times the wall
time of a rep while doing the same GPU work. Set REPS_ONLY once you have looked.

The wall table's rows need not tile what they are a share of, so it carries an
explicit residual: the time no stage was open for. By default that is a share of
the whole iteration range. Set CALL to an NVTX range nested in each iteration, as
a LIKE pattern, to break that range down instead: "%gemm" is the GEMM call in
par_gemmul8, which leaves the barriers around it out of the total.

The GPU table charges a kernel to the stage whose range contained its launch,
not the one open while it ran -- the only defensible rule once anything is
async. Its pct does tile to 100.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from nvstitch.query import Table, latest_run

RUN = None          # None -> newest run with stage ranges
LABEL = None        # None -> the first label, alphabetically, with stage ranges
RANK = 0
CALL = None         # None -> the iteration; or a LIKE pattern, e.g. "%gemm"
REPS_ONLY = False   # True drops the warmup iteration


def answer(con: duckdb.DuckDBPyConnection, *, run: str | None = RUN,
           label: str | None = LABEL, rank: int = RANK, call: str | None = CALL,
           reps_only: bool = REPS_ONLY) -> list[Table]:
    run = run or latest_run(con)
    if label is None:
        row = con.sql("select min(label) from stages where run = $run",
                      params={"run": run}).fetchone()
        label = row[0]
        if label is None:
            raise SystemExit(f"no stage ranges in run {run}")
    params = {"run": run, "label": label, "rank": rank}
    warmup = "and not s.is_warmup" if reps_only else ""

    if call is None:
        totals = "select run, label, rank, bucket, dur_ms as total_ms from iterations"
    else:
        totals = """select b.run, b.label, b.rank, b.bucket, n.dur_ms as total_ms
                    from nvtx n join iterations b using (run, label, rank)
                    where n.start_ns >= b.start_ns and n.start_ns < b.end_ns
                      and n.name like $call"""
        params["call"] = call

    # dur_ms unrounded: the residual is a difference of two totals, so rounding
    # before subtracting shifts it in the last digit.
    wall = con.sql(f"""
        with totals as ({totals})
        select s.bucket, s.stage, s.dur_ms as ms,
               round(100 * s.dur_ms / t.total_ms, 2) as pct, t.total_ms
        from stages s join totals t using (run, label, rank, bucket)
        where s.run = $run and s.label = $label and s.rank = $rank {warmup}
        order by s.is_warmup desc, s.bucket_index, s.stage
    """, params=params).df()
    if wall.empty:
        raise SystemExit(f"no stage ranges for run {run} / {label} / rank {rank}"
                         + (f" inside a range like {call!r}" if call else ""))

    # Residual and total per bucket. These are the rows SQL cannot produce
    # without a UNION ALL and a synthetic sort key.
    total_name = f"TOTAL ({call})" if call else "TOTAL (iteration)"
    parts = []
    for bucket, g in wall.groupby("bucket", sort=False):
        total_ms = g.total_ms.iloc[0]
        residual = total_ms - g.ms.sum()
        parts.append(g[["bucket", "stage", "ms", "pct"]])
        parts.append(pd.DataFrame([
            {"bucket": bucket, "stage": "(unaccounted)",
             "ms": residual, "pct": 100 * residual / total_ms},
            {"bucket": bucket, "stage": total_name, "ms": total_ms, "pct": 100.0},
        ]))

    gpu = con.sql(f"""
        select bucket, stage, launches, kernel_ms, busy_ms,
               memops, memops_ms, pct_of_kernels as pct
        from stage_gpu s
        where run = $run and label = $label and rank = $rank {warmup}
        order by is_warmup desc, bucket_index, stage
    """, params={"run": run, "label": label, "rank": rank}).df()

    where = f"{run} / {label} / rank {rank}"
    return [
        Table(f"{where} -- wall clock", pd.concat(parts, ignore_index=True),
              {"ms": ".3f", "pct": ".2f"}),
        Table(f"{where} -- GPU, by launching stage", gpu,
              {"kernel_ms": ".3f", "busy_ms": ".3f", "memops_ms": ".3f", "pct": ".2f"}),
    ]
