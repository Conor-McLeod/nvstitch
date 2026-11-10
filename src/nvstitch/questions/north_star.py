"""The one number: stage time on the rank that gates the run, newest run first.

For each rep, sum the numbered NVTX stage ranges per rank and keep the rank with
the largest sum; average that over the reps of one run. That is one run's north
star. The first table is the newest run against the one before it, the second is
every run newest first, and the third says which stage moved between the two.

NOT AVERAGED ACROSS RUNS. Each run is typically a different build or setting, so
a mean over runs mixes the change being measured in with run-to-run noise and
reports neither. The comparison that answers "did it get better" is the newest
run against its predecessor, which is what `delta` is.

HOW BIG A DELTA IS REAL. `rep_spread` is the gap between the fastest and slowest
rep of that run, the only in-run noise estimate available. Treat a delta smaller
than it as unmeasured.

A DELTA IS BETWEEN TWO RUNS, NOT TWO COMMITS. Whatever else changed between them
is in it. List runs-table columns in META to carry them alongside, so a setting
that changed between the two runs is the first thing you see.

THE GATING RANK, NOT THE MEAN OVER RANKS. A collective is over when its slowest
rank is, so the mean is not a time anything takes. `imbalance` is what the mean
would discount. Which rank gates it can move between runs and between reps
(`gated_by`), so this is not a per-rank filter.

WHAT IT LEAVES OUT. Unless the stages tile the iteration, this is not a wall
time: it tracks the stage ranges only. For wall time, use an unprofiled timing
from the runs table.

The third table tiles exactly: each rep contributes one rank's stage rows, so a
label's rows sum to its north star. Labels need not share a stage list, so
compare the sums across labels, never the rows.

Warmup is excluded: its one-time costs would swamp any average containing it.
Use stage_breakdown for the warmup.

`prev` is the previous run THIS TABLE HAS, in session start order: a run with no
stage ranges, or one in EXCLUDE, is skipped silently and the delta reaches back
past it. A delta between runs of different problem sizes means nothing.
"""

from __future__ import annotations

import duckdb

from nvstitch.query import Table

RUN = None          # None -> up to the newest run; or the run to treat as newest
LABEL = None        # None -> every label
EXCLUDE = []        # runs whose stage numbers aren't measurements of interest
META = []           # runs-table columns to show alongside, e.g. ["peer_copy"]


def answer(con: duckdb.DuckDBPyConnection, *, run: str | None = RUN,
           label: str | None = LABEL, exclude: list[str] = EXCLUDE,
           meta: list[str] = META) -> list[Table]:
    where, params = ["true"], {}
    if run:
        if con.sql("select 1 from run_order where run = $run",
                   params={"run": run}).fetchone() is None:
            raise SystemExit(f"no run {run!r} in the collection")
        where.append("o.run_index <= (select run_index from run_order where run = $run)")
        params["run"] = run
    if label:
        where.append("s.label = $label")
        params["label"] = label
    if exclude:
        where.append("s.run not in (select unnest($exclude))")
        params["exclude"] = list(exclude)

    columns = {c for (c, *_) in con.execute("describe runs").fetchall()}
    unknown = [c for c in meta if c not in columns]
    if unknown:
        raise SystemExit(f"META names column(s) the runs table doesn't have: "
                         f"{', '.join(unknown)}")
    meta_cols = "".join(f', "{c}"' for c in meta)
    meta_sql = "".join(f', c."{c}"' for c in meta)
    meta_join = (f"left join (select run, label{meta_cols} from runs) c "
                 f"using (run, label)" if meta else "")

    # One row per run x label: the north star, plus the within-run rep spread
    # that says how big a delta has to be before it is one.
    con.execute(f"""
        create or replace temp table history as
        with per_rank as (
            select s.run, s.label, s.rank, s.bucket, sum(s.dur_ms) as stages_ms
            from stages s join run_order o using (run)
            where not s.is_warmup and {" and ".join(where)}
            group by 1, 2, 3, 4),
        gating as (
            select run, label, bucket,
                   max(stages_ms)           as gating_ms,
                   arg_max(rank, stages_ms) as gating_rank,
                   avg(stages_ms)           as mean_ms
            from per_rank
            group by 1, 2, 3),
        per_run as (
            select g.run, g.label, any_value(o.run_index) as run_index,
                   avg(gating_ms)                        as north_star,
                   max(gating_ms) - min(gating_ms)       as rep_spread,
                   avg(gating_ms - mean_ms)              as imbalance,
                   string_agg(distinct cast(gating_rank as varchar), ',') as gated_by,
                   arg_max(gating_rank, gating_ms)       as slowest_rank
            from gating g join run_order o using (run)
            group by 1, 2)
        select j.*,
               lag(north_star) over (partition by label order by run_index) as prev,
               row_number() over (partition by label order by run_index desc) as age
               {meta_sql}
        from per_run j
        -- Left join: a run with no row in the runs table still has a north
        -- star, just no metadata.
        {meta_join}
    """, params)

    if con.sql("select count(*) from history").fetchone()[0] == 0:
        raise SystemExit("no stage ranges match -- check the arguments, "
                         "or whether the profiles follow the NVTX convention")

    latest = con.sql(f"""
        select label, run,
               round(north_star, 3)                         as north_star,
               round(rep_spread, 3)                         as rep_spread,
               round(prev, 3)                               as prev,
               round(north_star - prev, 3)                  as delta,
               round(100 * (north_star - prev) / prev, 2)   as pct,
               slowest_rank{meta_cols}
        from history where age = 1
        order by north_star
    """).df()

    history = con.sql(f"""
        select run, label,
               round(north_star, 3)        as north_star,
               round(north_star - prev, 3) as delta,
               round(rep_spread, 3)        as rep_spread,
               round(imbalance, 3)         as imbalance,
               gated_by{meta_cols}
        from history
        order by run_index desc, label
    """).df()

    # Which stage moved. Restricted to the gating rank of each rep, so the rows
    # still sum to the north star rather than to some other rank's total.
    moved = con.sql("""
        with edge as (select label, run, age from history where age <= 2),
        split as (
            select e.label, e.age, s.stage, avg(s.dur_ms) as ms
            from edge e
            join stages s using (run, label)
            join (select run, label, bucket,
                         arg_max(rank, stages_ms) as gating_rank
                  from (select run, label, rank, bucket,
                               sum(dur_ms) as stages_ms
                        from stages where not is_warmup group by 1, 2, 3, 4)
                  group by 1, 2, 3) g
              on  g.run = s.run and g.label = s.label
              and g.bucket = s.bucket and g.gating_rank = s.rank
            where not s.is_warmup
            group by 1, 2, 3)
        select label, stage,
               round(max(ms) filter (age = 1), 3)                            as latest,
               round(max(ms) filter (age = 2), 3)                            as prev,
               round(max(ms) filter (age = 1) - max(ms) filter (age = 2), 3) as delta
        from split
        group by 1, 2
        order by label, latest desc
    """).df()

    return [
        Table("latest run -- where we are, and whether it moved", latest,
              {"north_star": ".3f", "rep_spread": ".3f", "prev": ".3f",
               "delta": ".3f", "pct": ".2f"}),
        Table("history, newest first", history,
              {"north_star": ".3f", "delta": ".3f", "rep_spread": ".3f",
               "imbalance": ".3f"}),
        Table("which stage moved, on the gating rank", moved,
              {"latest": ".3f", "prev": ".3f", "delta": ".3f"}),
    ]
