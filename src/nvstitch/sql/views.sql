-- The view layer over a collection's fact tables.
--
-- connect() runs this with file_search_path set to the collection, so the
-- relative paths below resolve to <collection>/facts/. Declarative only: no
-- parameters and no formatting. A query that needs a run, a rank or a row SQL
-- cannot produce belongs in a question, not here.
--
-- Everything from `iterations` down relies on the NVTX naming convention in
-- docs/design.md: one `warmup N` or `rep N` range around each iteration, and
-- numbered stage ranges named `<n> <stage>` nested inside it. Profiles that
-- don't follow it simply have no rows in those views.
--
-- An application can layer its own views on top (connect's extra_sql). Views are
-- rebound when queried, so redefining `kernels` with an extra column also carries
-- that column through `kernel_stages`.

-- ------------------------------------------------------------ fact tables
-- One row per event, keyed by (run, label, rank). These are PROFILED timings.

create or replace view profiles as
    select * from read_parquet('facts/profiles.parquet');
create or replace view nvtx as
    select * from read_parquet('facts/nvtx.parquet');
create or replace view kernels as
    select * from read_parquet('facts/kernels.parquet');
create or replace view memops as
    select * from read_parquet('facts/memops.parquet');
create or replace view memcpy as
    select * from memops where kind = 'memcpy';
create or replace view mpi as
    select * from read_parquet('facts/mpi.parquet');
-- One row per GPU metrics sample, each rank sampling only its own GPU. Empty
-- for profiles taken without --gpu-metrics-devices. See v_gpu_metrics in
-- extract.sql for what was dropped and why there is no power column.
create or replace view gpu_metrics as
    select * from read_parquet('facts/gpu_metrics.parquet');

-- One row per run, in the order the runs were taken. Run names are arbitrary
-- and need not sort, so "newest" and "the run before" come from here: the
-- earliest session start among a run's profiles, ties broken by name.
create or replace view run_order as
    select run,
           min(utc_epoch_ns)                                  as started_ns,
           row_number() over (order by min(utc_epoch_ns), run) as run_index
    from profiles
    group by run;

-- ------------------------------------------------------------ iterations
-- One row per iteration: the `warmup N` / `rep N` ranges.
--
-- The view to interval-join kernels, memops or mpi events against when the
-- question is "which iteration was this in". It needs no stage ranges, so a
-- label that pushes iterations but no stages still has rows here.
create or replace view iterations as
    select run, label, rank,
           name                                               as bucket,
           name like 'warmup%'                                as is_warmup,
           try_cast(regexp_extract(name, '(\d+)$') as integer) as bucket_index,
           dur_ms, start_ns, end_ns, abs_start_ns, tid
    from nvtx
    where name like 'rep %' or name like 'warmup %';

-- Stage durations bucketed into the iteration they belong to.
--
-- Stage ranges nest inside whichever iteration range is open, so an interval
-- join recovers which iteration each stage belongs to.
--
-- is_warmup is a column rather than a filter on purpose. Warmup carries all the
-- one-time cost (library handles, first allocations), so it can be orders of
-- magnitude slower than the reps. Dropping those rows hides exactly the row you
-- want when asking why a first call is slow, and averaging them in is just as
-- wrong. So: label here, filter at the call site.
--
-- The columns, since none of them is self-describing:
--   dur_ms       host wall time in the range, (end_ns - start_ns) / 1e6
--   start_ns     ns from that rank's own nsys session start. Each rank's zero is
--                its own, so these are NOT comparable across ranks.
--   abs_start_ns start_ns + the session's UTC epoch, i.e. ns since the Unix
--                epoch. This is the one that is comparable across ranks.
--   tid          the thread (globalTid & 0xffffff) that pushed the range.
create or replace view stages as
    select n.run, n.label, n.rank,
           b.name                                             as bucket,
           b.name like 'warmup%'                              as is_warmup,
           try_cast(regexp_extract(b.name, '(\d+)$') as integer) as bucket_index,
           -- Drop the leading ordinal so that the same stage groups together
           -- even when variants of a code number it differently.
           regexp_extract(n.name, '^\d+ (.*)$', 1)            as stage,
           n.dur_ms, n.start_ns, n.end_ns, n.abs_start_ns,
           -- Carried so stage_gpu can match a kernel's launching thread. A stage
           -- range and the launches inside it are always on one thread; the GPU
           -- work they submit is not.
           n.tid
    from nvtx n
    join nvtx b using (run, label, rank)
    where (b.name like 'rep %' or b.name like 'warmup %')
      and n.start_ns >= b.start_ns
      and n.start_ns <  b.end_ns
      and regexp_matches(n.name, '^\d+ ');

-- Per-stage spread across the ranks of one run, in the steady-state reps.
--
-- abs_start_ns is what makes entry_skew_ns meaningful: each rank's raw start_ns
-- is relative to its own process start, so comparing raw offsets is nonsense.
create or replace view rank_spread as
    select run, label, stage,
           count(*)                                   as n,
           round(min(dur_ms), 3)                      as min_ms,
           round(max(dur_ms), 3)                      as max_ms,
           round(max(dur_ms) - min(dur_ms), 3)        as spread_ms,
           round(avg(dur_ms), 3)                      as avg_ms,
           -- How far apart the ranks enter this stage.
           max(abs_start_ns) - min(abs_start_ns)      as entry_skew_ns
    from stages
    where not is_warmup
    group by run, label, stage
    order by run, label, spread_ms desc;

-- Every kernel launch, tagged with the stage that submitted it.
--
-- A kernel is charged to the stage whose range contained its *launch*, not the
-- one open while it ran. That is the only defensible attribution once anything
-- is async: the kernel is the consequence of a launch that a stage made, and it
-- may well execute after that stage has been popped.
--
-- Inner join, so a launch made outside every numbered stage has no row here. For
-- those, read `kernels` directly. All of the kernel's own columns are carried,
-- including any an application adds by redefining `kernels`.
create or replace view kernel_stages as
    select s.run, s.label, s.rank, s.bucket, s.is_warmup, s.bucket_index, s.stage,
           k.* exclude (run, label, rank)
    from stages s
    join kernels k using (run, label, rank)
    where k.launch_tid = s.tid
      and k.launch_ns >= s.start_ns
      and k.launch_ns <  s.end_ns;

-- GPU work attributed to the stage that submitted it, per the rule in
-- kernel_stages.
--
-- Three different numbers, because they answer three different questions:
--   kernel_ms   sum of kernel durations -- total GPU work the stage caused
--   busy_ms     union of those intervals -- wall time the GPU was non-idle,
--               so kernel_ms - busy_ms is how much the stage overlapped with
--               itself across streams
--   memops_ms   copies and sets the stage submitted, excluded from the above
--
-- pct_of_kernels is a share of all the kernel time the iteration's stages
-- launched, so the column sums to 100 across the stages of one iteration.
--
-- Caveat that survives every aggregation here: NCCL device kernels spin while
-- waiting for peers, so their duration folds rank skew in with transfer. One
-- rank's profile cannot separate the two.
create or replace view stage_gpu as
    with launched as (
        select * from kernel_stages
    ),
    -- Gaps and islands: a kernel starts a new island when it begins after every
    -- earlier kernel of the same stage has ended. Two CTEs because DuckDB will
    -- not nest one window call inside another.
    starts as (
        select *,
               case when start_ns <= max(end_ns) over (
                        partition by run, label, rank, bucket, stage
                        order by start_ns
                        rows between unbounded preceding and 1 preceding)
                    then 0 else 1 end                         as new_island
        from launched
    ),
    islands as (
        select *,
               sum(new_island) over (
                   partition by run, label, rank, bucket, stage
                   order by start_ns)                         as island
        from starts
    ),
    busy as (
        select run, label, rank, bucket, stage,
               sum(span_ns) / 1e6                             as busy_ms
        from (select run, label, rank, bucket, stage,
                     max(end_ns) - min(start_ns)              as span_ns
              from islands
              group by run, label, rank, bucket, stage, island)
        group by run, label, rank, bucket, stage
    ),
    mem as (
        select s.run, s.label, s.rank, s.bucket, s.stage,
               count(*)                                       as memops,
               sum(m.dur_us) / 1000                           as memops_ms
        from stages s
        join memops m using (run, label, rank)
        where m.launch_tid = s.tid
          and m.launch_ns >= s.start_ns
          and m.launch_ns <  s.end_ns
        group by 1, 2, 3, 4, 5
    )
    select l.run, l.label, l.rank, l.bucket, l.is_warmup, l.bucket_index,
           l.stage,
           count(*)                                           as launches,
           round(sum(l.dur_us) / 1000, 3)                     as kernel_ms,
           round(any_value(b.busy_ms), 3)                     as busy_ms,
           coalesce(any_value(m.memops), 0)                   as memops,
           round(coalesce(any_value(m.memops_ms), 0), 3)      as memops_ms,
           round(100 * sum(l.dur_us) / sum(sum(l.dur_us)) over (
                     partition by l.run, l.label, l.rank, l.bucket), 2)
                                                              as pct_of_kernels
    from launched l
    join busy b using (run, label, rank, bucket, stage)
    left join mem m using (run, label, rank, bucket, stage)
    group by l.run, l.label, l.rank, l.bucket, l.is_warmup,
             l.bucket_index, l.stage
    order by l.run, l.label, l.rank, l.bucket_index, l.stage;
