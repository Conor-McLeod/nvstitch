# nvstitch design

nvstitch grows out of the `analysis/` directory of par_gemmul8. That code works
well for its author but assumes one cluster, one program and one workflow. The
goal here is a tool someone else can pick up without learning all of that.

Implemented and checked against real data (see [Verification](#verification)):
`nvstitch build`, the views, and two questions, `stage_breakdown` and `north_star`,
through `nvstitch ask`. `report` and the remaining questions are still to be
built.

## Scope

**nvstitch starts at a local folder of nsys profiles and ends at answers.**

| In scope | Out of scope (the user's job) |
|---|---|
| Finding profiles in a folder | Getting them there (rsync, cluster access) |
| Exporting `.nsys-rep` to `.sqlite`, or accepting `.sqlite` directly | Running jobs (SLURM, campaigns) |
| Stitching ranks and runs into fact tables | Locating `nsys` beyond `PATH` or `NSYS` |
| Integrity checks (rank mismatch, mixed nsys schemas) | Parsing an application's own logs |
| Views, questions and the report | Application-specific tuning workflows |

The integrity checks are a feature, not plumbing: refusing to mix schema versions
or disagreeing ranks is much of why the numbers can be trusted.

## Input contract

```
<collection>/
  <run>/                       one directory per run, any name
    <label>_rank<n>.nsys-rep   or .sqlite
  runs.csv                     optional
```

- A single-rank profile with no `_rank<n>` suffix must work.
- The rank in `MPI_RANKS` is cross-checked against the filename when both exist.
- "Newest run" orders by the profile's session start time, not by run name. Run
  names are arbitrary, so they cannot be assumed to sort.
- Derived data lives in the collection (`<collection>/facts/`, and `.sqlite` next
  to its `.nsys-rep`). This repository holds no data.
- The fact tables identify a profile by `run`, `label` and `rank`.
- nsys writes a table only when it has content, so a profile may lack MPI, NVTX or
  activity tables. Those fact tables are then simply empty for that profile.
- Which GPU metrics exist depends on the GPU. Some empty metric columns produce a
  warning; all of them empty is an error, as it means the nsys names have changed.

## The optional runs table

`<collection>/runs.csv` holds what a profile cannot contain: how long the run took
with no profiler attached, and what the run was.

```csv
run,label,time_ms,ozaki_fused,m,n,k,num_moduli
1425521,run_par_ozaki,14.144,1,12000,12000,12000,8
1425521,run_par_ozaki_async,15.952,1,12000,12000,12000,8
```

| Column | Meaning |
|---|---|
| `run` | Must match a directory name in the collection |
| `label` | Must match the label in the profile filename |
| `time_ms` | Wall time of the same workload with no profiler; the column is required, a value may be blank |
| anything else | Free-form run metadata, carried along without interpretation |

Rules:

- One row per `(run, label)`. Duplicates are an error that lists the rows;
  summarising repetitions is the user's decision.
- No matching rows at all is an error. Partial matches are reported, not fatal.
- `run` and `label` are always read as text.
- The file is read at query time, so editing it needs no rebuild.

Without the file, every time is labelled as profiled, with a warning against
quoting it as performance. With it, unprofiled and profiled times are shown in
separate columns and are never rescaled to fit each other. `report --vs` lists
the metadata columns that differ between the two runs.

Producing this file is an adapter's job, in whatever language suits. nvstitch does
not load parser plugins.

## Tiers of generality

| Tier | Needs | Content |
|---|---|---|
| 1 | Any nsys profile | Kernel hotspots, kernel launches, NCCL time |
| 2 | The NVTX naming convention | Stage breakdowns, rank skew, kernel timeline, gating-rank summary |
| 3 | One specific application | Stays in that application's repository |

The convention for tier 2 is NVTX ranges named `warmup N`, `rep N` and
`<n> <stage>`. It is optional: the report shows tier 1 for any profile and adds
tier 2 when the names match.

This split comes from a keyword scan of which views each par_gemmul8 script reads.
Check each script when porting it.

## Command line

```
nvstitch [-C DIR] build
nvstitch [-C DIR] report [RUN] [--vs RUN]
nvstitch [-C DIR] ask [QUESTION] [--run RUN] [--md]
```

`-C` selects the collection and defaults to the current directory.

Questions stay as one file each, with their parameters as literals at the top.
Earlier argparse subcommands in par_gemmul8 were dropped because flags carry only
scalars, while a question may want a list of ranks or two jobs to compare. Each
question has an `answer(con, ...)` function that takes those literals as keyword
defaults and returns a list of `Table`s (caption, frame and column formats), so
`ask`, `report` and Python callers all share it. `ask` passes only `--run`; for
any other argument, call `answer()` from Python.

## Packaging

- SQL files ship inside the package and load through `importlib.resources`.
- Install from a git URL for now; PyPI can wait.
- The public Python API is `connect`, `latest_run`, `filters`, `in_reps`, `show`
  and `Table`, plus each question's `answer()`. Application repositories build on
  these, so they need to stay stable.
- `connect()` takes a collection path and can load extra SQL, so an application
  can layer its own views on top.
- A `questions_dir` setting lets `ask` find application-specific questions.
- DuckDB is capped below 2 until the port is verified, to stay on the same 1.x
  line as par_gemmul8's `analysis/` (locked to 1.5.5).

## Verification

On 2026-11-09, `nvstitch build` and par_gemmul8's `analysis/build.py` were run on
the same `.sqlite` exports of job 2160148: 12 profiles (three methods, four ranks)
on GH200, with GPU metrics. All six fact tables matched, with the same rows (1.5M
in `gpu_metrics`) and the same column names, order and types, once `job_id` and
`method` are renamed to `run` and `label`. The only value that differs is the
label, which keeps the filename's `run_` prefix (`run_par_ozaki`) where
par_gemmul8 strips it.

Not covered: the exports were made with nsys 2026.2.1 (export schema 3.25.0),
while par_gemmul8's profiles were taken and previously exported with 2025.5.1; the
two builds ran on DuckDB 1.5.6 and 1.5.5.

On 2026-11-10 the same comparison was made for the questions, running
par_gemmul8's own question scripts unmodified apart from their literals. Every
table matched exactly after the same renames: `stage_breakdown` (with
`CALL="%gemm"`) for both staged methods, all four ranks, with and without the
warmup, and `north_star` (with `META=["peer_copy"]`). Since job 2160148 alone gives
`north_star` no previous run, it was also compared on that job plus a synthetic
successor, a copy a day later with every stage range 10% longer and `peer_copy`
flipped, built identically into both sides' tables. Both reported the same 10%
deltas.

## Exporting on macOS

The macOS Nsight Systems app ships the GUI (`nsys-ui`) but no native `nsys`
command. Its `Contents/target-linux-x64/nsys` is a Linux x86-64 build, which runs
in an x86-64 container (tested with OrbStack). `nsys` refuses to run unless its
folder is named `target-linux-x64`, and the collection must be mounted at the same
absolute path inside the container so that the paths nvstitch passes still
resolve:

```bash
#!/bin/bash
C=/absolute/path/to/collection
exec docker run --rm --platform linux/amd64 \
  -v "/Applications/NVIDIA Nsight Systems.app/Contents/target-linux-x64:/opt/target-linux-x64:ro" \
  -v "$C:$C" ubuntu:24.04 /opt/target-linux-x64/nsys "$@"
```

Pointing `NSYS` at this script lets `nvstitch build` export as usual. Under
emulation, 12 reports took about four minutes with `-j 3`.

## Porting from par_gemmul8

**Done:** `build.py`, `extract.sql`, the export step (replacing `make` and
`export.sh`), the generic views from `setup.sql`, the `common.py` helpers, and the
questions `stage_breakdown` and `north_star`.

**Still to move here:** the other tier 1 and 2 questions, and `plots/clock.py`.

**Stays in par_gemmul8/analysis:** `sync.sh`, `ingest.py` (which becomes the
adapter writing `runs.csv`, adding the `run_` label prefix), the views `clean`,
`reported_stages`, `timeline`, `profiler_overhead`, `gemm_calls` and `stage_wall`,
`fused_tune.py`, `timing_vs_nvtx`, `stage3_*` and `sweeps.py`.

**Known changes needed while porting:**

- Ported views and questions must use `run` and `label`, where par_gemmul8 has
  `job_id` and `method`.
- The `OzakiGemmModFusion` parsing in the `kernels` view moves to par_gemmul8.

**What par_gemmul8 needs to switch over:** `ingest.py` writing the runs table; its
own SQL passed to `connect(extra_sql=...)`, holding the views listed above and the
`mod_idx` column on `kernels`; and its settings for the generic questions:
`stage_breakdown` with `CALL="%gemm"`, and `north_star` with
`EXCLUDE=["1703852"]` and `META=["peer_copy"]`.

**Retiring `analysis/`:** the agreed bar, nvstitch reproducing its `north_star`
and `stage_breakdown` output on the same profiles, is met (see Verification). It
still holds questions nvstitch doesn't have yet, so keep it until those are
ported or moved into par_gemmul8's own question set.

## Open items

- A small committed sample collection and `nvstitch demo`, so a new user sees real
  output with no cluster or `nsys`. nsys profiles can carry hostnames and paths,
  so inspect the sample before committing it.
- `doctor`-style guidance when setup is incomplete.
- A marimo app on top of the same `run()` functions, after `report` exists.
- On macOS, the "nsys was not found" error should point to the container route
  rather than telling the user to put `nsys` on `PATH`.
