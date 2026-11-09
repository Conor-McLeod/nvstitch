# nvstitch

Stitch per-rank [Nsight Systems](https://developer.nvidia.com/nsight-systems)
profiles into one queryable set of tables, and get a readable report out of them.

**Status: early.** `build` is implemented but has only been tested on synthetic
profiles so far. `report` and `ask` are not implemented yet. The agreed scope and
contracts are in [docs/design.md](docs/design.md).

nvstitch is an independent project built on `nsys` output. It is not affiliated
with or endorsed by NVIDIA.

## What it does

You collect your profiles into a local folder. nvstitch takes it from there.

```
<collection>/
  <run>/                       one directory per run, any name
    <label>_rank<n>.nsys-rep   or .sqlite
  runs.csv                     optional: unprofiled timings and run metadata
```

```bash
nvstitch -C <collection> build
```

```bash
nvstitch -C <collection> report
```

```bash
nvstitch -C <collection> ask
```

Getting the profiles into that folder, and running the jobs that produce them,
is deliberately up to you.
