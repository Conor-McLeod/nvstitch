# nvstitch

Stitch per-rank [Nsight Systems](https://developer.nvidia.com/nsight-systems)
profiles into one queryable set of tables, and get a readable report out of them.

**Status: early.** `build` and `ask` work, with two questions so far, and on a
real 12-profile job they give exactly the same answers as the par_gemmul8 code they
replace. `report` is not implemented yet. The agreed scope and
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
nvstitch -C <collection> ask
```

```bash
nvstitch -C <collection> ask stage_breakdown --run <run>
```

Getting the profiles into that folder, and running the jobs that produce them,
is deliberately up to you.

## nsys on macOS

The macOS Nsight Systems app includes the GUI but no `nsys` command, so `build`
can't export reports there on its own. Either export the `.sqlite` files on a
machine that has `nsys` and copy them in, or run the Linux `nsys` that the app
ships with in an x86-64 container. [docs/design.md](docs/design.md#exporting-on-macos)
has the wrapper script.
