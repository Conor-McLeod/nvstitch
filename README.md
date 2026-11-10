# nvstitch

Stitch per-rank [Nsight Systems](https://developer.nvidia.com/nsight-systems)
profiles into one queryable set of tables, and get a readable report out of them.

**Status: early.** `build` is implemented, and on a real 12-profile job it produced
exactly the same tables as the par_gemmul8 code it replaces. `report` and `ask` are
not implemented yet. The agreed scope and
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

## nsys on macOS

The macOS Nsight Systems app includes the GUI but no `nsys` command, so `build`
can't export reports there on its own. Either export the `.sqlite` files on a
machine that has `nsys` and copy them in, or run the Linux `nsys` that the app
ships with in an x86-64 container. [docs/design.md](docs/design.md#exporting-on-macos)
has the wrapper script.
