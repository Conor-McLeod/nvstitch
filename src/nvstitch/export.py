"""Export every out-of-date .nsys-rep in a collection to a sibling .sqlite.

.nsys-rep is a closed binary format that only nsys itself can read, so this is
the unavoidable first step before any SQL touches a profile. A collection that
already holds up-to-date .sqlite files needs no nsys at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def is_stale(rep: Path) -> bool:
    """True if rep's .sqlite is missing or not newer than rep."""
    db = rep.with_suffix(".sqlite")
    return not db.exists() or db.stat().st_mtime <= rep.stat().st_mtime


def export_one(nsys: str, rep: Path) -> None:
    """Export rep, replacing its .sqlite only once the export has succeeded.

    The temporary directory is what keeps a half-written database from looking
    up to date, and it is nested so that the build's <run>/*.sqlite glob cannot
    pick up a file left behind by a killed export.

    nsys draws a progress bar on stderr, which is also where a real failure is
    reported, so stderr is kept and shown only if the export fails.
    """
    with tempfile.TemporaryDirectory(dir=rep.parent, prefix=".export-") as tmp:
        out = Path(tmp) / f"{rep.stem}.sqlite"
        result = subprocess.run(
            [nsys, "export", "--type", "sqlite", "--force-overwrite=true",
             "-o", str(out), str(rep)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, errors="replace")
        if result.returncode != 0:
            sys.exit(f"export failed for {rep}:\n{result.stderr}")
        os.replace(out, rep.with_suffix(".sqlite"))


def export_profiles(collection: Path, jobs: int = 1) -> None:
    reps = sorted(collection.glob("*/*.nsys-rep"))
    stale = [rep for rep in reps if is_stale(rep)]
    if not stale:
        return

    nsys = os.environ.get("NSYS", "nsys")
    if shutil.which(nsys) is None:
        sys.exit(f"{len(stale)} profile(s) need exporting, but nsys was not found "
                 f"(tried '{nsys}'). Put it on PATH or set NSYS to the binary.")

    def export(rep: Path) -> None:
        print(f"exporting {rep.parent.name}/{rep.name}")
        export_one(nsys, rep)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(export, stale))
    print(f"exported {len(stale)}, up to date {len(reps) - len(stale)}")
