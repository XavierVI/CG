#!/usr/bin/env python
"""
Ingest CG solver performance data into a DuckDB database.

Each solver run produces two artifacts:
  - a Caliper ``.cali`` file (runtime-profile service) holding per-rank MPI
    function timings/message stats plus run metadata in the Caliper globals
  - a ``.json`` file describing the process communication topology
    (per-rank hostname, node placement, send/recv sizes, neighbor lists)

This script reads one (or a directory of) such pairs and appends them to two
tables in a DuckDB database:

  cg_perf_data — one row per Caliper record (per rank, per region/MPI function):
    run_id: varchar                     flux run/submit job id (Caliper global)
    mpi.rank: int
    sum#sum#time.duration: double
    min#aggregate.slot: int
    region: varchar                     comma-joined if Caliper reports a list
    path: varchar                       comma-joined call path
    mpi.function: varchar
    sum#mmc.coll: bigint                collective message count
    min#mms.min: bigint                 message size stats (bytes)
    avg#mms.avg: double
    max#mms.max: bigint
    sum#mmc.send: bigint
    sum#mmc.recv: bigint

  cg_metadata — one row per run:
    run_id: varchar (primary key)
    procs, nodes, ppn: int              ppn derived as procs // nodes
    RDZV: int                           FI_CXI_RDZV_THRESHOLD
    MATCH_MODE: int                     FI_CXI_RX_MATCH_MODE
                                        (hardware = 0, software = 1, hybrid = 2)
    GPU_IPC: int                        MPICH_GPU_IPC_ENABLED
    ASYNC: int                          MPICH_ASYNC_PROGRESS
    sparse_matrix: varchar
    max_iters, converged_iters: int
    relative_residual: double
    proc_topo: json                     full contents of the topology .json

Usage:
    python data.py --cali cg_solver.cali --json cg_solver.json --db cg_runs.duckdb
    python data.py --dir ./results --db cg_runs.duckdb          # every *.cali + matching *.json
    python data.py --cali run.cali --db cg_runs.duckdb --replace  # re-ingest an existing run_id
"""
import argparse
import json
from pathlib import Path

import caliperreader as cr
import duckdb as ddb
import polars as pl


PERF_TABLE = "cg_perf_data"
METADATA_TABLE = "cg_metadata"

# Caliper record key -> polars dtype. Everything caliperreader hands back is a
# string (or list of strings); values are converted with these target types.
# Message-size stats are bytes, so use 64-bit ints; the avg can be fractional.
PERF_COLUMNS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "mpi.rank": pl.Int32,
    "sum#sum#time.duration": pl.Float64,
    "min#aggregate.slot": pl.Int32,
    "region": pl.Utf8,
    "path": pl.Utf8,
    "mpi.function": pl.Utf8,
    "sum#mmc.coll": pl.Int64,
    "min#mms.min": pl.Int64,
    "avg#mms.avg": pl.Float64,
    "max#mms.max": pl.Int64,
    "sum#mmc.send": pl.Int64,
    "sum#mmc.recv": pl.Int64,
}

# FI_CXI_RX_MATCH_MODE is usually exported as an integer already, but map the
# string names too (HPE libfabric encoding).
MATCH_MODE_MAP: dict[str, int] = {
    "hardware": 0,
    "software": 1,
    "hybrid": 2,
}

PERF_DDL = f"""
CREATE TABLE IF NOT EXISTS {PERF_TABLE} (
    run_id VARCHAR,
    "mpi.rank" INTEGER,
    "sum#sum#time.duration" DOUBLE,
    "min#aggregate.slot" INTEGER,
    region VARCHAR,
    path VARCHAR,
    "mpi.function" VARCHAR,
    "sum#mmc.coll" BIGINT,
    "min#mms.min" BIGINT,
    "avg#mms.avg" DOUBLE,
    "max#mms.max" BIGINT,
    "sum#mmc.send" BIGINT,
    "sum#mmc.recv" BIGINT
)
"""

METADATA_DDL = f"""
CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
    run_id VARCHAR PRIMARY KEY,
    procs INTEGER,
    nodes INTEGER,
    ppn INTEGER,
    RDZV INTEGER,
    MATCH_MODE INTEGER,
    GPU_IPC INTEGER,
    ASYNC INTEGER,
    sparse_matrix VARCHAR,
    max_iters INTEGER,
    converged_iters INTEGER,
    relative_residual DOUBLE,
    proc_topo JSON
)
"""


def _convert(value, dtype: pl.DataType):
    """Convert a raw caliperreader value (string / list of strings) to dtype."""
    if value is None:
        return None
    if isinstance(value, list):
        value = ",".join(str(v) for v in value)
    if dtype == pl.Utf8:
        return str(value)
    if dtype == pl.Float64:
        return float(value)
    return int(value)


def read_cali(cali_path: str | Path) -> tuple[pl.DataFrame, dict]:
    """
    Read a Caliper file into (perf_df, globals).

    perf_df has one row per Caliper record with the PERF_COLUMNS schema
    (run_id included on every row); globals is the raw Caliper globals dict.
    """
    reader = cr.CaliperReader()
    reader.read(str(cali_path))

    run_id = reader.globals.get("run_id")
    if not run_id:
        run_id = Path(cali_path).stem
        print(f"WARNING: no run_id global in {cali_path}; using filename stem {run_id!r}")

    rows = []
    for record in reader.records:
        row = {"run_id": run_id}
        for key, dtype in PERF_COLUMNS.items():
            if key == "run_id":
                continue
            row[key] = _convert(record.get(key), dtype)
        rows.append(row)

    perf_df = pl.DataFrame(rows, schema=PERF_COLUMNS)
    return perf_df, dict(reader.globals)


def read_topology(json_path: str | Path | None) -> dict | None:
    """Read the communication-topology JSON emitted alongside the .cali file."""
    if json_path is None:
        return None
    with open(json_path) as f:
        return json.load(f)


def _get_int(globals_: dict, key: str) -> int | None:
    value = globals_.get(key)
    return int(value) if value is not None else None


def _match_mode_code(globals_: dict) -> int | None:
    """FI_CXI_RX_MATCH_MODE as an int code, accepting either '1' or 'software'."""
    value = globals_.get("FI_CXI_RX_MATCH_MODE")
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in MATCH_MODE_MAP:
        return MATCH_MODE_MAP[value]
    return int(value)


def build_metadata_row(globals_: dict, topology: dict | None) -> dict:
    """Assemble one metadata-table row from the Caliper globals + topology JSON."""
    run_id = globals_.get("run_id")
    procs = _get_int(globals_, "procs")
    nodes = _get_int(globals_, "nodes")

    residual = globals_.get("relative_residual")
    return {
        "run_id": run_id,
        "procs": procs,
        "nodes": nodes,
        "ppn": (procs // nodes) if procs and nodes else None,
        "RDZV": _get_int(globals_, "FI_CXI_RDZV_THRESHOLD"),
        "MATCH_MODE": _match_mode_code(globals_),
        "GPU_IPC": _get_int(globals_, "MPICH_GPU_IPC_ENABLED"),
        "ASYNC": _get_int(globals_, "MPICH_ASYNC_PROGRESS"),
        "sparse_matrix": globals_.get("sparse_matrix"),
        "max_iters": _get_int(globals_, "max_iters"),
        "converged_iters": _get_int(globals_, "converged_iters"),
        "relative_residual": float(residual) if residual is not None else None,
        "proc_topo": json.dumps(topology) if topology is not None else None,
    }


def store_run(
    db_path: str | Path,
    perf_df: pl.DataFrame,
    metadata_row: dict,
    replace: bool = False,
) -> bool:
    """
    Append one run (perf rows + metadata row) to the database, creating the
    tables if needed. If the run_id is already present it is skipped, unless
    replace=True, in which case its old rows are deleted first. Both tables are
    written in a single transaction. Returns True if the run was stored.
    """
    run_id = metadata_row["run_id"]
    con = ddb.connect(str(db_path))
    try:
        con.execute(PERF_DDL)
        con.execute(METADATA_DDL)

        exists = con.execute(
            f"SELECT count(*) FROM {METADATA_TABLE} WHERE run_id = ?", [run_id]
        ).fetchone()[0]
        if exists and not replace:
            print(f"run_id {run_id!r} already in {db_path}; skipping (use --replace to re-ingest)")
            return False

        con.execute("BEGIN TRANSACTION")
        if exists:
            con.execute(f"DELETE FROM {PERF_TABLE} WHERE run_id = ?", [run_id])
            con.execute(f"DELETE FROM {METADATA_TABLE} WHERE run_id = ?", [run_id])

        placeholders = ", ".join("?" for _ in metadata_row)
        con.execute(
            f"INSERT INTO {METADATA_TABLE} VALUES ({placeholders})",
            list(metadata_row.values()),
        )

        con.register("perf_view", perf_df)
        con.execute(f"INSERT INTO {PERF_TABLE} BY NAME SELECT * FROM perf_view")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    print(f"Stored run_id {run_id!r}: {len(perf_df)} perf rows -> {db_path}")
    return True


def ingest(cali_path: str | Path, json_path: str | Path | None, db_path: str | Path, replace: bool = False) -> bool:
    """Read one .cali/.json pair and store it in the database."""
    perf_df, globals_ = read_cali(cali_path)
    topology = read_topology(json_path)
    metadata_row = build_metadata_row(globals_, topology)
    return store_run(db_path, perf_df, metadata_row, replace=replace)


def ingest_directory(directory: str | Path, db_path: str | Path, replace: bool = False) -> int:
    """
    Ingest every *.cali file in a directory, pairing each with a same-stem
    *.json topology file when one exists. Returns the number of runs stored.
    """
    directory = Path(directory)
    stored = 0
    cali_files = sorted(directory.glob("*.cali"))
    if not cali_files:
        print(f"No .cali files found in {directory}")
        return 0
    for cali_path in cali_files:
        json_path = cali_path.with_suffix(".json")
        stored += ingest(cali_path, json_path if json_path.exists() else None, db_path, replace=replace)
    return stored


def main():
    parser = argparse.ArgumentParser(
        description="Store CG solver Caliper (.cali) + topology (.json) output in a DuckDB database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cali", type=str, default=None, help="Path to a .cali file.")
    parser.add_argument("--json", type=str, default=None, help="Path to the matching topology .json file.")
    parser.add_argument("--dir", type=str, default=None,
                        help="Ingest every *.cali (+ same-stem *.json) in this directory instead of --cali/--json.")
    parser.add_argument("--db", type=str, default="database.duckdb", help="DuckDB database file to append to.")
    parser.add_argument("--replace", action="store_true",
                        help="Re-ingest runs whose run_id is already in the database (delete + insert).")
    args = parser.parse_args()

    if (args.cali is None) == (args.dir is None):
        parser.error("Provide exactly one of --cali or --dir.")

    if args.dir:
        n = ingest_directory(args.dir, args.db, replace=args.replace)
        print(f"Ingested {n} run(s) from {args.dir}")
    else:
        ingest(args.cali, args.json, args.db, replace=args.replace)


if __name__ == "__main__":
    main()
