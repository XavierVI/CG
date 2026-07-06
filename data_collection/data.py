#!/usr/bin/env python3
"""
Parse Cray MPICH / ACG solver log files and store their contents in a single
DuckDB table.

Model: ONE row per log file. The run-level scalars are ordinary columns. The
variable-length child datasets (performance breakdown, per-rank communication,
per-rank mapping) are stored as native DuckDB STRUCT[] columns, and the optional
sidecar .json file is stored verbatim in a JSON column.

Rationale for not flattening into a wide table: performance rows, communication
rows, and mapping rows have independent cardinalities per run. A fully
denormalised single table would require cross-joining them, producing a
combinatorial explosion of meaningless rows. STRUCT[] keeps it to one table
with no joins; use UNNEST to expand a child dataset when needed, e.g.:

    SELECT filename, p.operation, p.seconds_per_proc
    FROM runs, UNNEST(perf_breakdown) AS x(p);

Usage:
    python3 load_mpi_logs.py LOG [LOG ...] [-d DB] [-t TABLE] [--append]

LOG arguments may be individual files, shell globs, or directories (every
*.log directly inside a directory is read). For each log, a sidecar .json with
the same stem is loaded if present; it is optional.
"""

import argparse
import glob
import json
import math
import os
import re
import sys

import duckdb


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
# Run-level scalar columns (one value per file).
RUN_SCALAR_COLS = [
    ("filename", "VARCHAR"),
    ("mpi_version", "VARCHAR"),
    ("mpi_build_info", "VARCHAR"),
    ("thread_level", "VARCHAR"),
    ("num_processes", "INTEGER"),
    ("rccl_version", "INTEGER"),
    ("solver", "VARCHAR"),
    ("communication", "VARCHAR"),
    ("unknowns", "BIGINT"),
    ("solves", "INTEGER"),
    ("total_iterations", "BIGINT"),
    ("total_flops_gflop", "DOUBLE"),
    ("total_flop_rate_gflops", "DOUBLE"),
    ("total_solver_time_s", "DOUBLE"),
    ("last_max_iterations", "BIGINT"),
    ("last_tol_residual", "DOUBLE"),
    ("last_tol_rel_residual", "DOUBLE"),
    ("last_tol_diff_iterates", "DOUBLE"),
    ("last_tol_rel_diff_iterates", "DOUBLE"),
    ("last_iterations", "BIGINT"),
    ("rhs_2norm", "DOUBLE"),
    ("initial_guess_2norm", "DOUBLE"),
    ("initial_residual_2norm", "DOUBLE"),
    ("residual_2norm", "DOUBLE"),
    ("diff_solution_iterates_2norm", "DOUBLE"),
    ("floating_point_exceptions", "VARCHAR"),
]

# Field layout of each STRUCT element in the nested child columns. These also
# define exactly which keys parse_file emits per child record.
PERF_FIELDS = [
    ("operation", "VARCHAR"),
    ("seconds_per_proc", "DOUBLE"),
    ("times_per_proc", "BIGINT"),
    ("bytes_per_proc", "BIGINT"),
    ("throughput_gbs_per_proc", "DOUBLE"),
    ("us_per_op", "DOUBLE"),
    ("msgs_per_proc", "DOUBLE"),
    ("us_per_msg", "DOUBLE"),
]

COMM_FIELDS = [
    ("rank", "INTEGER"),
    ("send_bytes", "BIGINT"),
    ("send_bytes_per_it", "BIGINT"),
    ("send_msgs", "BIGINT"),
    ("send_msgs_per_it", "INTEGER"),
    ("send_max_bytes_per_msg", "BIGINT"),
    ("recv_bytes", "BIGINT"),
    ("recv_bytes_per_it", "BIGINT"),
    ("recv_msgs", "BIGINT"),
    ("recv_msgs_per_it", "INTEGER"),
    ("recv_max_bytes_per_msg", "BIGINT"),
]

MAP_FIELDS = [
    ("rank", "INTEGER"),
    ("cpu_cores", "VARCHAR"),
    ("device", "INTEGER"),
    ("node", "VARCHAR"),
]


def _struct_type(fields):
    inner = ", ".join(f'"{name}" {typ}' for name, typ in fields)
    return f"STRUCT({inner})[]"


# Full column list for the single table: scalars, then nested children, then
# the raw sidecar JSON.
NESTED_COLS = [
    ("perf_breakdown", _struct_type(PERF_FIELDS)),
    ("rank_comm", _struct_type(COMM_FIELDS)),
    ("rank_mapping", _struct_type(MAP_FIELDS)),
    ("raw_json", "JSON"),
]
TABLE_COLS = RUN_SCALAR_COLS + NESTED_COLS


# --------------------------------------------------------------------------- #
# value conversion helpers
# --------------------------------------------------------------------------- #
def to_float(s):
    if s is None:
        return None
    t = s.strip().lower()
    if t == "":
        return None
    if t in ("inf", "+inf", "infinity"):
        return math.inf
    if t in ("-inf", "-infinity"):
        return -math.inf
    if t == "nan":
        return math.nan
    return float(t)


def to_int(s):
    return None if s is None else int(s)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
LAST_SOLVE_LABELS = {
    "maximum iterations": ("last_max_iterations", to_int),
    "tolerance for residual": ("last_tol_residual", to_float),
    "tolerance for relative residual": ("last_tol_rel_residual", to_float),
    "tolerance for difference in solution iterates":
        ("last_tol_diff_iterates", to_float),
    "tolerance for relative difference in solution iterates":
        ("last_tol_rel_diff_iterates", to_float),
    "iterations": ("last_iterations", to_int),
    "right-hand side 2-norm": ("rhs_2norm", to_float),
    "initial guess 2-norm": ("initial_guess_2norm", to_float),
    "initial residual 2-norm": ("initial_residual_2norm", to_float),
    "residual 2-norm": ("residual_2norm", to_float),
    "difference in solution iterates 2-norm":
        ("diff_solution_iterates_2norm", to_float),
    "floating-point exceptions":
        ("floating_point_exceptions", lambda x: x.strip()),
}

NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def search(pattern, text, group=1, flags=0):
    m = re.search(pattern, text, flags)
    return m.group(group) if m else None


def parse_json(path):
    """Load and return the sidecar JSON file as a Python object."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return json.load(fh)


def parse_file(path):
    """
    Parse one log file. Returns (run, perf, comm, mapping) where:
      run     dict of run-level scalars (keys == RUN_SCALAR_COLS names)
      perf    list of dicts (keys == PERF_FIELDS names)
      comm    list of dicts (keys == COMM_FIELDS names)
      mapping list of dicts (keys == MAP_FIELDS names)
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    fname = os.path.basename(path)

    run = {
        "filename": fname,
        "mpi_version": (search(r"MPI VERSION\s*:\s*(.+)", text) or "").strip()
                       or None,
        "mpi_build_info": (search(r"MPI BUILD INFO\s*:\s*(.+)", text) or "")
                          .strip() or None,
        "thread_level": search(r"thread level:\s*([A-Za-z0-9_]+)", text),
        "num_processes": to_int(search(r"(\d+)\s+MPI processes", text)),
        "rccl_version": to_int(search(r"RCCL version\s+(\d+)", text)),
        "solver": search(r"using\s+(\S+)\s+solver", text),
        "communication": search(r"Using\s+(\S+)\s+for communication", text),
        "unknowns": to_int(search(r"unknowns:\s*(\d+)", text)),
        "solves": to_int(search(r"\bsolves:\s*(\d+)", text)),
        "total_iterations": to_int(search(r"total iterations:\s*(\d+)", text)),
        "total_flops_gflop":
            to_float(search(r"total flops:\s*(" + NUM + r")", text)),
        "total_flop_rate_gflops":
            to_float(search(r"total flop rate:\s*(" + NUM + r")", text)),
        "total_solver_time_s":
            to_float(search(r"total solver time:\s*(" + NUM + r")", text)),
    }

    # last-solve scalars: scoped to the segment after "last solve:" so that
    # "iterations" / "maximum iterations" are not confused with the totals above
    for col, _ in LAST_SOLVE_LABELS.values():
        run[col] = None

    parts = text.split("last solve:", 1)
    if len(parts) > 1:
        for line in parts[1].splitlines():
            if ":" not in line:
                continue
            label, _, value = line.partition(":")
            entry = LAST_SOLVE_LABELS.get(label.strip())
            if entry:
                col, conv = entry
                run[col] = conv(value)

    # performance breakdown: scoped between the header and "last solve:"
    perf = []
    pstart = text.find("performance breakdown:")
    if pstart != -1:
        pend = text.find("last solve:")
        segment = text[pstart: pend if pend != -1 else len(text)]
        for line in segment.splitlines():
            head = re.match(r"\s*([A-Za-z0-9_]+):\s+(" + NUM + r")\s+seconds",
                            line)
            if not head:
                continue
            rec = {name: None for name, _ in PERF_FIELDS}
            rec["operation"] = head.group(1)
            rec["seconds_per_proc"] = to_float(head.group(2))
            rest = line.split(":", 1)[1]
            for value, unit in re.findall(r"(" + NUM + r")\s+(\S+)", rest):
                # order matters: more specific suffixes first
                if "op/proc" in unit:
                    rec["us_per_op"] = to_float(value)
                elif unit == "msg/proc":
                    rec["msgs_per_proc"] = to_float(value)
                elif "msg/proc" in unit:          # e.g. us/msg/proc
                    rec["us_per_msg"] = to_float(value)
                elif "GB/s" in unit:
                    rec["throughput_gbs_per_proc"] = to_float(value)
                elif unit.startswith("B/"):
                    rec["bytes_per_proc"] = to_int(value)
                elif "times" in unit:
                    rec["times_per_proc"] = to_int(value)
                # "seconds" is already captured; anything else is ignored
            perf.append(rec)

    # per-rank halo-exchange communication (sends + receives folded per rank)
    comm = {}
    pattern = (r"rank\s+(\d+)\s+(sends|receives)\s+(\d+)\s+B\s+(\d+)\s+B/it"
               r"\s+in\s+(\d+)\s+msg\s+(\d+)\s+msg/it\s+max\s+(\d+)\s+B/msg")
    for m in re.finditer(pattern, text):
        rank = int(m.group(1))
        rec = comm.setdefault(rank, {name: None for name, _ in COMM_FIELDS})
        rec["rank"] = rank
        pre = "send" if m.group(2) == "sends" else "recv"
        rec[pre + "_bytes"] = int(m.group(3))
        rec[pre + "_bytes_per_it"] = int(m.group(4))
        rec[pre + "_msgs"] = int(m.group(5))
        rec[pre + "_msgs_per_it"] = int(m.group(6))
        rec[pre + "_max_bytes_per_msg"] = int(m.group(7))
    comm_rows = [comm[r] for r in sorted(comm)]

    # rank-to-core/device/node mapping
    mapping = []
    map_pat = (r"rank\s+(\d+)\s*->\s*CPU cores\s+(.+?)\s+and device\s+(\d+)"
               r"\s+on\s+(\S+)")
    for m in re.finditer(map_pat, text):
        mapping.append({
            "rank": int(m.group(1)),
            "cpu_cores": m.group(2).strip(),
            "device": int(m.group(3)),
            "node": m.group(4).strip(),
        })

    return run, perf, comm_rows, mapping


# --------------------------------------------------------------------------- #
# database writing
# --------------------------------------------------------------------------- #
def create_table(con, name):
    ddl = ", ".join(f'"{c}" {t}' for c, t in TABLE_COLS)
    con.execute(f'CREATE TABLE IF NOT EXISTS "{name}" ({ddl})')


def build_row(run, perf, comm, mapping, raw_json):
    """Assemble the value tuple for one table row, in TABLE_COLS order."""
    values = [run.get(c) for c, _ in RUN_SCALAR_COLS]
    values.append(perf)
    values.append(comm)
    values.append(mapping)
    values.append(None if raw_json is None else json.dumps(raw_json))
    return values


def insert_rows(con, name, rows):
    if not rows:
        return
    collist = ", ".join(f'"{c}"' for c, _ in TABLE_COLS)
    placeholders = ", ".join("?" for _ in TABLE_COLS)
    con.executemany(
        f'INSERT INTO "{name}" ({collist}) VALUES ({placeholders})', rows)


# --------------------------------------------------------------------------- #
# file collection
# --------------------------------------------------------------------------- #
def collect_logs(args):
    """
    Expand the CLI path arguments (files, globs, directories) into a list of
    (log_path, json_path_or_None) pairs. A sidecar .json sharing the log's
    stem is attached when present; its absence does not drop the log.
    """
    log_paths = []
    for arg in args:
        if os.path.isdir(arg):
            for entry in sorted(os.listdir(arg)):
                full = os.path.join(arg, entry)
                if entry.endswith(".log") and os.path.isfile(full):
                    log_paths.append(full)
        else:
            expanded = glob.glob(arg)
            if not expanded and os.path.isfile(arg):
                expanded = [arg]
            log_paths.extend(p for p in sorted(expanded)
                             if p.endswith(".log") and os.path.isfile(p))

    # de-duplicate while preserving order
    seen = set()
    unique = []
    for p in log_paths:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    pairs = []
    for log in unique:
        sidecar = log[:-len(".log")] + ".json"
        pairs.append((log, sidecar if os.path.isfile(sidecar) else None))
    return pairs


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--logs", nargs="+", metavar="LOG",
                    help="log files, globs, or directories of *.log files")
    ap.add_argument("-d", "--db", default="mpi_logs.duckdb",
                    help="DuckDB database file (default: mpi_logs.duckdb)")
    ap.add_argument("-t", "--table", default="runs",
                    help="target table name (default: runs)")
    ap.add_argument("--append", action="store_true",
                    help="keep existing rows for re-ingested files "
                         "(default: replace them)")
    args = ap.parse_args()

    pairs = collect_logs(args.logs)
    if not pairs:
        print("error: no input files found", file=sys.stderr)
        return 1

    rows = []
    filenames = []
    for log_path, json_path in pairs:
        try:
            run, perf, comm, mapping = parse_file(log_path)
            raw_json = parse_json(json_path) if json_path else None
        except Exception as exc:                      # noqa: BLE001
            print(f"error parsing '{log_path}': {exc}", file=sys.stderr)
            continue

        rows.append(build_row(run, perf, comm, mapping, raw_json))
        filenames.append(run["filename"])
        print(f"parsed {os.path.basename(log_path)}: "
              f"{len(perf)} ops, {len(comm)} ranks, {len(mapping)} mappings"
              f"{'' if json_path else ' (no json file)'}")

    if not rows:
        print("error: nothing parsed successfully", file=sys.stderr)
        return 1

    con = duckdb.connect(args.db)
    try:
        create_table(con, args.table)
        con.execute("BEGIN")
        if not args.append and filenames:
            ph = ", ".join("?" for _ in filenames)
            con.execute(
                f'DELETE FROM "{args.table}" WHERE filename IN ({ph})',
                filenames)
        insert_rows(con, args.table, rows)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        print(con.sql("SHOW TABLES"))
        print(con.sql(f"SELECT * FROM {args.table} LIMIT 10"))
        con.close()

    print(f"\nwrote {len(rows)} run(s) to {args.db} (table '{args.table}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())