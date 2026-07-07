import duckdb as ddb
import caliperreader as cr
import thicket as th
import polars as pl

# data_th = th.Thicket.from_caliper(
#     ["./cg_solver.cali"]
# )

reader = cr.CaliperReader()
reader.read("./cg_solver.cali")


for record in reader.records:
    # if record.get('mpi.function', '') == 'MPI_Waitall':
    print(record)

print(reader.attributes())

# data_df = pl.DataFrame(reader.records, infer_schema_length=None)
# data_df.glimpse()