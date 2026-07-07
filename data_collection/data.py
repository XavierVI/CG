import duckdb as ddb
import caliperreader as cr
import polars as pl


"""
SCHEMA raw perf data table:
- run_id: varchar         This is actually the ID used with flux run/submit,
- batch_job_id: varchar
- mpi.rank: int
- sum#sum#time.duration: double
- min#aggregate.slot: int
- region: varchar
- path: varchar
- mpi.function: varchar
- sum#mmc.coll: int
- min#mms.min: int
- avg#mms.avg: int
- max#mms.max: int
- sum#mmc.send: int
- sum#mmc.recv: int


SCHEMA metadata table:
- file_name: varchar
- batch_job_id: varchar
- procs: int
- nodes: int
- RDZV: int
- MATCH_MODE: int (software = 0, hardware = 1, hybrid = 2)
- GPU_IPC: int
- ASYNC: int
- sparse_matrix: varchar
- proc_topo: json (it's a type in duckdb)

"""
reader = cr.CaliperReader()
reader.read("./cg_solver.cali")


for record in reader.records:
    # convert 'region' and 'path' to comma separated strings
    # print(record)
    
    if record.keys().__contains__('region') and type(record['region']) == list:
        record['region'] = ','.join(record['region']).replace('[', '').replace(']', '').replace(' ', '')
    
    if record.keys().__contains__('path'):
        if len(record['path']) > 1:
            record['path'] = ','.join(record['path']).replace('[', '').replace(']', '').replace(' ', '')
        else:
            record['path'] = record['path'][0]
    
    print(record)

print("==============================================================================================================================================================")
print(dir(reader))
print(reader.attributes())
# print(dir(reader.attribute('run_id')))
# print(reader.attribute('run_id').metadata())
print(reader.globals['run_id'])
# print(reader.globals['uri_id'])


data_df = pl.DataFrame(reader.records)
data_df = data_df.filter(
        ( pl.col("mpi.function") == "MPI_Allreduce" ) |
        ( pl.col("mpi.function") == "MPI_Isend"     ) |
        ( pl.col("mpi.function") == "MPI_Irecv"     ) |
        ( pl.col("mpi.function") == "MPI_Waitall"   ) |
        ( pl.col("mpi.function") == "parallel_spmv" )
    # & (pl.col("path").str.contains("CG"))
)
# append run_id to the data_df
data_df = data_df.with_columns(pl.lit(reader.globals['run_id']).alias("run_id"))
data_df.glimpse()
