# CG
Standalone CG solver



# Structure

## Scripts
This directory contains pythong scripts for turning matrices stored in matrix market format to PETSc binary.


## src
The source directory contains two nested subdirectories: cpu and hip. The former is the original code from the main branch of this repo, as of 7/6/26. The later is a combination of the code from cpu, and the heterogeneous branch of the original repository.


# Notes for Data Collection
The hip codebase needs to be instrumented with Caliper, and capture the following information
- 


# References
- [Caliper ConfigManager](https://software.llnl.gov/Caliper/CaliperBasics.html#configmanager-api)
- [Caliper Environment Variable Configuration](https://software.llnl.gov/Caliper/configuration.html)
- [Caliper Architecture & Workflow](https://software.llnl.gov/Caliper/workflow.html)
- [Caliper Reader](https://software.llnl.gov/Caliper/pythonreader.html)
- [Caliper Source Code for mms and mmc attributes](https://github.com/llnl/Caliper/blob/4ff69d07166da66de217d138f1576733fb5be522/src/caliper/controllers/controllers.cpp#L409)