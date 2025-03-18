# Storm Babet

A case study utilising forecast-based event attribution to study the response of storm Babet (October 2023) to climate change. Also included are some plotting routines to compare to other attribution methods.

## Structure
- PGW_analysis: Plotting routines for the analysis of pseudo-global warming simulations
- analogues: initial code to calculate flow analogues from ERA5
- babet: Contains classes and functions that are used in the data post-processing and analysis
- data: some smaller data files, not tracked in git
- data_scripts: python scripts to calculate post-processed variables such as vertically integrated advection. These can then be used with slurm.
- docs: environment file which creates a virtual environment to run the code in this repo
- notebooks: main data analysis on forecast-based runs in python notebooks, includes folder of figures
