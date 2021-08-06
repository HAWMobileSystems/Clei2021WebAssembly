# PolyBenchC Benchmark Scripts

scripts to setup and run PolyBenchC benchmark

## Setup
- adapt paths in `config.py` (only those you want to use)
- use Python 3.7 or newer
- install Python requirements from `requirements.txt`
- run `polybenchc_setup.py` to download and patch PolyBenchC
- Jupyter server for analysis

## Run Benchmarks
- uncomment benchmarks you want to run (in `polybenchc_benchmark.py`)
    - compilation to the specific target is required in order to run benchmarks
- run `polybenchc_benchmark.py`


## Analyze Benchmarks
- run cells in Jupyter notebook `polybenchc_analyze.ipynb`