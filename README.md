# AI-Driven Virtual Memory Optimizer

A local trace-driven simulator for comparing LRU page replacement with Random-Forest-assisted placement across simulated DRAM and NVM tiers. It models a 16-entry TLB, a hash-map page table, configurable DRAM capacity, access latencies, page placement, and run metrics. It does not control host memory or modify the operating system.

## Requirements

- Windows 10/11 or Ubuntu Linux
- Python 3.10 or newer
- A C++17 compiler available as `g++` (MinGW on Windows)
- Python packages listed in `requirements.txt`

## Setup on Windows

Open PowerShell in this folder and run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
build.bat
python -m streamlit run app.py --server.address 127.0.0.1
```

If PowerShell does not find `g++`, install MinGW-w64 and add its `bin` directory to `PATH`. The dashboard can also compile the C++ engine automatically when you run a benchmark.

In VS Code, run **Python: Select Interpreter** and choose `.venv\Scripts\python.exe` so editor diagnostics use the installed project packages.

## Setup on Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
g++ -std=c++17 -O2 -Wall -Wextra memory_simulator.cpp -o memory_simulator
python -m streamlit run app.py --server.address 127.0.0.1
```

In VS Code, select `.venv/bin/python` as the workspace interpreter.

The dashboard is available at `http://127.0.0.1:8501`. Binding to loopback keeps it local to this machine.

## Run the Workbench

1. Upload a trace file, or choose a generated workload pattern and select **Run Live System Benchmark**. The default strided scan creates 200,000 references; the tensor-local option preserves a matrix-multiply-style locality pattern for comparison.
2. Set DRAM, NVM, and TLB-walk latencies and the simulated DRAM frame capacity.
3. Select **Run Live System Benchmark**. The app runs the LRU pass, builds telemetry, trains or refreshes the Random Forest, creates page predictions, then runs the AI placement pass.
4. Compare EMAT and fault/NVM-access timelines, then inspect the eight-column DRAM/NVM page map.

Trace files are comma-separated rows in this order. Generated workloads use the access index as the timestamp; uploaded traces may use another monotonic time unit, and reuse gaps use that same unit.

```text
timestamp_or_access_index,R_or_W,hex_virtual_address
```

Example: `125,R,0x7fff00000000`.

The command-line workload generator also accepts `--operations`, `--matrix-size`, `--pattern` (`strided-scan` or `tensor-local`), `--seed`, and `--output`. The strided scan traverses matrix A using a 7-row stride and matrix B in reverse-column order. To build the classifier manually after an LRU simulation, run:

```bash
python train_model.py
```

## Generated Files

- `memory_trace.txt`: generated or uploaded trace
- `telemetry_ml_dataset.csv`: per-page features and training labels from the LRU profiling pass
- `page_classifier.joblib`: trained Random Forest model
- `page_predictions.csv`: per-page DRAM/NVM decisions produced by the model
- `run_metrics_LRU.csv` and `run_metrics_AI.csv`: aggregate metrics
- `run_timeline_*.csv`: interval metrics for the fault-rate plots
- `page_map_*.csv`: per-page tier placements

Generated outputs and the compiled binary are excluded from version control. A full workload trace may be large, so it is also excluded.

## Simulation Notes

This is an educational two-pass simulator. The C++ engine profiles page frequency, the last observed read/write state, and each page's mean inter-access gap in the trace's timestamp units during the LRU pass. The Python classifier learns from these three features. Pages with more than 30 accesses and a mean reuse gap below 2,000 trace units are eligible for DRAM labels; only the top-scoring eligible pages up to the DRAM capacity are labeled for promotion. During the AI pass, prediction confidence influences which resident page is evicted, with LRU recency breaking equal-priority ties. The training target is a capacity-aware heuristic, not ground-truth hardware behavior, and measured improvements depend on the trace and settings. Held-out accuracy measures agreement with these heuristic labels, not real hardware optimality. Values are simulated estimates, not measurements of physical CXL, DRAM, or NVM devices.

The SRS describes a 512-page DRAM bank. The dashboard supports capacities up to 512 pages and defaults to 128 pages. The generated strided scan touches 130 pages, so this near-capacity setting deliberately stresses replacement behavior; increase the slider to 512 to model the full-capacity configuration.