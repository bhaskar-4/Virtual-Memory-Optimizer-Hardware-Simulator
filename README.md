# AI-Driven Virtual Memory Optimizer

A simple, visual way to explore how memory management behaves under pressure.

This project compares a traditional LRU page replacement strategy against a smarter, AI-assisted page-placement policy in a simulated memory system. Instead of changing real operating system memory, it models the process in software so we can study how pages move between DRAM and NVM under different workloads.

Think of it like this:

- A memory system is like a busy library.
- DRAM is the desk right in front of you where you keep things you need immediately.
- NVM is the storage shelves in the back where valuable things live but take longer to reach.
- LRU is like keeping the most recently used book on the desk and throwing out the oldest one when space runs out.
- The AI policy tries to predict which pages are likely to be useful again soon and which ones can be moved away.

The result is a lightweight simulator and dashboard that helps us understand how page-placement decisions affect system performance.

---

## Why this project matters

Memory is one of the biggest bottlenecks in modern computing systems.

When a system runs out of fast memory, it has to move pages in and out of slower storage. This causes:

- more latency
- more stalls
- lower throughput
- wasted energy and bandwidth

This problem appears in many places:

- large AI models
- database indexing
- scientific computing and data pipelines
- graph processing
- memory-heavy applications
- edge and embedded workloads

The core challenge is simple but difficult to solve well:

> Which pages should stay in fast memory, and which ones can safely wait in slower memory?

---

## Problem domain

This project belongs to the field of memory management and computer architecture.

The real-world challenge is that modern systems have multiple memory levels:

- CPU registers
- cache
- DRAM
- NVM / flash-like storage
- backing storage

Each level has different speed, cost, and capacity trade-offs.

A page that is accessed often should stay in fast memory. But if the system predicts incorrectly, it will keep evicting useful pages and cause expensive misses.

This is especially hard when workloads are:

- scan-heavy
- repetitive but not purely temporal
- near the boundary of available DRAM capacity
- dominated by large structured data access patterns

A good policy is not just “keep recent pages.” It must also understand reuse patterns and workload behavior.

---

## Solution domain

This project uses a trace-driven simulation to model memory behavior and compare two policies:

1. LRU baseline
2. AI-assisted page placement

### Baseline: LRU

LRU keeps recently accessed pages in fast memory and evicts the least recently used page when memory is full.

This works well for many standard workloads, but it struggles when access patterns are scan-heavy or when useful pages are not immediately adjacent in time.

### AI-assisted policy

The simulator profiles each page using a small set of features such as:

- access frequency
- recency / last write state
- mean reuse gap

A Random Forest model learns from these features to decide whether a page should be promoted into DRAM or left in NVM.

The idea is not to “replace the whole memory system,” but to make a better decision when the page replacement policy is under pressure.

---

## Real-world challenges addressed

### 1. Capacity pressure

When the working set is larger than the available DRAM capacity, every page decision matters. Small mistakes become expensive misses.

### 2. Scan-heavy workloads

A scan reads many pages in sequence. This can wipe out the locality that LRU relies on.

This is a common challenge in:

- matrix operations
- analytics workloads
- large table scans
- scientific simulations
- machine learning data preprocessing

### 3. Mixed workload behavior

Some workloads have bursts of high reuse, while others have broad but irregular access patterns. A single simple policy cannot handle all of them well.

### 4. Slow storage penalties

Moving a page from NVM back to DRAM is not free. The cost is measured in latency, bandwidth, and slowdown. A system that makes the wrong choice will feel slower even if the number of pages is unchanged.

### 5. Limited visibility into future access

No policy can perfectly know the future. The best strategies use patterns, statistics, and learned intuition to make better guesses.

---

## How the simulator works

This project follows a two-pass simulation design:

### Pass 1: LRU profiling

The simulator processes the workload and records page-level behavior:

- how often each page is accessed
- how recently it was used
- whether it was read or written
- its average gap between accesses

This creates the training data for the ML model.

### Pass 2: AI decision pass

The trained model predicts which pages deserve placement in DRAM. Then the simulator replays the workload with the AI policy in place.

This allows a fair comparison between:

- LRU-only behavior
- AI-assisted placement behavior

The dashboard visualizes the outcome using metrics like:

- EMAT (effective memory access time)
- faults
- NVM access counts
- timeline charts
- per-page placement maps

---

## What kind of workloads does it support?

This project is built for structured, memory-sensitive access traces, especially workloads that stress page replacement logic.

### Supported workload patterns

#### 1. Strided scan

A large, repeated walking pattern that jumps through memory in a structured way.

This is useful for testing how well a system handles:

- matrix traversal
- large spatial scans
- analytical processing
- cache-thrashing access sequences

This is one of the most useful workloads for exposing the weakness of LRU.

#### 2. Tensor-local workload

A workload that keeps nearby data items together in a pattern similar to tensor or matrix computation.

This is useful for modeling:

- scientific computing
- ML preprocessing and tensor math
- linear algebra
- regular locality patterns

#### 3. User-uploaded traces

The project can also accept external memory traces in CSV format. This makes it useful for experimenting with real traces or custom workloads.

The trace format is:

```text
timestamp_or_access_index,R_or_W,hex_virtual_address
```

Example:

```text
125,R,0x7fff00000000
```

---

## Why the AI helps in some cases

The model does not magically solve every memory problem. It helps when the workload has patterns that LRU misses.

A good analogy:

- LRU is like always keeping the last few books you touched.
- AI is like noticing that some books are repeatedly needed in a predictable pattern, even if they are not the most recently touched.

This means AI can help when the workload is more than just “recently used.” It can capture patterns such as:

- repeated scanning of large structures
- recurring page reuse after a moderate delay
- pages that are important but not currently hot

That said, performance depends on the workload and on memory configuration. This is a research and simulation tool, not a real hardware controller.

---

## Project architecture

The project combines several components:

- `app.py` — Streamlit dashboard for running benchmarks and visualizing results
- `workload_generator.py` — generates synthetic structured workloads
- `memory_simulator.cpp` — C++ simulator for the LRU and AI passes
- `train_model.py` — trains the Random Forest model from page telemetry
- `requirements.txt` — Python dependencies

The workflow is:

1. Generate or load a workload
2. Simulate LRU behavior
3. Collect page statistics
4. Train the model
5. Replay the workload with AI decisions
6. Compare metrics and visualize results

---

## Run the project

### Requirements

- Windows 10/11 or Ubuntu Linux
- Python 3.10 or newer
- C++17 compiler such as `g++`
- Python dependencies from `requirements.txt`

### Windows setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
build.bat
python -m streamlit run app.py --server.address 127.0.0.1
```

### Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
g++ -std=c++17 -O2 -Wall -Wextra memory_simulator.cpp -o memory_simulator
python -m streamlit run app.py --server.address 127.0.0.1
```

Open:

```text
http://127.0.0.1:8501
```

---

## What you can analyze in the dashboard

The dashboard lets you:

- compare LRU vs AI memory policies
- tune DRAM capacity and latency values
- inspect page-map placement decisions
- study fault and NVM access behavior over time
- test different synthetic workloads
- upload custom traces and compare outcomes

This makes it useful for both learning and experimentation.

---

## Generated output files

The project produces several files during a run:

- `memory_trace.txt` — generated or uploaded workload trace
- `telemetry_ml_dataset.csv` — per-page behavioral features
- `page_classifier.joblib` — trained model for page decisions
- `page_predictions.csv` — output of the model
- `run_metrics_LRU.csv` — LRU summary metrics
- `run_metrics_AI.csv` — AI summary metrics
- `run_timeline_*.csv` — per-interval timeline metrics
- `page_map_*.csv` — final page placements across memory tiers

---

## Important note

This is an educational simulation project designed for analysis and learning.

It is not a direct controller for physical RAM or a real operating-system memory manager. The simulated numbers are estimates meant to compare policy behavior under controlled trace-driven conditions.

The model learns from a heuristic labeling strategy and measures how well that policy matches the simulated target, not how a real hardware system behaves in every environment.

---

## Summary

This project sits at the intersection of:

- computer architecture
- virtual memory management
- machine learning for systems
- trace-based performance analysis

It demonstrates a practical idea:

> even a simple learning-based policy can improve memory decisions when the workload is structured in a way that defeats traditional heuristics.

That makes it a good prototype for exploring smarter memory placement, workload-aware optimization, and trace-driven system design.

---

## License

This project is intended for academic and research-oriented learning use.
