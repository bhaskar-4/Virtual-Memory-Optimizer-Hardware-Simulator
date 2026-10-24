import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
TRACE_PATH = ROOT / "memory_trace.txt"
ENGINE_SOURCE = ROOT / "memory_simulator.cpp"
ENGINE_PATH = ROOT / ("memory_simulator.exe" if sys.platform == "win32" else "memory_simulator")
MODEL_PATH = ROOT / "page_classifier.joblib"
DATASET_PATH = ROOT / "telemetry_ml_dataset.csv"
PREDICTIONS_PATH = ROOT / "page_predictions.csv"

st.set_page_config(page_title="Memory Tiering Workbench", page_icon="MT", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f3f5f2; }
    [data-testid="stSidebar"] { background: #e8eee9; }
    .page-title { font-size: 2rem; font-weight: 700; color: #172b27; margin-bottom: 0; }
    .page-subtitle { color: #5e706a; margin: 0.15rem 0 1.25rem; }
    .tier-slot { border-radius: 4px; padding: 0.55rem 0.2rem; text-align: center;
                 font-size: 0.72rem; font-weight: 700; border: 1px solid transparent; }
    .tier-dram { background: #d9f0e2; color: #17633b; border-color: #9dccad; }
    .tier-nvm { background: #fff0d2; color: #805310; border-color: #e7c47e; }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_process(command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "The process returned a non-zero exit code."
        raise RuntimeError(f"{label} failed:\n{detail}")
    return result


def ensure_engine() -> None:
    if ENGINE_PATH.exists() and ENGINE_PATH.stat().st_mtime >= ENGINE_SOURCE.stat().st_mtime:
        return
    run_process(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", str(ENGINE_SOURCE), "-o", str(ENGINE_PATH)],
        "C++ engine compilation",
    )


def run_simulation(policy: str, tlb_walk: int, dram_latency: int, nvm_latency: int, dram_pages: int):
    return run_process(
        [
            str(ENGINE_PATH),
            str(tlb_walk),
            str(dram_latency),
            str(nvm_latency),
            policy,
            str(dram_pages),
            str(PREDICTIONS_PATH),
        ],
        f"{policy} simulation",
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected simulation output was not created: {path.name}")
    return pd.read_csv(path)


def run_benchmark(
    uploaded_trace,
    pattern: str,
    tlb_walk: int,
    dram_latency: int,
    nvm_latency: int,
    dram_pages: int,
):
    if uploaded_trace is not None:
        TRACE_PATH.write_bytes(uploaded_trace.getvalue())
    else:
        run_process(
            [
                sys.executable,
                str(ROOT / "workload_generator.py"),
                "--output",
                str(TRACE_PATH),
                "--pattern",
                pattern,
            ],
            "Sample workload generation",
        )

    ensure_engine()
    baseline = run_simulation("LRU", tlb_walk, dram_latency, nvm_latency, dram_pages)
    PREDICTIONS_PATH.unlink(missing_ok=True)
    try:
        training = run_process(
            [
                sys.executable,
                str(ROOT / "train_model.py"),
                "--dataset",
                str(DATASET_PATH),
                "--model",
                str(MODEL_PATH),
                "--predictions",
                str(PREDICTIONS_PATH),
            ],
            "Random Forest training and prediction",
        ).stdout
    except RuntimeError as error:
        training = f"Classifier unavailable; requested LRU fallback.\n{error}"
    optimized = run_simulation("AI", tlb_walk, dram_latency, nvm_latency, dram_pages)
    return baseline.stdout, training, optimized.stdout


st.markdown('<div class="page-title">Memory Tiering Workbench</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Trace-driven comparison of LRU paging and learned page placement</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Simulation settings")
    uploaded_trace = st.file_uploader("Workload trace (CSV)", type=["txt", "csv"])
    workload_pattern_label = st.selectbox(
        "Generated workload pattern",
        options=["Strided scan", "Tensor-local"],
    )
    workload_pattern = {
        "Strided scan": "strided-scan",
        "Tensor-local": "tensor-local",
    }[workload_pattern_label]
    dram_latency = st.slider("DRAM latency (ns)", min_value=10, max_value=100, value=50)
    nvm_latency = st.slider("NVM latency (ns)", min_value=100, max_value=500, value=200)
    tlb_walk = st.slider("TLB miss walk (ns)", min_value=5, max_value=50, value=15)
    dram_pages = st.slider("DRAM frame capacity", min_value=4, max_value=512, value=128, step=4)
    run_button = st.button("Run Live System Benchmark", type="primary", use_container_width=True)
    generate_button = st.button("Generate sample workload", use_container_width=True)
    st.caption("Trace rows: timestamp or logical access index, R/W operation, hexadecimal address.")

if generate_button:
    try:
        result = run_process(
            [
                sys.executable,
                str(ROOT / "workload_generator.py"),
                "--output",
                str(TRACE_PATH),
                "--pattern",
                workload_pattern,
            ],
            "Sample workload generation",
        )
        st.success(result.stdout.strip())
    except (OSError, RuntimeError) as error:
        st.error(str(error))

if run_button:
    try:
        with st.spinner("Compiling the engine, profiling LRU, training the classifier, and replaying predictions..."):
            baseline_log, training_log, ai_log = run_benchmark(
                uploaded_trace, workload_pattern, tlb_walk, dram_latency, nvm_latency, dram_pages
            )
        st.session_state["benchmark_complete"] = True
        st.session_state["benchmark_logs"] = (baseline_log, training_log, ai_log)
        st.success("Benchmark completed. Results below are from the current trace and settings.")
        if "LRU fallback" in ai_log:
            st.warning("Classifier predictions were unavailable or incomplete; the AI pass used LRU fallback.")
    except (OSError, RuntimeError, FileNotFoundError) as error:
        st.error(str(error))

if st.session_state.get("benchmark_complete"):
    try:
        lru_metrics = load_csv(ROOT / "run_metrics_LRU.csv").iloc[0]
        ai_metrics = load_csv(ROOT / "run_metrics_AI.csv").iloc[0]
        ai_map = load_csv(ROOT / "page_map_AI.csv")
        lru_timeline = load_csv(ROOT / "run_timeline_LRU.csv")
        ai_timeline = load_csv(ROOT / "run_timeline_AI.csv")

        emat_delta = 100.0 * (lru_metrics["emat"] - ai_metrics["emat"]) / lru_metrics["emat"]
        metric_columns = st.columns(4)
        metric_columns[0].metric("Unique pages", f"{int(ai_metrics['total_pages']):,}")
        metric_columns[1].metric("LRU EMAT", f"{lru_metrics['emat']:.2f} ns")
        metric_columns[2].metric("AI EMAT", f"{ai_metrics['emat']:.2f} ns", f"{emat_delta:+.1f}% vs LRU")
        metric_columns[3].metric(
            "AI NVM access rate", f"{ai_metrics['fault_rate']:.2f}%", "requests served from NVM"
        )

        chart_columns = st.columns(2)
        with chart_columns[0]:
            emat_figure = go.Figure(
                [
                    go.Bar(name="LRU", x=["Effective memory access time"], y=[lru_metrics["emat"]], marker_color="#d16b4e"),
                    go.Bar(name="Random Forest", x=["Effective memory access time"], y=[ai_metrics["emat"]], marker_color="#27845a"),
                ]
            )
            emat_figure.update_layout(title="Average access latency", yaxis_title="Nanoseconds", barmode="group", height=350)
            st.plotly_chart(emat_figure, use_container_width=True)

        with chart_columns[1]:
            fault_figure = go.Figure()
            fault_figure.add_scatter(
                x=lru_timeline["progress_percent"], y=lru_timeline["fault_rate"],
                mode="lines", name="LRU", line={"color": "#d16b4e", "width": 2},
            )
            fault_figure.add_scatter(
                x=ai_timeline["progress_percent"], y=ai_timeline["fault_rate"],
                mode="lines", name="Random Forest", line={"color": "#27845a", "width": 2},
            )
            fault_figure.update_layout(
                title="Page fault / NVM access rate by trace interval",
                xaxis_title="Trace progress (%)", yaxis_title="Rate (%)", height=350,
            )
            st.plotly_chart(fault_figure, use_container_width=True)

        st.subheader("AI page placement")
        st.caption("First 64 pages in VPN order; green pages occupy simulated DRAM frames, amber pages remain in NVM.")
        visible_pages = ai_map.head(64)
        for start in range(0, len(visible_pages), 8):
            row = visible_pages.iloc[start : start + 8]
            slots = st.columns(8)
            for slot, (_, page) in zip(slots, row.iterrows()):
                tier = str(page["tier"])
                tier_class = "tier-dram" if tier == "DRAM" else "tier-nvm"
                with slot:
                    st.markdown(
                        f'<div class="tier-slot {tier_class}">{tier}-P[{int(page["page_id"])}]</div>',
                        unsafe_allow_html=True,
                    )

        with st.expander("Run details"):
            for title, output in zip(("LRU simulator", "Classifier", "AI simulator"), st.session_state["benchmark_logs"]):
                st.markdown(f"**{title}**")
                st.code(output.strip() or "Completed without console output", language="text")
    except (FileNotFoundError, KeyError, ValueError) as error:
        st.error(f"Could not load benchmark outputs: {error}")
else:
    st.info("Upload a trace or generate the sample workload, adjust the simulated hardware settings, and run the benchmark.")