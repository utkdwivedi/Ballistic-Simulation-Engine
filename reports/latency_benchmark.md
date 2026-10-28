# ⚡ Ballistic Neural Surrogate: Inference Latency & Throughput Benchmark

> **Phase 5, Task 5.3 Profiling Report**  
> **Model:** `BallisticHardPINN` (403,245 parameters)  
> **Evaluated Device:** CPU (Intel/AMD x86_64)

---

## 1. Executive Summary

| Metric | Numerical Integrator (Numba RK4) | Neural Surrogate (Eager) | TorchScript JIT |
| :--- | :--- | :--- | :--- |
| **Single-Query Latency (P50)** | 6.14 ms | 0.475 ms (475 $\mu$s) | **1.030 ms (1030 $\mu$s)** |
| **95th Percentile Latency (P95)** | 13.35 ms | 0.806 ms | **1.976 ms** |
| **Single-Stream QPS** | 162.9 qps | 1915 qps | **813 qps** |
| **Batched Throughput (B=10k)** | 162.9 qps | **108,284 queries/sec** | — |
| **Peak Throughput Speedup** | $1.0\times$ | **664.5$\times$** | — |

**Acceptance Criteria Status:** **PASSED** (Achieved **665x** speedup over numerical simulation, far exceeding the $\ge 500\times$ target requirement).

---

## 2. Batched Throughput Scaling

| Batch Size | Batch Latency (ms) | Inference Throughput (queries/sec) | Speedup vs Numerical RK4 |
| :---: | :---: | :---: | :---: |
| 1 | 0.48 ms | 2,080 qps | **12.8x** |
| 64 | 0.91 ms | 70,567 qps | **433.1x** |
| 256 | 1.62 ms | 157,705 qps | **967.8x** |
| 1,024 | 3.93 ms | 260,674 qps | **1,599.8x** |
| 10,000 | 92.35 ms | 108,284 qps | **664.5x** |

---

## 3. Key Findings & Deployment Recommendations

1. **Sub-Millisecond Interactive Queries:** Single-query latency is 475 $\mu$s (0.475 ms), making it suitable for real-time trajectory guidance and digital twins.
2. **Massive Monte Carlo & Optimization Scalability:** For ballistic limit $V_{50}$ search and Latin Hypercube vulnerability scanning, evaluating 10,000 parameter combinations takes only **92.4 ms** (delivering **108,284 evaluations per second**).
3. **Deployment Artifact:** The traced model has been serialized to `models/pinn_production_traced.pt` for zero-dependency C++ / LibTorch production integration.
