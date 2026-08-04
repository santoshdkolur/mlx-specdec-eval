# `mlx-specdec-eval` User Questionnaire & Discussion Backlog

This document tracks open questions, design topics, and concepts to revisit in future sessions.

---

## 📋 Open Questions & Topics to Revisit

### 1. `num-draft-tokens` ($K$) Deep Dive & Mathematical Profiling
- **Date Added:** 2026-08-02
- **Status:** 🟡 Pending Further Discussion
- **Topic:** Deep dive into how `num-draft-tokens` ($K$) affects speculative inference performance.
- **Key Discussion Points to Revisit:**
  1. **Optimal Draft Length Bounds:** How to mathematically model the sweet spot for $K$ given draft model latency $T_D$, target verification latency $T_T$, and token acceptance rate $\alpha$.
  2. **Domain Variability:** Analyzing how $K$ sensitivity changes across rigid code syntax vs unstructured prose.
  3. **Memory Bandwidth Thresholds:** Profiling KV cache memory footprint expansion as $K$ increases on Apple Silicon Unified Memory.
  4. **Dynamic $K$ Adaptation:** Exploring potential support for adaptive $K$ scheduling (dynamically scaling $K$ up or down during generation based on real-time acceptance rate $\alpha$).

---

## 🔄 How to Update This File

When new questions arise during benchmarking or development, append them to this file under a new numbered section with status set to `🟡 Pending Further Discussion`.
