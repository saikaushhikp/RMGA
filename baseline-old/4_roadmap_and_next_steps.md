# Project Roadmap and Next Steps

## Overview

This document outlines the 6-week research and development roadmap for the **Temporal Entropy-Consistency (TEC) for Batch-Size-1 Video TTA** project. Each phase builds on the previous, progressing from baseline failure analysis through full method development and ablation studies.

---

## Week 1–2: Establishing Baseline Failures

### Goals
- Confirm that standard TTA methods (TENT, BN-Adapt) fail at batch-size-1 on the simulated video stream.
- Establish quantitative failure baselines to compare against in later weeks.
- Set up the full experimental pipeline (data, model, logging, metrics).

### Tasks

#### Week 1: Infrastructure Setup
- [ ] Set up Python virtual environment with PyTorch, torchvision, numpy, matplotlib, tqdm.
- [ ] Download and verify **Tiny-ImageNet-C** dataset; write the streaming DataLoader (batch_size=1, sequential ordering).
- [ ] Load pre-trained **MobileNetV2** (ImageNet weights from torchvision); verify forward pass on a single frame.
- [ ] Implement the **Source-Only** baseline: frozen model, no adaptation; measure Top-1 accuracy on each corruption type at all 5 severities.
- [ ] Set up logging infrastructure: per-frame accuracy, running entropy, and time-per-frame CSV logs.

#### Week 2: Baseline Failure Experiments
- [ ] Implement **BN-Adapt** baseline: update running mean/variance from test stream; measure accuracy vs. frame index.
- [ ] Implement **Standard TENT** (batch_size=1): update BN affine params via entropy minimization; document the collapse behavior (accuracy curve, entropy curve).
- [ ] Generate **failure analysis plots**:
  - Top-1 accuracy over frame index for each baseline (Source-Only, BN-Adapt, TENT).
  - Entropy over frame index: show TENT entropy spike and subsequent collapse.
  - BN variance statistics over time: show that batch variance = 0 at N=1.
- [ ] Write up **baseline failure report** summarizing quantitative results and collapse diagnostics.

### Deliverables
- `baseline_implementation.py` — fully functional with all 3 baseline conditions.
- `results/baseline_failure_plots/` — figures showing collapse behavior.
- `results/baseline_accuracy_tables.csv` — per-corruption accuracy numbers.

---

## Week 3–4: Developing TEC Core Components

### Goals
- Design and implement the **sliding window entropy buffer**.
- Implement the **TEC loss function** and verify it provides stable gradient signals.

### Tasks

#### Week 3: Sliding Window Entropy Buffer
- [ ] Implement `EntropyBuffer` class: fixed-size circular buffer storing scalar entropy values.
  - Operations: `push(H_t)`, `mean()`, `std()`, `is_ready()` (returns True once buffer is full).
- [ ] Validate buffer behavior with unit tests:
  - Correct FIFO behavior (oldest value dropped when buffer is full).
  - Correct mean/std computation.
  - Graceful handling of edge cases (empty buffer, single element).
- [ ] Integrate buffer into the adaptation loop: after each forward pass, compute `H(p_t)` and push to buffer.
- [ ] Plot entropy buffer mean over time vs. raw frame entropy: verify smoothing behavior.

#### Week 4: TEC Loss Function
- [ ] Implement `tec_loss(logits, buffer, lambda_consistency)`:
  ```python
  H_t = entropy(softmax(logits))
  H_mean = buffer.mean()
  consistency_penalty = (H_t - H_mean) ** 2
  loss = H_t + lambda_consistency * consistency_penalty
  ```
- [ ] Verify that `tec_loss` gradients are:
  - Smaller in magnitude than raw TENT loss at batch-size-1 (sanity check).
  - Non-zero (i.e., no degenerate cases with zero gradients).
- [ ] Run initial TEC experiments on the simulated blur stream:
  - Compare TEC (λ=0.1, W=8) vs. TENT on 500-frame stream.
  - Plot accuracy and entropy curves side-by-side.
- [ ] Tune initial values of `W` (window size) and `λ` (consistency weight) via a coarse grid search.

### Deliverables
- `tec/entropy_buffer.py` — EntropyBuffer class with unit tests.
- `tec/tec_loss.py` — TEC loss function.
- `results/tec_vs_tent_initial/` — initial comparison plots.

---

## Week 5: Selective Backpropagation Integration

### Goals
- Implement the **selective backpropagation gate** that skips adaptation on anomalous frames.
- Integrate all TEC components into a unified adaptation loop.
- Evaluate the complete TEC method on the full simulated video stream.

### Tasks

- [ ] Implement `SelectiveAdapter` class:
  - Takes `logits`, `buffer`, `lambda_consistency`, `tau` (collapse threshold) as input.
  - Computes TEC loss; if `(H_t - H_mean)² > tau`, skip backpropagation and return without updating parameters.
  - Otherwise, compute gradients and apply Adam/SGD update to BN affine parameters.
- [ ] Implement adaptive threshold τ:
  - Initial value: `tau = 3 * buffer.std()` (anomaly detection based on buffer statistics).
  - Fall back to a fixed `tau = 2.0` when buffer is not yet full.
- [ ] Integration test:
  - Run `SelectiveAdapter` on the 500-frame blur stream.
  - Log the fraction of frames where backpropagation is skipped (expect ~10–20% for gradual shift).
- [ ] Benchmarking:
  - Measure average ms/frame on RTX 3050 (target: < 20 ms).
  - Measure peak VRAM usage (target: < 2 GB).
- [ ] Full stream evaluation:
  - Run TEC on all 4 focus corruptions (blur, motion blur, fog, snow) at severity schedules.
  - Compare against Source-Only, BN-Adapt, and TENT (when TENT does not collapse).

### Deliverables
- `tec/selective_adapter.py` — SelectiveAdapter class.
- `tec/tec_adaptation_loop.py` — unified TEC adaptation loop.
- `results/tec_full_evaluation/` — per-corruption accuracy tables and plots.

---

## Week 6: Comprehensive Evaluation and Ablation Studies

### Goals
- Evaluate TEC across **all 15 Tiny-ImageNet-C corruption types** (not just the 4 focus types).
- Conduct **ablation studies** on key hyperparameters: window size W and consistency weight λ.
- Prepare a final results summary and write the project report.

### Tasks

#### Comprehensive Evaluation
- [ ] Run all methods (Source-Only, BN-Adapt, TENT, TEC) on all 15 corruption types at all 5 severity levels.
- [ ] Compute:
  - **Mean Corruption Error (mCE)**: standard benchmark metric for robustness.
  - **Accuracy vs. frame index** for each method × corruption × severity.
  - **Collapse rate**: fraction of experiments where TENT entropy exceeds 0.9 × H_max.
- [ ] Generate final comparison table (Table 1 of future paper).

#### Ablation Study: Window Size W
- [ ] Evaluate TEC with W ∈ {4, 8, 16, 32} on the defocus blur stream.
- [ ] Metrics: accuracy, average entropy, fraction of skipped frames.
- [ ] Plot: accuracy vs. W bar chart; entropy curve for each W value.

#### Ablation Study: Consistency Weight λ
- [ ] Evaluate TEC with λ ∈ {0.01, 0.1, 0.5, 1.0} with best W from above.
- [ ] Metrics: same as above.
- [ ] Plot: accuracy vs. λ; gradient magnitude distributions for each λ.

#### Report Writing
- [ ] Write project report sections:
  - Introduction (from `1_problem_statement.md`).
  - Related Work (from `2_literature_and_benchmarks.md`).
  - Method: TEC formulation, buffer design, selective backpropagation.
  - Experiments: setup, baseline failures, TEC results, ablations.
  - Conclusion and future work.
- [ ] Prepare figures for report: all accuracy/entropy plots, ablation charts, architecture diagram of TEC.

### Deliverables
- `results/final_tables/` — all quantitative results in CSV/LaTeX format.
- `results/ablation_plots/` — window size and lambda ablation figures.
- `report/tec_report_draft.pdf` — complete project report draft.

---

## Timeline Summary

| Week | Phase | Key Milestone |
|------|-------|---------------|
| 1 | Infrastructure + Source-Only | Pipeline running, Source-Only accuracy measured |
| 2 | Baseline Failures | TENT collapse documented, BN-Adapt evaluated |
| 3 | Entropy Buffer | EntropyBuffer implemented and validated |
| 4 | TEC Loss | TEC loss functional, initial improvement shown |
| 5 | Selective Backprop | Full TEC method integrated and benchmarked |
| 6 | Evaluation + Report | All corruptions evaluated, ablations complete, report drafted |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MobileNetV2 performs too well on Tiny-ImageNet-C even without adaptation | Low | High (no problem to solve) | Use severity 5 corruptions; pre-test before full study |
| TENT does NOT collapse at batch-size-1 on some corruptions | Medium | Medium | Document conditions; TEC should still provide smoother adaptation |
| RTX 3050 VRAM insufficient for any forward+backward pass | Low | High | Use gradient checkpointing; reduce image resolution to 32×32 |
| Gradual severity schedule too slow to show meaningful shift | Medium | Low | Increase schedule amplitude A; shorten period T |
| TEC hyperparameter sensitivity | Medium | Medium | Run coarse grid search in Week 4; report sensitivity explicitly |
