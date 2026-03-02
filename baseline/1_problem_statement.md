# Problem Statement: The Batch-Size-1 Conundrum in Test-Time Adaptation

## 1. Overview

Test-Time Adaptation (TTA) has emerged as a compelling paradigm to handle distribution shift between training data and real-world deployment data. The fundamental premise is straightforward: rather than re-training or fine-tuning a model on a labelled target dataset, TTA methods update model parameters *during inference* using only the unlabelled test samples themselves.

The dominant family of TTA methods — including **TENT** (Test Entropy Minimization), **BN-Adapt** (Batch Normalization Statistics Adaptation), **EATA**, and **CoTTA** — all share a critical architectural assumption: **test samples arrive in reasonably-sized mini-batches (typically 64–256 images)**. This assumption allows the methods to exploit reliable batch-level statistics (mean and variance) for Batch Normalization (BN) layers and to compute informative gradient signals from aggregated entropy across multiple samples.

This assumption, however, **catastrophically fails** in real-world streaming video deployment scenarios.

---

## 2. The Streaming Video Problem

### 2.1 Deployment Context

Consider an autonomous driving vehicle, a surveillance system, or a robotic agent navigating an environment. In each of these scenarios:

- The model processes **one frame at a time** (batch size = 1) to meet strict latency requirements.
- Frames arrive **sequentially** and exhibit **strong temporal correlation** — neighbouring frames are nearly identical in appearance.
- The visual distribution shifts **gradually** over time (e.g., fog gradually thickens, motion blur increases with vehicle speed, daylight transitions to dusk).
- The system operates on **edge hardware** (e.g., NVIDIA Jetson, RTX 3050 laptop GPU) with severe memory and compute constraints.

### 2.2 Why Batch-Size-1 Breaks Standard TTA

#### Batch Normalization Variance Collapse

Standard BN computes the mean (μ) and variance (σ²) over a mini-batch of N samples:

```
μ_B = (1/N) Σ x_i
σ²_B = (1/N) Σ (x_i - μ_B)²
```

When **N = 1**, this degenerates immediately:
- `μ_B = x_1`  (the single sample is its own mean)
- `σ²_B = 0`   (variance of a single point is always zero)

The normalized activation then becomes:

```
x̂ = (x - μ_B) / sqrt(σ²_B + ε) → (x - x) / sqrt(0 + ε) = 0 / sqrt(ε) ≈ 0
```

Every neuron collapses to near-zero activation regardless of the input. The affine rescaling `γ * x̂ + β` then simply outputs the learned bias `β`. **The network loses all discriminative power.**

#### Entropy Gradient Instability

TENT minimizes the Shannon entropy of the model's predictive distribution:

```
H(p) = - Σ_k p_k log(p_k)
```

With N = 1, the entropy gradient `∂H/∂θ` is computed from a single, potentially noisy prediction. This produces:

- **High gradient variance**: A single misclassified or uncertain frame generates a large erroneous gradient update.
- **Positive feedback loops**: An incorrect update lowers confidence in the correct class, which raises entropy further, which triggers another large incorrect update — a classic **model collapse spiral**.
- **No statistical averaging**: In a batch of N samples, noise in individual gradients tends to cancel out. With N = 1, there is no such averaging.

#### Non-IID Temporal Correlation

Video frames are temporally correlated: frame t+1 is nearly identical to frame t. From a statistical learning standpoint, consecutive frames are **not independent and identically distributed (i.i.d.)** samples from the target distribution. Methods that assume i.i.d. batches (e.g., for diversity-based sample selection in EATA) are misled by this redundancy.

---

## 3. The Temporal Continuity Opportunity

While temporal correlation is a *problem* for standard batch-based methods, it is also an *opportunity*. The near-continuity of video frames implies:

1. **Entropy should change smoothly**: The model's uncertainty about frame t should be similar to its uncertainty about frame t+1 (unless a genuine scene change occurs).
2. **Prediction distributions evolve gradually**: A sudden spike in entropy most likely indicates a corrupted frame or a noisy gradient signal, *not* a genuine distribution shift.
3. **Historical frames carry useful statistical context**: A buffer of recent frames can serve as a *virtual batch*, providing more stable statistics than a single frame.

---

## 4. Proposed Solution: Temporal Entropy-Consistency (TEC)

We propose **Temporal Entropy-Consistency (TEC)**, a lightweight extension to TENT that addresses the batch-size-1 instability by leveraging temporal smoothness as a regularization signal.

### 4.1 Core Components

#### (a) Sliding Window Entropy Buffer

Instead of adapting on each frame independently, TEC maintains a fixed-size circular buffer `B` of the entropy values from the `W` most recent frames:

```
B_t = { H(p_{t-W+1}), H(p_{t-W+2}), ..., H(p_t) }
```

The buffer provides a reference distribution for how uncertain the model *should be* on the current frame, based on recent history.

#### (b) Temporal Entropy Consistency Loss

The TEC loss penalizes abrupt deviations from the temporal entropy baseline:

```
L_TEC = H(p_t) + λ * (H(p_t) - mean(B_{t-1}))²
```

where:
- `H(p_t)` is the standard TENT entropy minimization term.
- `mean(B_{t-1})` is the exponential moving average of the W most recent entropy values (excluding the current frame).
- `λ` is a consistency weight hyperparameter (to be ablated; initial value: 0.1).

The squared deviation term acts as a **temporal anchor**: it suppresses large gradient updates when the current frame's entropy deviates strongly from recent history, which is the primary cause of collapse.

#### (c) Selective Backpropagation Gate

A hard-threshold gate prevents any adaptation on frames where the TEC loss exceeds a collapse-risk threshold `τ`:

```
if (H(p_t) - mean(B_{t-1}))² > τ:
    skip backpropagation  # frame is too anomalous, do not adapt
else:
    compute ∂L_TEC/∂θ and update θ
```

This mimics the reliable sample filtering of EATA, but adapted to the temporal, single-frame setting.

### 4.2 Computational Overhead

TEC adds minimal overhead:
- **Buffer**: A circular buffer of `W` scalar float values (W ≤ 32 ≈ negligible memory).
- **Extra computation**: One squared difference, one buffer mean — O(W) scalar operations.
- **No additional forward passes**: All required activations are already computed during the standard forward pass.

This makes TEC suitable for deployment on constrained hardware such as an RTX 3050 laptop GPU.

---

## 5. Research Questions

This project aims to answer the following concrete research questions:

| # | Question | Metric |
|---|----------|--------|
| RQ1 | Does standard TENT collapse at batch-size-1 on simulated video corruption streams? | Top-1 Accuracy vs. Time |
| RQ2 | Does TEC prevent collapse while maintaining competitive accuracy? | Top-1 Accuracy, Avg Entropy |
| RQ3 | What is the optimal sliding window size W for TEC? | Ablation: W ∈ {4, 8, 16, 32} |
| RQ4 | What is the optimal consistency weight λ? | Ablation: λ ∈ {0.01, 0.1, 0.5, 1.0} |
| RQ5 | Does TEC generalise across different corruption types? | Accuracy per corruption type |

---

## 6. Scope and Constraints

- **Model**: MobileNetV2 (pre-trained on ImageNet) — chosen for low memory footprint (~14 MB).
- **Dataset**: Tiny-ImageNet-C (simulated gradual corruptions) and ImageNet-Vid-C (simulated video stream).
- **Hardware**: NVIDIA RTX 3050 laptop GPU (4 GB VRAM).
- **Batch size**: Fixed at **1** throughout all experiments.
- **Adaptation scope**: Only Batch Normalization affine parameters (γ, β) are updated — following the TENT protocol.
