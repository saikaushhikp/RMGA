# Literature Review and Benchmark Overview

## 1. Introduction

Test-Time Adaptation (TTA) sits at the intersection of domain adaptation, continual learning, and robust inference. This section surveys the key methods relevant to our project — TENT, BN-Adapt, NOTE, EATA, and CoTTA — with a critical focus on their limitations under **edge-device compute constraints** and **batch-size-1 streaming** scenarios. We also describe our benchmark construction.

---

## 2. Core TTA Methods

### 2.1 TENT: Test Entropy Minimization
**Paper**: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy Minimization", ICLR 2021.

**Core Idea**: TENT adapts a model at test time by minimizing the Shannon entropy of its predictions on incoming unlabelled test data. Formally, it minimizes:

```
L_TENT = - Σ_k p_k log(p_k)
```

Only the affine parameters (scale γ and shift β) of Batch Normalization layers are updated. All other parameters remain frozen. This is a deliberate design choice: BN affine parameters are small in number, easy to update, and primarily responsible for domain-specific feature scaling.

**Strengths**:
- Simple, elegant, and effective on standard benchmarks (CIFAR-10-C, ImageNet-C).
- Low number of trainable parameters (only BN affine params).
- Self-supervised — requires no labels.

**Limitations for Our Setting**:
- **Batch-size-1 failure**: BN statistics (mean and variance) become degenerate for N=1 (variance = 0). TENT's gradient signal is computed over mini-batches; a single-sample entropy gradient is highly noisy.
- **No temporal regularization**: TENT treats each batch as i.i.d. It has no mechanism to leverage temporal continuity in video streams.
- **Catastrophic forgetting**: Prolonged TENT adaptation on a single corruption domain can degrade performance on other domains due to BN parameter drift.
- **Edge compute**: Standard TENT assumes GPU resources sufficient to process batches of 64+ images simultaneously, which is infeasible in batch-size-1 streaming on low-VRAM devices.

---

### 2.2 BN-Adapt (Test-Time Batch Normalization)
**Paper**: Schneider et al., "Improving robustness against common corruptions by covariate shift adaptation", NeurIPS 2020; also related to Nado et al., 2020.

**Core Idea**: Rather than learning new BN parameters, BN-Adapt simply **replaces the stored training-time running statistics** (μ_train, σ²_train) with statistics estimated from the test stream. Typically, this is done by running the test data through the model in evaluation mode but re-estimating batch statistics from a held-out set of test samples.

**Strengths**:
- No gradient computation needed — extremely fast and memory efficient.
- Often sufficient for moderate distribution shifts (e.g., common image corruptions).
- No risk of catastrophic forgetting since no parameters are updated via backpropagation.

**Limitations for Our Setting**:
- **Still batch-dependent**: Reliable statistic estimation requires accumulating sufficient samples (typically 32–256). At batch-size-1, single-sample estimates are too noisy.
- **Static adaptation**: Once statistics are estimated, they are fixed. It cannot adapt to *gradual* or *non-stationary* shifts without periodic re-estimation.
- **No entropy optimization**: BN-Adapt does not optimize for prediction confidence, only feature distribution alignment.

---

### 2.3 NOTE: Non-i.i.d. Test-Time Adaptation
**Paper**: Gong et al., "NOTE: Robust Continual Test-Time Adaptation Against Temporal Correlation", NeurIPS 2022.

**Core Idea**: NOTE directly addresses the i.i.d. violation in real-world test streams. It maintains an **instance-aware BN** that mixes instance normalization (independent of batch) with batch normalization, controlled by a learnable mixing coefficient. It also introduces **PBRS (Prediction-Balanced Reservoir Sampling)** to maintain a class-balanced memory bank of recent samples, correcting for temporal correlation in gradient updates.

**Strengths**:
- Explicitly designed for non-i.i.d. test streams — the most directly relevant baseline.
- Instance-aware BN can operate at batch-size-1 without variance collapse.
- PBRS improves gradient quality by sampling diverse recent predictions.

**Limitations for Our Setting**:
- **Memory bank overhead**: PBRS requires storing a substantial number of past samples (features or images). On a 4 GB VRAM device, this can be prohibitive for larger feature maps.
- **Complexity**: The learnable mixing coefficient adds a hyperparameter and extra computation compared to standard TENT.
- **Still struggles with extreme domain shift**: When the corruption changes rapidly (e.g., blur intensity changes every 10 frames), the memory bank may contain stale/irrelevant samples.
- **No explicit temporal smoothness**: NOTE handles non-i.i.d. correlation via resampling, not by explicitly modelling entropy smoothness over time.

---

### 2.4 EATA: Efficient Anti-Forgetting Test-Time Adaptation
**Paper**: Niu et al., "Efficient Test-Time Model Adaptation without Forgetting", ICML 2022.

**Core Idea**: EATA introduces two mechanisms on top of TENT: (1) **Sample-efficient entropy filtering** — only adapt on samples where the model is neither too confident (low entropy) nor too uncertain (high entropy), as these represent the most informative and reliable samples; (2) **Fisher information regularization** — penalize large deviations of adapted parameters from their original values, weighted by their Fisher information, to prevent catastrophic forgetting.

**Strengths**:
- Efficient: skips adaptation on uninformative samples, reducing compute.
- Anti-forgetting: Fisher regularization prevents over-adaptation.
- Strong results on long-horizon adaptation benchmarks (e.g., ImageNet-C with all 15 corruptions sequentially).

**Limitations for Our Setting**:
- **Fisher information computation cost**: Computing and storing Fisher information (or a diagonal approximation) for all BN parameters adds significant memory overhead — problematic on RTX 3050.
- **Sample filtering assumes i.i.d.**: The entropy thresholds for "reliable" samples are calibrated assuming a diverse batch. In a temporally-correlated stream, all frames may cluster in one entropy regime, causing EATA to either skip everything or adapt on everything.
- **Batch-size-1**: Like TENT, EATA computes gradients over a mini-batch. At N=1, the Fisher approximation is poor and gradient variance is high.

---

### 2.5 CoTTA: Continual Test-Time Adaptation
**Paper**: Wang et al., "Continual Test-Time Domain Adaptation", CVPR 2022.

**Core Idea**: CoTTA uses two mechanisms to handle continual, non-stationary test streams: (1) **Augmentation-averaged predictions** — the teacher model makes predictions by averaging over multiple augmented versions of each test image, providing a stable pseudo-label; (2) **Stochastic weight restore** — a small fraction of neurons are randomly reset to their original pre-trained values at each step, preventing drift and maintaining plasticity.

**Strengths**:
- Explicitly designed for continually-changing test distributions.
- Stochastic restore provides a principled anti-forgetting mechanism.
- Teacher-student framework stabilizes pseudo-labels.

**Limitations for Our Setting**:
- **High compute cost**: Augmentation-averaged predictions require multiple (typically 32–64) forward passes per test sample. At batch-size-1, this means 32–64× the latency of a standard forward pass — completely infeasible for real-time video.
- **Memory**: Storing both student and teacher model weights doubles VRAM usage.
- **Latency**: Multi-pass inference cannot meet the real-time frame processing requirements of streaming video on edge hardware.

---

## 3. Comparative Summary

| Method | Handles N=1 | Temporal Regularization | Edge-Friendly | Anti-Forgetting |
|--------|------------|------------------------|---------------|-----------------|
| TENT | ✗ (collapses) | ✗ | ✓ (lightweight) | ✗ |
| BN-Adapt | Partial | ✗ | ✓✓ (no backprop) | ✓ (no update) |
| NOTE | ✓ (instance BN) | Partial (PBRS) | ✗ (memory bank) | Partial |
| EATA | ✗ (poor Fisher est.) | ✗ | ✗ (Fisher cost) | ✓ |
| CoTTA | ✗ (multi-pass) | ✗ | ✗✗ (32–64× cost) | ✓ |
| **TEC (ours)** | **✓ (buffer)** | **✓✓ (explicit)** | **✓✓** | **Partial** |

---

## 4. Benchmark Construction

### 4.1 Tiny-ImageNet-C (Static Corruptions)

**Tiny-ImageNet-C** applies the 15 corruption types from Hendrycks & Dietterich (2019) at 5 severity levels to the 10,000-image Tiny-ImageNet validation set (64×64 resolution, 200 classes). We use this as a controlled testbed for baseline evaluation (Weeks 1–2), where we measure performance degradation under each corruption type independently.

- **Resolution**: 64×64 (the native resolution of Tiny-ImageNet). Note that the baseline implementation defaults to 224×224 input size (as expected by ImageNet-pretrained MobileNetV2); images will be upsampled by torchvision transforms when loading real Tiny-ImageNet-C data.
- **Classes**: 200 (manageable for MobileNetV2 adapted from ImageNet).
- **Corruptions used**: Gaussian noise, shot noise, impulse noise, defocus blur, glass blur, motion blur, zoom blur, snow, frost, fog, brightness, contrast, elastic transform, pixelate, JPEG compression.

### 4.2 Simulated ImageNet-Vid-C (Gradual Streaming Corruptions)

To simulate a realistic video stream with **gradual distribution shift**, we construct a synthetic video dataset by dynamically applying corruptions with **time-varying severity**:

```python
# Pseudo-code for severity schedule
severity_t = base_severity + A * sin(2π * t / T) + ε_t
```

where:
- `base_severity` is the baseline corruption level (e.g., 2 out of 5).
- `A` controls the amplitude of gradual shift.
- `T` is the period over which severity cycles (e.g., 200 frames).
- `ε_t` is small random noise to simulate frame-level jitter.

**Focus corruptions for gradual shift experiments**:
- **Defocus blur** (simulates camera focus drift in autonomous driving).
- **Motion blur** (simulates varying vehicle speed).
- **Fog** (simulates weather changes over a drive).
- **Snow** (simulates weather transitions).

This simulated stream provides a controlled environment where:
1. The "ground truth" corruption trajectory is known (useful for analysis).
2. The shift is gradual (reflecting real video dynamics).
3. No proprietary or large-scale video dataset is required (accessibility for edge hardware).

### 4.3 Evaluation Protocol

- **Streaming accuracy**: Top-1 accuracy computed over the entire stream (one prediction per frame, no batching).
- **Temporal accuracy curve**: Accuracy in a sliding window of 50 frames, plotted over time to visualize collapse events.
- **Average entropy over time**: To diagnose collapse (entropy → maximum = log(200) ≈ 5.3 nats for random predictions).
- **Computation cost**: Average milliseconds per frame (forward + backward pass) on RTX 3050.

---

## 5. Positioning Our Contribution

Existing work does not simultaneously satisfy all of the following requirements:
1. Operates at batch-size-1 without BN variance collapse.
2. Leverages temporal continuity as an explicit regularization signal.
3. Meets edge-hardware compute constraints (< 20 ms per frame on RTX 3050).
4. Does not require a memory bank of past images.

**TEC addresses all four** by replacing the i.i.d. batch assumption with a lightweight scalar entropy buffer, enabling stable entropy gradient signals through temporal smoothing. This is, to our knowledge, the first TTA method to explicitly model the temporal entropy profile of a video stream as a first-class regularization objective.
