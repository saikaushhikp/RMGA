# Execution Guide for `baseline_implementation.py`

This guide explains how to set up your environment, install dependencies, and run the baseline evaluation script for the **Temporal Entropy-Consistency (TEC) for Batch-Size-1 Video TTA** project.

---

## Prerequisites

- **Python**: 3.9 or later (3.10 recommended)
- **OS**: Linux, macOS, or Windows 10/11
- **Hardware**: NVIDIA GPU recommended (RTX 3050 or better); CPU execution is supported but significantly slower
- **Disk space**: ~500 MB for PyTorch + torchvision; ~14 MB for MobileNetV2 weights (downloaded automatically on first run)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/saikaushhikp/VidTENT.git
cd VidTENT
```

---

## Step 2: Create a Python Virtual Environment

Using `venv` (built into Python 3):

```bash
# Create the environment (name it 'vidtent_env' or anything you like)
python -m venv vidtent_env

# Activate the environment
# On Linux / macOS:
source vidtent_env/bin/activate

# On Windows (PowerShell):
.\vidtent_env\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\vidtent_env\Scripts\activate.bat
```

You should see `(vidtent_env)` prefixed in your terminal prompt after activation.

---

## Step 3: Install Dependencies

### 3a. Install PyTorch with CUDA support (for RTX 3050)

The RTX 3050 supports CUDA 11.8 / 12.1. Install the appropriate PyTorch build:

```bash
# For CUDA 12.1 (recommended for RTX 30-series on recent drivers):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (no GPU):
pip install torch torchvision
```

Verify your CUDA installation:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected output (with RTX 3050):
```
CUDA available: True
Device: NVIDIA GeForce RTX 3050 Laptop GPU
```

### 3b. Install other dependencies

```bash
pip install numpy matplotlib tqdm
```

### 3c. (Optional) Create a `requirements.txt` for reproducibility

```bash
pip freeze > requirements.txt
```

---

## Step 4: Navigate to the Baseline Directory

```bash
cd baseline
```

---

## Step 5: Run the Script

The script accepts a `--mode` argument to select which baseline(s) to evaluate.

### Run All Three Baselines Sequentially (recommended for first run):

```bash
python baseline_implementation.py --mode all
```

### Run a Single Baseline:

```bash
# Source-Only (frozen model, no adaptation):
python baseline_implementation.py --mode source_only

# BN-Adapt (update BN running statistics only):
python baseline_implementation.py --mode bn_adapt

# TENT (entropy minimization — EXPECTED TO COLLAPSE):
python baseline_implementation.py --mode tent
```

### Full Argument Reference:

```
python baseline_implementation.py [--mode {source_only,bn_adapt,tent,all}]
                                   [--num_frames NUM_FRAMES]
                                   [--num_classes NUM_CLASSES]
                                   [--image_size IMAGE_SIZE]
                                   [--lr LR]
                                   [--log_interval LOG_INTERVAL]

Arguments:
  --mode          Which baseline(s) to run (default: all)
  --num_frames    Number of frames in the synthetic video stream (default: 500)
  --num_classes   Number of output classes: 1000 for ImageNet, 200 for Tiny-ImageNet (default: 1000)
  --image_size    Input image spatial resolution (default: 224)
  --lr            TENT learning rate (default: 0.001)
  --log_interval  Print progress every N frames (default: 100)
```

### Example: Quick Test with 100 Frames

```bash
python baseline_implementation.py --mode all --num_frames 100 --log_interval 25
```

---

## Step 6: Expected Terminal Outputs

### Startup Output (all modes)

```
=================================================================
  VidTENT — Baseline Evaluation (batch_size=1)
=================================================================
  Mode        : all
  Num frames  : 500
  Num classes : 1000
  Image size  : 224×224
  TENT lr     : 0.001
=================================================================

[Device] Using GPU: NVIDIA GeForce RTX 3050 Laptop GPU
[DataLoader] Streaming loader ready: 500 frames, batch_size=1, image_size=224×224, num_classes=1000.
```

> **Note**: If you see `[Device] CUDA not available — using CPU (expect slower runtimes)`, the script will still run but will be 5–20× slower.

---

### Source-Only Output

```
--- Running Source-Only Baseline ---
[Model] MobileNetV2 loaded (1000 classes).
[Source-Only] Model fully frozen. No adaptation will occur.
  [Source-Only] Frame 0100 | Top-1 Accuracy: 0.00%  |  Mean Entropy: 6.9077 nats  |  Frames: 100
  [Source-Only] Frame 0200 | Top-1 Accuracy: 0.00%  |  Mean Entropy: 6.9077 nats  |  Frames: 200
  ...
[Source-Only] Final: Top-1 Accuracy: 0.00%  |  Mean Entropy: 6.9077 nats  |  Frames: 500
```

> **Why 0% accuracy?** The synthetic dataloader generates random noise images with random labels. A model pre-trained on real ImageNet images will have near-zero accuracy on pure noise — this is expected and correct. Replace the `SyntheticVideoStream` class with your actual Tiny-ImageNet-C loader to get meaningful accuracy values.

> **Why entropy ≈ 6.91?** The maximum entropy for 1000 classes is `log(1000) ≈ 6.91 nats`. Random noise inputs produce maximally uncertain (uniform) predictions — which is expected.

---

### BN-Adapt Output

```
--- Running BN-Adapt Baseline ---
[Model] MobileNetV2 loaded (1000 classes).
[BN-Adapt] BatchNorm layers set to update running statistics from test stream.
  [BN-Adapt] Frame 0100 | Top-1 Accuracy: 0.00%  |  Mean Entropy: 6.9077 nats  |  Frames: 100
  ...
[BN-Adapt] Final: Top-1 Accuracy: 0.00%  |  Mean Entropy: 6.9077 nats  |  Frames: 500
```

> BN-Adapt will show slightly different entropy values than Source-Only as running statistics adapt to the noise distribution. On real corrupted images, BN-Adapt typically outperforms Source-Only significantly.

---

### TENT Output (Collapse Demonstration)

```
--- Running TENT Baseline (WARNING: expect collapse at batch_size=1) ---
[Model] MobileNetV2 loaded (1000 classes).
[TENT] 52 BN affine parameter tensors to optimise, lr=0.001.
[TENT] WARNING: Expected to collapse at batch_size=1 due to BN variance=0.

  [TENT] Frame 0100 | Loss: 6.9077 | Top-1 Accuracy: 0.00% | Mean Entropy: 6.9077 nats | Elapsed: 8.3s
  [TENT] Frame 0200 | Loss: 6.9077 | Top-1 Accuracy: 0.00% | Mean Entropy: 6.9077 nats | Elapsed: 16.7s

[TENT] No NaN collapse detected on synthetic data.
Note: On real corrupted data (Tiny-ImageNet-C), collapse is expected
due to degenerate BN statistics at batch_size=1.

[TENT] Final: Top-1 Accuracy: 0.00%  |  Mean Entropy: 6.9077 nats  |  Frames: 500
```

> **On real data (Tiny-ImageNet-C or ImageNet-C), the TENT collapse will manifest as:**
> - Loss becoming `nan` (printed as "COLLAPSE DETECTED at frame N")
> - Top-1 accuracy rapidly dropping toward 0% after an initial period of reasonable performance
> - Mean entropy climbing toward the maximum value (`log(num_classes)`)
> - The model essentially degenerating into outputting a uniform distribution over all classes

---

### Final Comparison Table

```
=================================================================
Baseline              Top-1 Acc (%)    Mean Entropy     Frames
-----------------------------------------------------------------
Source-Only                   0.00%          6.9077        500
BN-Adapt                      0.00%          6.9077        500
TENT                          0.00%          6.9077        500
=================================================================

Note: TENT accuracy degradation (if any) reflects the expected
batch-size-1 collapse. This motivates the TEC method.
```

---

## Step 7: Using Real Data (Tiny-ImageNet-C)

To use real corrupted data instead of the synthetic stream:

1. Download Tiny-ImageNet-C from [hendrycks/robustness](https://github.com/hendrycks/robustness) (Tiny-ImageNet-C) or generate it using the corruption scripts in that repository. The dataset can also be found on Zenodo: https://zenodo.org/record/2536630
2. Replace the `SyntheticVideoStream` class with a custom `Dataset` that reads images from the Tiny-ImageNet-C directory structure.
3. Set `--num_classes 200` when running with Tiny-ImageNet-C.
4. Ensure MobileNetV2 is fine-tuned on clean Tiny-ImageNet (200 classes) before evaluating TTA — the 1000-class ImageNet pre-trained weights will not transfer well to the 200-class evaluation setup.

Example custom dataset skeleton:

```python
from torchvision.datasets import ImageFolder

class TinyImageNetCStream(Dataset):
    def __init__(self, root, corruption, severity, transform=None):
        # root: path to Tiny-ImageNet-C directory
        # corruption: e.g. 'gaussian_noise', 'defocus_blur'
        # severity: 1-5
        path = os.path.join(root, corruption, str(severity))
        self.dataset = ImageFolder(path, transform=transform)
        # Sort by filename to preserve temporal ordering
        self.dataset.samples.sort(key=lambda x: x[0])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'torch'` | PyTorch not installed in active venv | Activate venv and run `pip install torch torchvision` |
| `CUDA out of memory` | VRAM insufficient for 224×224 images | Add `--image_size 64` to use 64×64 resolution |
| `UserWarning: ... weights download` | First-time download of MobileNetV2 weights | Allow the download to complete (~14 MB); check internet connection |
| Script runs very slowly (>5s per frame) | Running on CPU | Install the CUDA version of PyTorch and ensure GPU is detected |
| `[TENT] *** COLLAPSE DETECTED ***` | Expected at batch_size=1 on real data | This is the intended behavior; it motivates TEC |
| `Top-1 Accuracy: 0.00%` on all baselines | Using synthetic random data | Replace `SyntheticVideoStream` with real Tiny-ImageNet-C loader |
