"""
baseline_implementation.py
==========================
Baseline evaluation script for the Temporal Entropy-Consistency (TEC) project.

This script evaluates three Test-Time Adaptation (TTA) baselines on a synthetic
streaming video dataset (batch_size=1) using a pre-trained MobileNetV2 model:

  1. Source-Only  — No adaptation; model weights completely frozen.
  2. BN-Adapt     — Update only Batch Normalization running statistics (mean/variance)
                    from the test stream; no backpropagation.
  3. TENT         — Update BN affine parameters (gamma, beta) by minimising Shannon
                    entropy; backpropagation at every frame (expected to COLLAPSE).

Hardware target : NVIDIA RTX 3050 laptop (4 GB VRAM)
Model           : MobileNetV2 (ImageNet pre-trained, torchvision)
Dataset         : Synthetic streaming dataloader (placeholder — replace with
                  Tiny-ImageNet-C or your own corrupted dataset)
Batch size      : 1 (fixed throughout all baselines)

Usage
-----
  python baseline_implementation.py --mode source_only
  python baseline_implementation.py --mode bn_adapt
  python baseline_implementation.py --mode tent
  python baseline_implementation.py --mode all

Expected behaviour
------------------
  source_only : Prints per-frame accuracy; no crash.
  bn_adapt    : Prints per-frame accuracy; no crash.
  tent        : May print NaN losses or rapidly degrading accuracy after a few
                hundred frames — this is the intentional "collapse" demonstration.

Author  : VidTENT Project
Date    : 2026
"""

import argparse
import copy
import math
import random
import sys
import time
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# 0.  Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# 1.  Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU.

    On an RTX 3050 laptop the model's forward+backward pass fits comfortably
    in < 2 GB VRAM even at batch_size=1.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available — using CPU (expect slower runtimes).")
    return device


# ---------------------------------------------------------------------------
# 2.  Model loading
# ---------------------------------------------------------------------------

def load_mobilenetv2(num_classes: int = 1000, device: Optional[torch.device] = None) -> nn.Module:
    """Load a pre-trained MobileNetV2 from torchvision.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 1000 (ImageNet).
        If using Tiny-ImageNet (200 classes), a final-layer replacement is needed
        (see `replace_classifier` below).
    device : torch.device, optional
        Target device. Defaults to CPU if not provided.

    Returns
    -------
    nn.Module
        MobileNetV2 model moved to `device`, in eval mode.
    """
    if device is None:
        device = torch.device("cpu")

    # Load ImageNet pre-trained weights (downloads ~14 MB on first run).
    # Requires torchvision >= 0.13 for the enum-based weights API.
    # For older versions, replace with: models.mobilenet_v2(pretrained=True)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # If Tiny-ImageNet is used (200 classes), replace the final classifier head.
    if num_classes != 1000:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        print(f"[Model] Replaced classifier head for {num_classes} classes.")

    model = model.to(device)
    model.eval()
    print(f"[Model] MobileNetV2 loaded ({num_classes} classes).")
    return model


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final linear layer of MobileNetV2 for a different class count.

    Call this function before loading fine-tuned weights for Tiny-ImageNet-C.

    Parameters
    ----------
    model : nn.Module
        MobileNetV2 instance.
    num_classes : int
        Target number of output classes.

    Returns
    -------
    nn.Module
        Model with updated classifier.
    """
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# 3.  Synthetic streaming dataloader (placeholder)
# ---------------------------------------------------------------------------

class SyntheticVideoStream(Dataset):
    """Placeholder dataset simulating a corrupted video stream at batch_size=1.

    In a real experiment, replace this class with a loader for Tiny-ImageNet-C
    or ImageNet-Vid-C frames. The key property is that frames are returned
    **sequentially** (no shuffling) to simulate temporal ordering.

    Each sample is a random tensor of shape (3, 224, 224) labelled with a
    random class in [0, num_classes). This synthetic data will yield near-
    chance accuracy for all baselines — it exists solely to demonstrate the
    adaptation pipeline without requiring a downloaded dataset.

    Parameters
    ----------
    num_frames : int
        Number of frames in the stream. Default 500.
    num_classes : int
        Number of output classes. Default 1000 (ImageNet).
    image_size : int
        Spatial resolution (square). Default 224.
    corruption_shift : bool
        If True, shifts the pixel distribution gradually over time to simulate
        a gradual corruption (e.g., increasing Gaussian noise variance).
    """

    def __init__(
        self,
        num_frames: int = 500,
        num_classes: int = 1000,
        image_size: int = 224,
        corruption_shift: bool = True,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.image_size = image_size
        self.corruption_shift = corruption_shift

        # Pre-generate labels once (fixed for reproducibility)
        self.labels: List[int] = [
            random.randint(0, num_classes - 1) for _ in range(num_frames)
        ]

        # ImageNet normalisation constants
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return self.num_frames

    def _get_noise_std(self, frame_idx: int) -> float:
        """Return a gradually increasing noise standard deviation.

        Simulates a gradual corruption shift (e.g., increasing blur/noise).
        Severity cycles sinusoidally with a period of 200 frames.
        """
        if not self.corruption_shift:
            return 0.1
        # Base std=0.1, amplitude=0.1, period=200 frames
        return 0.1 + 0.1 * math.sin(2 * math.pi * frame_idx / 200.0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Return a single (image_tensor, label) pair.

        The image is a random base image plus frame-specific Gaussian noise
        to simulate a corrupted video frame.
        """
        noise_std = self._get_noise_std(idx)

        # Random "natural image" base (uniform in [0, 1])
        base_image = torch.rand(3, self.image_size, self.image_size)

        # Add frame-specific Gaussian noise (corruption)
        noise = torch.randn_like(base_image) * noise_std
        corrupted_image = (base_image + noise).clamp(0.0, 1.0)

        # Apply ImageNet normalisation
        image_tensor = self.normalize(corrupted_image)

        return image_tensor, self.labels[idx]


def build_streaming_dataloader(
    num_frames: int = 500,
    num_classes: int = 1000,
    image_size: int = 224,
    corruption_shift: bool = True,
) -> DataLoader:
    """Construct the streaming DataLoader with batch_size=1 and no shuffling.

    Parameters
    ----------
    num_frames : int
        Total number of frames in the stream.
    num_classes : int
        Number of output classes.
    image_size : int
        Spatial resolution.
    corruption_shift : bool
        Whether to apply a gradual corruption shift.

    Returns
    -------
    DataLoader
        DataLoader configured for streaming (batch_size=1, shuffle=False).
    """
    dataset = SyntheticVideoStream(
        num_frames=num_frames,
        num_classes=num_classes,
        image_size=image_size,
        corruption_shift=corruption_shift,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,          # <-- Key constraint: single frame at a time
        shuffle=False,         # <-- Sequential ordering: simulate temporal stream
        num_workers=0,         # Use 0 to avoid multiprocessing overhead on edge devices
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f"[DataLoader] Streaming loader ready: {len(dataset)} frames, "
        f"batch_size=1, image_size={image_size}×{image_size}, "
        f"num_classes={num_classes}."
    )
    return loader


# ---------------------------------------------------------------------------
# 4.  Metrics tracking
# ---------------------------------------------------------------------------

class MetricsTracker:
    """Tracks Top-1 accuracy and per-frame statistics over a streaming evaluation.

    Attributes
    ----------
    correct : int
        Cumulative number of correct predictions.
    total : int
        Cumulative number of predictions.
    per_frame_correct : List[int]
        Binary list indicating correct (1) or incorrect (0) prediction per frame.
    per_frame_entropy : List[float]
        Shannon entropy of the predicted distribution for each frame.
    """

    def __init__(self) -> None:
        self.correct: int = 0
        self.total: int = 0
        self.per_frame_correct: List[int] = []
        self.per_frame_entropy: List[float] = []

    def update(self, logits: Tensor, labels: Tensor) -> None:
        """Update metrics with a new batch (batch_size=1 expected).

        Parameters
        ----------
        logits : Tensor
            Raw model output of shape (1, num_classes).
        labels : Tensor
            Ground truth class indices of shape (1,).
        """
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=-1)

            # Top-1 accuracy
            predicted = probs.argmax(dim=-1)
            is_correct = (predicted == labels).int().item()
            self.correct += int(is_correct)
            self.total += 1
            self.per_frame_correct.append(int(is_correct))

            # Shannon entropy: H = -sum(p * log(p + eps))
            eps = 1e-8
            entropy = -(probs * (probs + eps).log()).sum(dim=-1).item()
            self.per_frame_entropy.append(float(entropy))

    @property
    def top1_accuracy(self) -> float:
        """Return running Top-1 accuracy as a percentage."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total

    @property
    def mean_entropy(self) -> float:
        """Return mean entropy across all frames seen so far."""
        if not self.per_frame_entropy:
            return 0.0
        return float(np.mean(self.per_frame_entropy))

    def summary(self) -> str:
        """Return a formatted summary string."""
        return (
            f"Top-1 Accuracy: {self.top1_accuracy:.2f}%  |  "
            f"Mean Entropy: {self.mean_entropy:.4f} nats  |  "
            f"Frames: {self.total}"
        )


# ---------------------------------------------------------------------------
# 5.  Utility: Shannon entropy loss
# ---------------------------------------------------------------------------

def entropy_loss(logits: Tensor) -> Tensor:
    """Compute the mean Shannon entropy of the softmax distribution.

    H(p) = -sum_k p_k * log(p_k)

    Used by TENT as the self-supervised loss for batch-norm adaptation.

    Parameters
    ----------
    logits : Tensor
        Model output of shape (N, C) where C is number of classes.

    Returns
    -------
    Tensor
        Scalar entropy loss (mean over N samples).

    Notes
    -----
    At batch_size=1 (N=1), the entropy of a single sample provides a very
    noisy gradient estimate. This is the root cause of TENT's collapse in
    streaming video scenarios.
    """
    probs = torch.softmax(logits, dim=-1)
    # Add a small epsilon for numerical stability (avoids log(0))
    log_probs = torch.log(probs + 1e-8)
    # Entropy per sample; then mean over the batch (here batch=1)
    per_sample_entropy = -(probs * log_probs).sum(dim=-1)
    return per_sample_entropy.mean()


# ---------------------------------------------------------------------------
# 6.  Baseline 1: Source-Only
# ---------------------------------------------------------------------------

def configure_source_only(model: nn.Module) -> nn.Module:
    """Configure the model for Source-Only evaluation (no adaptation).

    The model is completely frozen: no parameter updates, no running statistics
    updates. This represents the lower bound of performance under distribution
    shift — the model relies entirely on features learned during pre-training.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model to freeze.

    Returns
    -------
    nn.Module
        Model with all parameters frozen, in eval mode.
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    print("[Source-Only] Model fully frozen. No adaptation will occur.")
    return model


def evaluate_source_only(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    log_interval: int = 100,
) -> MetricsTracker:
    """Run Source-Only evaluation over the full streaming sequence.

    Parameters
    ----------
    model : nn.Module
        Frozen pre-trained model.
    loader : DataLoader
        Streaming DataLoader (batch_size=1, shuffle=False).
    device : torch.device
        Computation device.
    log_interval : int
        Print progress every N frames.

    Returns
    -------
    MetricsTracker
        Filled metrics tracker after full stream evaluation.
    """
    model = configure_source_only(model)
    tracker = MetricsTracker()

    start_time = time.time()
    with torch.no_grad():
        for frame_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            tracker.update(logits, labels)

            if (frame_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"  [Source-Only] Frame {frame_idx + 1:04d} | "
                    f"{tracker.summary()} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    print(f"\n[Source-Only] Final: {tracker.summary()}")
    return tracker


# ---------------------------------------------------------------------------
# 7.  Baseline 2: BN-Adapt
# ---------------------------------------------------------------------------

def configure_bn_adapt(model: nn.Module) -> nn.Module:
    """Configure the model for BN-Adapt evaluation.

    BN-Adapt updates the running mean and variance of all BatchNorm layers
    from the test stream statistics. No backpropagation is performed; the
    affine parameters (gamma, beta) remain frozen at their training values.

    This is achieved by:
    1. Setting the model to training mode (so BN uses batch statistics).
    2. Freezing all parameters so gradients are not computed.
    3. Enabling momentum=None in BN layers so they compute a cumulative
       mean/variance from all test frames seen so far.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model.

    Returns
    -------
    nn.Module
        Model configured for BN-Adapt.
    """
    # Freeze all parameters (no gradient updates)
    for param in model.parameters():
        param.requires_grad_(False)

    # Set only BN layers to train mode so they update running statistics
    # from incoming test batches. All other layers remain in eval mode.
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Train mode: use batch statistics (mean/var from current batch)
            module.train()
            # momentum=None activates cumulative moving average mode in PyTorch BN:
            # running_mean = (1/n) * sum(batch_means) — each batch gets equal weight.
            # This is more stable than exponential moving average (momentum=0.1)
            # for a streaming stream where early batches should not dominate.
            module.momentum = None
            # Reset running stats so adaptation starts fresh
            module.reset_running_stats()

    print("[BN-Adapt] BatchNorm layers set to update running statistics from test stream.")
    return model


def evaluate_bn_adapt(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    log_interval: int = 100,
) -> MetricsTracker:
    """Run BN-Adapt evaluation over the full streaming sequence.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model.
    loader : DataLoader
        Streaming DataLoader (batch_size=1, shuffle=False).
    device : torch.device
        Computation device.
    log_interval : int
        Print progress every N frames.

    Returns
    -------
    MetricsTracker
        Filled metrics tracker after full stream evaluation.
    """
    model = configure_bn_adapt(model)
    tracker = MetricsTracker()

    start_time = time.time()
    # No torch.no_grad() here: BN train mode requires a forward pass with
    # gradient tracking disabled at the outermost level for efficiency,
    # but we still need BatchNorm to update its running statistics.
    with torch.no_grad():
        for frame_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass: BN layers update their running stats automatically
            logits = model(images)
            tracker.update(logits, labels)

            if (frame_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"  [BN-Adapt] Frame {frame_idx + 1:04d} | "
                    f"{tracker.summary()} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    print(f"\n[BN-Adapt] Final: {tracker.summary()}")
    return tracker


# ---------------------------------------------------------------------------
# 8.  Baseline 3: TENT (expected to collapse at batch_size=1)
# ---------------------------------------------------------------------------

def configure_tent(model: nn.Module, lr: float = 1e-3) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Configure the model for TENT (Test Entropy Minimization).

    TENT updates only the affine parameters (gamma=weight, beta=bias) of all
    BatchNorm layers by minimising Shannon entropy via gradient descent.

    **WARNING**: At batch_size=1, BN statistics degenerate (variance → 0) and
    the entropy gradient is extremely noisy. TENT is expected to collapse after
    a few hundred frames, manifesting as:
      - Rapid degradation of Top-1 accuracy toward 0%.
      - Entropy rising toward its maximum (log(num_classes)).
      - Potential NaN values in loss or parameters.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model.
    lr : float
        Learning rate for the SGD/Adam update on BN affine parameters.
        Default 1e-3 (from the original TENT paper).

    Returns
    -------
    Tuple[nn.Module, torch.optim.Optimizer]
        Configured model and optimizer.
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad_(False)

    # Set BN layers to train mode and unfreeze ONLY their affine parameters
    params_to_optimise = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()           # Use batch statistics (degenerate at N=1!)
            module.requires_grad_(True)
            # Collect affine parameters (gamma=weight, beta=bias)
            if module.weight is not None:
                module.weight.requires_grad_(True)
                params_to_optimise.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                params_to_optimise.append(module.bias)

    # Use SGD with the learning rate from the original TENT paper
    optimizer = torch.optim.SGD(params_to_optimise, lr=lr, momentum=0.9)

    print(
        f"[TENT] {len(params_to_optimise)} BN affine parameter tensors to optimise, "
        f"lr={lr}."
    )
    print("[TENT] WARNING: Expected to collapse at batch_size=1 due to BN variance=0.")
    return model, optimizer


def evaluate_tent(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    log_interval: int = 100,
) -> MetricsTracker:
    """Run TENT evaluation over the full streaming sequence.

    At each frame, TENT performs:
      1. Forward pass to get logits.
      2. Compute entropy loss: H = -sum(p * log p).
      3. Backward pass to compute gradients w.r.t. BN affine params.
      4. Optimizer step to update BN affine params.
      5. Record accuracy metric.

    **At batch_size=1, steps 2–4 are extremely noisy and may cause collapse.**

    Parameters
    ----------
    model : nn.Module
        Pre-trained model.
    loader : DataLoader
        Streaming DataLoader (batch_size=1, shuffle=False).
    device : torch.device
        Computation device.
    lr : float
        Learning rate for TENT adaptation.
    log_interval : int
        Print progress every N frames.

    Returns
    -------
    MetricsTracker
        Filled metrics tracker after full stream evaluation.
    """
    model, optimizer = configure_tent(model, lr=lr)
    tracker = MetricsTracker()
    collapse_detected = False

    start_time = time.time()
    for frame_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # --- Forward pass (with gradient tracking for TENT) ---
        optimizer.zero_grad()
        logits = model(images)

        # --- Entropy loss ---
        loss = entropy_loss(logits)

        # --- Detect NaN / collapse ---
        if torch.isnan(loss) or torch.isinf(loss):
            if not collapse_detected:
                print(
                    f"\n[TENT] *** COLLAPSE DETECTED at frame {frame_idx + 1}: "
                    f"loss={loss.item():.4f} (NaN/Inf). "
                    f"This is the expected batch-size-1 failure mode. ***\n"
                )
                collapse_detected = True
            # Skip backprop on NaN (continuing to demonstrate post-collapse behaviour)
            tracker.update(logits, labels)
            continue

        # --- Backward pass and parameter update ---
        loss.backward()
        optimizer.step()

        # --- Record metrics (after the update, to measure adapted performance) ---
        tracker.update(logits, labels)

        if (frame_idx + 1) % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f"  [TENT] Frame {frame_idx + 1:04d} | "
                f"Loss: {loss.item():.4f} | "
                f"{tracker.summary()} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    if not collapse_detected:
        print(
            "\n[TENT] No NaN collapse detected on synthetic data. "
            "On real corrupted data (Tiny-ImageNet-C), collapse is expected "
            "due to degenerate BN statistics at batch_size=1."
        )

    print(f"\n[TENT] Final: {tracker.summary()}")
    return tracker


# ---------------------------------------------------------------------------
# 9.  Results reporting
# ---------------------------------------------------------------------------

def print_comparison_table(results: dict) -> None:
    """Print a formatted comparison table of all baseline results.

    Parameters
    ----------
    results : dict
        Dictionary mapping baseline name to MetricsTracker instance.
        e.g. {"Source-Only": tracker1, "BN-Adapt": tracker2, "TENT": tracker3}
    """
    print("\n" + "=" * 65)
    print(f"{'Baseline':<20} {'Top-1 Acc (%)':>15} {'Mean Entropy':>15} {'Frames':>10}")
    print("-" * 65)
    for name, tracker in results.items():
        print(
            f"{name:<20} "
            f"{tracker.top1_accuracy:>14.2f}% "
            f"{tracker.mean_entropy:>15.4f} "
            f"{tracker.total:>10d}"
        )
    print("=" * 65)
    print(
        "\nNote: TENT accuracy degradation (if any) reflects the expected "
        "batch-size-1 collapse. This motivates the TEC method.\n"
    )


# ---------------------------------------------------------------------------
# 10.  Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Baseline TTA evaluation for VidTENT (batch_size=1 streaming).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["source_only", "bn_adapt", "tent", "all"],
        default="all",
        help=(
            "Which baseline(s) to run. "
            "'all' runs Source-Only, BN-Adapt, and TENT sequentially."
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=500,
        help="Number of frames in the synthetic video stream.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of output classes (1000 for ImageNet, 200 for Tiny-ImageNet).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Spatial resolution of input images.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for TENT optimizer.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Print progress every N frames.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function: parse arguments, load model and data, run evaluations."""
    args = parse_args()

    print("\n" + "=" * 65)
    print("  VidTENT — Baseline Evaluation (batch_size=1)")
    print("=" * 65)
    print(f"  Mode        : {args.mode}")
    print(f"  Num frames  : {args.num_frames}")
    print(f"  Num classes : {args.num_classes}")
    print(f"  Image size  : {args.image_size}×{args.image_size}")
    print(f"  TENT lr     : {args.lr}")
    print("=" * 65 + "\n")

    # --- Device ---
    device = get_device()

    # --- DataLoader ---
    loader = build_streaming_dataloader(
        num_frames=args.num_frames,
        num_classes=args.num_classes,
        image_size=args.image_size,
        corruption_shift=True,
    )

    # --- Results accumulator ---
    results = {}

    # -----------------------------------------------------------------------
    # Run Source-Only baseline
    # -----------------------------------------------------------------------
    if args.mode in ("source_only", "all"):
        print("\n--- Running Source-Only Baseline ---")
        # Load a fresh model copy for each baseline to avoid cross-contamination
        model = load_mobilenetv2(num_classes=args.num_classes, device=device)
        tracker = evaluate_source_only(
            model, loader, device, log_interval=args.log_interval
        )
        results["Source-Only"] = tracker

    # -----------------------------------------------------------------------
    # Run BN-Adapt baseline
    # -----------------------------------------------------------------------
    if args.mode in ("bn_adapt", "all"):
        print("\n--- Running BN-Adapt Baseline ---")
        model = load_mobilenetv2(num_classes=args.num_classes, device=device)
        tracker = evaluate_bn_adapt(
            model, loader, device, log_interval=args.log_interval
        )
        results["BN-Adapt"] = tracker

    # -----------------------------------------------------------------------
    # Run TENT baseline (expected to collapse)
    # -----------------------------------------------------------------------
    if args.mode in ("tent", "all"):
        print("\n--- Running TENT Baseline (WARNING: expect collapse at batch_size=1) ---")
        model = load_mobilenetv2(num_classes=args.num_classes, device=device)
        tracker = evaluate_tent(
            model, loader, device, lr=args.lr, log_interval=args.log_interval
        )
        results["TENT"] = tracker

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    if results:
        print_comparison_table(results)


if __name__ == "__main__":
    main()
