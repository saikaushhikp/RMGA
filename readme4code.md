# UCF50 Action Recognition with ViTTA
## Requirements
```
pip install torch torchvision opencv-python scikit-learn tqdm numpy
```

## Usage

### Train + Evaluate (standard)
```bash
python ucf50_action_recognition.py --clean_dir UCF50 --mixed_dir UCF50_mixed --epochs 30 --batch_size 8 --num_frames 16 --img_size 112 --save_every 5
```

### Train + Evaluate WITH ViTTA
```bash
python ucf50_action_recognition.py --clean_dir UCF50 --mixed_dir UCF50_mixed --epochs 30 --ViTTA --vitta_clips 4 --vitta_steps 1 --vitta_lr 1e-4
```

### Eval-only from saved weights (with ViTTA)
```bash
python ucf50_action_recognition.py --clean_dir UCF50 --mixed_dir UCF50_mixed --mode eval_only --load_weights best_model.pth --ViTTA
```

## Swapping the backbone model
Change only these 3 lines at the top of the script:
```python
MODEL_NAME  = "mobilenet_v3_small"   # -> e.g. "efficientnet_b0", "resnet18"
FEATURE_DIM = 576                    # -> match the backbone's output channels
PRETRAINED  = True
```

| Backbone            | MODEL_NAME            | FEATURE_DIM |
|---------------------|-----------------------|-------------|
| MobileNetV3-Small   | mobilenet_v3_small    | 576         |
| MobileNetV3-Large   | mobilenet_v3_large    | 960         |
| EfficientNet-B0     | efficientnet_b0       | 1280        |
| ResNet-18           | resnet18              | 512         |
| ResNet-50           | resnet50              | 2048        |

## ViTTA Summary
Based on: https://arxiv.org/pdf/2211.15393

At test time, for each video:
1. Sample N temporally-diverse clips (temporal augmentation).
2. Update Batch Norm statistics from those clips (test-time BN adaptation).
3. Optionally minimise prediction entropy via a few gradient steps.
4. Average softmax scores across clips → final prediction.
