![EG-DINO Architecture Diagram](docs/architecture.png)

# EG-DINO: Edge-Guided Foundation Models for Continual Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/eg-dino/semantic-segmentation-on-crack500)](https://paperswithcode.com/sota)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

> Official PyTorch implementation for our CVPR paper on Edge-Guided Continual Learning for Segmentation.

EG-DINO introduces a structural prior using deep learned boundaries (HED) to bridge the stability-plasticity gap in Continual Learning (CL) for dense prediction tasks. By dynamically gating features between a frozen and fine-tuned DINOv2 backbone, our method mitigates catastrophic forgetting while maximizing target domain transfer.

## 🏗️ Architecture

![EG-DINO Architecture](docs/architecture.png)
*Figure 1: The EG-DINO pipeline. The unfrozen backbone (green) adapts to new domains while the frozen backbone (blue) retains source knowledge. These streams are regulated by a compact gating module (small circle) guided by HED boundary priors, directly feeding into the parallel edge pyramid decoder stages (orange).*

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/YOUR_USERNAME/EG-DINO.git](https://github.com/YOUR_USERNAME/EG-DINO.git)
cd EG-DINO
pip install -r requirements.txt

2. Data Preparation

Structure your datasets (e.g., Crack500, DeepCrack) as follows. The HED edges will be automatically computed and cached during your first training or testing run.

dataset_crack500/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/

💻 Usage

Our scripts utilize command-line arguments for seamless adaptation to different datasets and image resolutions.
Training

To train EG-DINO from scratch or fine-tune it on a target domain:
python train.py \
  --data_root ./dataset_crack500 \
  --img_height 630 \
  --img_width 350 \
  --batch_size 8 \
  --epochs 100 \
  --lr_encoder 5e-6 \
  --lr_decoder 5e-4
Note: The best model weights will be automatically saved to the ./weights directory.

Evaluation (Zero-Shot & Transfer)

To evaluate your trained model and compute both Fixed (Argmax) and Optimal Dataset Scale (ODS) metrics:
python test.py \
  --data_root ./dataset_crack500 \
  --test_split test \
  --img_height 630 \
  --img_width 350 \
  --weights ./weights/eg_dino_630x350.pth

  📊 Main Results
Ablation Study on Architecture Design

Evaluation of Continual Learning Protocol (Dataset C → Dataset D). All results are reported using the F1 Score (%).

| Method | Oracle Source (C) | Zero-Shot (C &rarr; D) | Transfer (C &rarr; D) | Forgetting Check (D &rarr; C) |
| :--- | :---: | :---: | :---: | :---: |
| DINOv2 (Frozen) + FCN | 73.8 | 76.2 | 78.6 | 70.6 |
| DINOv2 (Finetuned) + FCN | 74.8 | 80.2 | 84.8 | 69.2 |
| DINOv2 (LoRA) + FCN | 74.3 | 79.4 | 84.4 | 67.2 |
| DINOv3 (Finetuned) + FCN | 74.5 | 76.5 | 82.3 | 70.0 |
| SAM 3 | 73.8 | 75.4 | **87.2** | 71.0 |
| EG-DINO - Canny | 74.4 | 81.8 | 85.0 | 69.8 |
| **EG-DINO - HED (Ours)** | **75.2** | **83.0** | 84.8 | **72.7** |

🔗 Citation

If you find this code or our conceptual pipeline useful in your research, please consider citing:
@inproceedings{eg_dino_cvpr202X,
  title={YOUR PAPER TITLE HERE},
  author={YOUR NAME HERE},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={202X}
}