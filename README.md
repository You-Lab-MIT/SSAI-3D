# SSAI-3D: System- and Sample-agnostic Isotropic 3D Microscopy

## Overview

This repository provides the implementation and resources for [SSAI-3D](https://arxiv.org/abs/2406.06337), a deep learning framework for achieving high-fidelity isotropic 3D resolution recovery in microscopy. SSAI-3D is a weakly physics-informed, domain-shift-resistant framework demonstrated across diverse microscopy system and a wide array of 3D biological samples. Key features and advantages of SSAI-3D include:

Self-Supervised Learning: Leverages the inherent information within a single image stack to generate a synthetic training dataset.

* **Versatility and Robustness**: Effective across a variety of microscopy systems (different contrast mechanisms, unknown blurring and noise characteristics, commercial/custom-built, low/high resolution, low/high SNR) and biological samples (living/fixed, human/animal, high/low lateral-axial similarity).
* **Experimentally Validated High Fidelity**: Produces state-of-the-art 3D reconstructions with high accuracy in both simulated and publicly available experimental datasets with ground truth, compared to existing methods.
* **Efficiency**: Requires only a single image stack and the shortest training time compared to existing methods.

---

## System Requirements

### Software Requirements

#### Primary Dependencies

```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
```

### Hardware Requirements

- NVIDIA RTX 3080 GPU is recommended.

Note: It is advisable to utilize a system with at least 16 GB of RAM for optimal performance.

--- 

## Installation Guide

Typical installation time on a desktop computer is estimated to be around 10 minutes.

### Setup Instructions

To install the necessary dependencies and set up the environment, execute the following commands:

```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

---
## Demonstration

Using the provided network checkpoints, typical run time on a desktop computer is estimated to be around 5 minutes.

### Data Preparation 

- The required datasets can be downloaded from this [link](https://drive.google.com/drive/folders/19KhzBk-VbITqaTUqJYe6j4UGu8oleUC_?usp=drive_link). Once downloaded, please transfer them to the `./dataset` directory.

### Model Checkpoints

- Download the model checkpoints from this [link](https://drive.google.com/drive/folders/1bMJhmWZNUGmZzrsBTQkDy8JnxjP9iKKV?usp=drive_link) and relocate them to the `./experiments` folder.

### Model Inference

For detailed instructions on conducting model inference, please consult the `./demo/demo.ipynb` file within the demo directory.
