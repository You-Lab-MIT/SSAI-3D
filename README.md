# SSAI-3D: System- and Sample-agnostic Isotropic 3D Microscopy

## Overview

This repository provides the implementation and resources for [SSAI-3D](https://arxiv.org/abs/2406.06337), a deep learning framework for achieving high-fidelity isotropic 3D resolution recovery in microscopy. SSAI-3D is a weakly physics-informed, domain-shift-resistant framework demonstrated across diverse microscopy system and a wide array of 3D biological samples.

<img width="1012" alt="image" src="https://github.com/user-attachments/assets/8beebb61-3ea2-49dc-b273-50f17e796c37">

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
## Demonstration -- Testing

Using the provided network checkpoints, typical run time on a desktop computer is estimated to be around 5 minutes.

### Data Preparation 

- The required datasets can be downloaded from this [link](https://drive.google.com/drive/folders/19KhzBk-VbITqaTUqJYe6j4UGu8oleUC_?usp=drive_link). Once downloaded, please unzip and transfer them to the `./dataset` directory.

### Model Checkpoints

- Download the model checkpoints from this [link](https://drive.google.com/drive/folders/1bMJhmWZNUGmZzrsBTQkDy8JnxjP9iKKV?usp=drive_link) and unzip and transfer them to the `./experiments` folder.

### Model Inference

For detailed instructions on conducting model inference, please consult the `./demo/demo_test.ipynb` file within the demo directory.


---
## Demonstration -- Training

Using the pre-trained NAFNet weights, typical fine-tuning time on a desktop computer is estimated to be around 5~30 minutes, depending on the dataset.

### Data Preparation 

- The required anisotropic raw dataset can be downloaded from [here](https://drive.google.com/file/d/1p3CUWhaSJXAA_9k8p4nRrhjBmbegQ-vJ/view?usp=sharing). Once downloaded, please transfer it to `~/SSAI-3D/demo` directory.

### Pre-trained Model Checkpoints

- Download the pre-trained model checkpoint from [here](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view), and transfer it to `~/SSAI-3D/demo/experiments/pretrained_models`.

- [OPTIONAL] If denoising is needed, download model checkpoint from [here](https://drive.google.com/file/d/1Lkg5a8xtjze7cKitdMl8bIY38cLAIojT/view?usp=sharing), and denoise before restoration step. Please note that the denoising model is trained on private data.

### Model Training

For detailed instructions on conducting model inference, please consult the `./demo/demo_train.ipynb` file within the demo directory.
