# SSAI3D README Documentation

## Overview

SSAI3D is an advanced Python module leveraging deep learning methodologies to enhance the axial resolution of deteriorated images in fluorescence microscopy through state-of-the-art super-resolution techniques, utilizing a singular image stack. A comprehensive demo is provided, featuring a Jupyter notebook equipped with several pairs of pre-trained models and corresponding datasets.

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

### Setup Instructions

To install the necessary dependencies and set up the environment, execute the following commands:

```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

---
## Demonstration

### Data Preparation 

- The required datasets can be downloaded from this [link](https://drive.google.com/drive/folders/19KhzBk-VbITqaTUqJYe6j4UGu8oleUC_?usp=drive_link). Once downloaded, please transfer them to the `./data` directory.

### Model Checkpoints

- Download the model checkpoints from this [link](https://drive.google.com/drive/folders/1bMJhmWZNUGmZzrsBTQkDy8JnxjP9iKKV?usp=drive_link) and relocate them to the `./experiments` folder.

### Model Inference

For detailed instructions on conducting model inference, please consult the `./demo/demo.ipynb` file within the demo directory.