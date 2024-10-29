# SSAI-3D: System- and Sample-agnostic Isotropic 3D Microscopy

<!-- <img width="1012" alt="image" src="./resource/logo.jpg"> -->

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
## Demonstration 

This demonstration is organized into three sections:

<!-- > **Fine-tuning Example**  
```./demo/demo_train.ipynb```: 
- This section is for the sparse fine-tuning of the pre-trained model on the individual 3D volume stack for axial resolution restoration. 


> **Testing Example**  
```./demo/demo_test.ipynb```
- This notebook is for the inference of the specific model on its corresponding fine-tuned volume stack. Please note that the model and dataset here are coupled, e.g. there is one specific pre-fine-tuned model for one dataset. 

> **Generalizable Model Testing Example**  
```./demo/demo_generalizable.ipynb```: 
- This section is for using a generalizable model, optimal performance may be achieved by following the fine-tuning and testing procedures outlined in Sections 1 and 2. -->

> **1. Fine-tuning Example**  
> ```./demo/demo_train.ipynb```
> - This section provides sparse fine-tuning of the pre-trained model on individual 3D volume stacks for axial resolution restoration. Please see below for detailed instructions on how to run the notebook.

> **2. Testing Example**  
> `./demo/demo_test.ipynb`  
> - This notebook is intended for inference with a specific pre-fine-tuned model on its corresponding volume stack. Note that each model is paired with a particular dataset; one unique pre-fine-tuned model applies to each dataset. Please see below for detailed instructions on how to run the notebook.

> **2. Generalizable Model Testing Example**  
> `./demo/demo_generalizable.ipynb`  
> - This section demonstrates the use of a generalizable model. For optimal performance, it is recommended to first follow the fine-tuning and testing procedures in  **Fine-tuning Example** and **Testing Example**. Please see below for detailed instructions on how to run the notebook.

<!-- The first two sections guide you through fine-tuning and testing of the SSAI-3D model on individual 3D stack data for axial resolution restoration, which is the intended use of SSAI-3D. While the third section provides instructions for using a generalizable model, optimal performance may be achieved by following the fine-tuning and testing procedures outlined in Sections 1 and 2. -->

<!-- Which are respectively the fine-tuning and the testing of the SSAI-3D method on single 3D stack data for axial deblurring. We have additionally included a generalizable model, however for optimal performance, one should refer to the first and second sections. -->

---

### 1. Fine-tuning Example

SSAI-3D supports sparse fine-tuning and inference on 3D microscopy data to achieve high-quality axial resolution restoration on a specific dataset.

#### Data Preparation

- Download the required anisotropic raw dataset of mouse brain neurons from [this link](https://drive.google.com/file/d/1p3CUWhaSJXAA_9k8p4nRrhjBmbegQ-vJ/view?usp=sharing). Move the dataset to the `~/SSAI-3D/demo` directory.
- **[Optional]** Download the anisotropic raw dataset of liver obtained from [CARE](https://www.nature.com/articles/s41592-018-0216-7#data-availability).

#### Pre-trained Model Checkpoints

- Download the pre-trained model checkpoint from [this link](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view). Place the file in ```~/SSAI-3D/demo/experiments/pretrained_models```
  
- **[Optional]** For applications requiring denoising, download an additional checkpoint [here](https://drive.google.com/file/d/1Lkg5a8xtjze7cKitdMl8bIY38cLAIojT/view?usp=sharing) and save it in the same directory. *Note*: This model is trained on private data.

#### Model Fine-tuning Process

To fine-tune the model on your dataset, refer to the following notebook:
```
./demo/demo_train.ipynb
```

#### Results

This demonstration includes training examples on two datasets:

- **Mouse Brain Neurons**: Available for download [here](https://zenodo.org/records/7882519).
- **Liver Data**: Obtained from the [CARE project](https://www.nature.com/articles/s41592-018-0216-7#data-availability).

Both datasets consist of 16-bit unsigned integer 3D volumes. Running `demo_train.ipynb` on either dataset will yield resolution-enhanced results, similar to the sample outputs shown below.

![Sample Output Image 1](./resource/output.png)
![Sample Output Image 2](./resource/output_.png)

#### Estimated Runtime

Typical runtime for resolution restoration on a standard desktop computer is approximately **30 minutes**. For a faster demonstration, please proceed to the next section: **2. Testing Example**.

---

### 2. Testing Example

This section enables rapid testing of SSAI-3D on a dataset with a fine-tuned model, bypassing the fine-tuning stage. The provided notebook uses the pre-supplied dataset and its corresponding fine-tuned model to evaluate deconvolution results.

Since SSAI-3D is optimized for fine-tuning on specific 3D volumes, users seeking a more generalizable model should refer to the next section: **3. Generalizable Model Testing Example**.


#### Data Preparation

- Download the required anisotropic raw dataset from [this link](https://drive.google.com/file/d/1p3CUWhaSJXAA_9k8p4nRrhjBmbegQ-vJ/view?usp=sharing) and place it in `~/SSAI-3D/demo`.

#### Fine-tuned Model Checkpoints

- Download the fine-tuned model [here](https://drive.google.com/file/d/1Q3d7y96dQsd3Xk4l8c05M2VnUlu9q9CL/view?usp=sharing) and save it in ```~/SSAI-3D/demo/experiments/demo_neurons```

#### Model Inference

For detailed instructions on running model inference, see:
```
./demo/demo_test.ipynb
```

---


### 3. Generalizable Model Testing Example

For testing on custom data, a generalizable demo is also provided. However, since SSAI-3D is optimized for fine-tuning on specific 3D volumes, we recommend referring to  **1. Fine-tuning Example** for optimal performance.




#### Data Preparation

- Prepare your own 3D volumetric data in the shape `(depth, width, height)`.

#### Fine-tuned Model Checkpoints

- Download the generalizable pre-trained, fine-tuned model from [this link](https://drive.google.com/file/d/1mhpZ00h3UvXvTfsA_feYd1B2sWD-06uY/view). Place the model in ```./experiments/pretrained_models```

#### Model Inference

For detailed instructions on running model inference, see:
```
./demo/demo_generalizable.ipynb
```
---
This guide will help you maximize the performance of SSAI-3D for fine-tuning, testing, and application to diverse fluorescence microscopy datasets. For optimal results with new data, consider performing fine-tuning as described in **1. Fine-tuning Example** above.
