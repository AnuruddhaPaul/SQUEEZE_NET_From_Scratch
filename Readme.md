# SqueezeNet from Scratch on MNIST

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)

This repository contains a full implementation of the **SqueezeNet v1.1** architecture built from scratch using PyTorch. The model is specifically adapted to train on the **MNIST dataset** (grayscale handwritten digits).

## 📌 Project Overview

**SqueezeNet**, introduced in the paper *"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"*, focuses on model efficiency. It achieves comparable accuracy to AlexNet while using significantly fewer parameters (approx **0.7 MB** to **1.2 MB** model size).

It achieves this using the **Fire Module** and by removing fully connected layers entirely.

### ⚙️ Adaptations for MNIST
The standard SqueezeNet is designed for ImageNet ($224 \times 224$). To make it work for MNIST, we applied the following changes:

1.  **Input Channels:** Modified the first convolution layer to accept **1 channel** (Grayscale) instead of 3 (RGB).
2.  **Image Resizing:** MNIST images ($28 \times 28$) are resized to **$96 \times 96$**.
    * *Reason:* SqueezeNet uses 3 MaxPool layers (downsampling factor of $8$) followed by Global Average Pooling. A $28 \times 28$ input would shrink too small to retain useful spatial features for the final classifier. Resizing to $96 \times 96$ allows for clean downsampling ($96 \to 47 \to 23 \to 11 \to \text{Classifier}$).
3.  **Output Classifier:** The final `Conv2d` layer is modified to output **10 channels** (corresponding to digits 0-9), replacing the original 1000 classes.

## 🏗️ Architecture Details

### 1. The Fire Module
The building block of SqueezeNet. It replaces standard $3 \times 3$ convolutions with a two-step process to reduce parameters:
1.  **Squeeze Layer:** A $1 \times 1$ convolution "squeezes" the input channels (dimensionality reduction).
2.  **Expand Layer:** A mix of $1 \times 1$ and $3 \times 3$ convolutions expand the data back out to capture features.

### 2. No Fully Connected Layers
Unlike VGG or AlexNet, SqueezeNet **does not use Linear (Dense) layers**.
* It generates a feature map with 10 channels (one for each class).
* It applies **Global Average Pooling** (AdaptiveAvgPool) to average each map into a single score.
* This removes millions of parameters typically found in the classifier head.

### Model Configuration

| Layer | Type | Output Size ($96 \times 96$ Input) |
| :--- | :--- | :--- |
| **Conv1** | Conv $3 \times 3$ | $47 \times 47$ |
| **Pool1** | MaxPool | $23 \times 23$ |
| **Fire 2-3** | Fire Modules | $23 \times 23$ |
| **Pool2** | MaxPool | $11 \times 11$ |
| **Fire 4-5** | Fire Modules | $11 \times 11$ |
| **Pool3** | MaxPool | $5 \times 5$ |
| **Fire 6-9** | Fire Modules | $5 \times 5$ |
| **Classifier** | Conv $1 \times 1$ $\to$ AvgPool | $1 \times 1$ (10 classes) |

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* PyTorch
* Torchvision

### Installation
Clone the repository:
```bash
git clone [https://github.com/your-username/squeezenet-mnist-scratch.git](https://github.com/your-username/squeezenet-mnist-scratch.git)
cd squeezenet-mnist-scratch
pip install torch torchvision
