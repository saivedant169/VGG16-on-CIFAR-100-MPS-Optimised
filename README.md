
---

# VGG16 on CIFAR-100 (MPS Optimized)

This repository contains a full implementation of the VGG16 network architecture built from scratch using PyTorch. The project trains and evaluates the model on the CIFAR-100 dataset and includes an interactive demo for visualizing predictions. The code is optimized for Apple’s MPS chipset, ensuring smooth performance on Apple Silicon (e.g., M1, M2).

## Overview

- **VGG16 Architecture**: Manual implementation with 13 convolutional layers, max pooling, and fully connected layers.
- **Dataset**: CIFAR-100, with training and testing scripts.
- **Interactive Demo**: A feature to input an image index and view the corresponding image, ground truth, and predicted class.
- **MPS Support**: Automatically detects and utilizes the MPS device when available, with fallback to CUDA or CPU.

> This project was developed following the guidelines provided in the [Computer Vision Project 2 assignment](citeturn0file0).

## Features

- **Manual VGG16 Implementation** using PyTorch’s autograd.
- **Training & Evaluation** scripts with comprehensive logging of accuracy and loss.
- **Interactive Mode**: Easily visualize predictions for any image in the dataset.
- **Optimized for MPS**: Seamless support for Apple’s MPS chipset, leveraging native hardware acceleration.

## Requirements

- **Python 3.8+**
- **PyTorch 1.12+** (with MPS support for Apple Silicon)
- Additional libraries:
  - torchvision
  - matplotlib
  - numpy

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/vgg16-cifar100-mps.git
   cd vgg16-cifar100-mps
   ```

2. **Set up a virtual environment (recommended):**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install torch torchvision matplotlib numpy
   ```
   
   > *Note: For MPS support, please verify that your PyTorch installation is configured for Apple Silicon. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for details.*

## Code Structure

- **vgg16.py**: Contains the VGG16 network implementation.
- **train.py**: Script to train the model on the CIFAR-100 dataset.
- **evaluate.py**: Script to evaluate model performance on the test set.
- **interactive_demo.py**: Provides an interactive command-line interface to display images and predictions.
- **README.md**: This file.

## Usage

### 1. Training

To train the model, run the training script with your desired hyperparameters:

```bash
python train.py --epochs 50 --batch-size 128 --learning-rate 0.001
```

This script will:
- Load and preprocess the CIFAR-100 dataset.
- Initialize the VGG16 model.
- Detect the available device (MPS, CUDA, or CPU) automatically.
- Save the trained model to the `./models` directory.

### 2. Evaluation

After training, evaluate the model on the test dataset:

```bash
python evaluate.py --model-path ./models/vgg16_cifar100.pth
```

### 3. Interactive Demo

To visualize predictions for a specific image index:

```bash
python interactive_demo.py
```

Follow the prompt to enter an image index; the script will display:
- The corresponding image using matplotlib.
- The ground truth label.
- The predicted label from the trained network.

## MPS Device Support

The code is designed to automatically use Apple’s MPS chipset if available. The following snippet in each script ensures optimal device selection:

```python
import torch

device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
)
```

This guarantees that on Apple Silicon, the computations will be accelerated via the MPS backend, with appropriate fallbacks for CUDA or CPU environments.

## Results & Analysis

The training logs and evaluation metrics (loss curves, accuracy, etc.) are output during execution. You can also save these results for further analysis. For a sample of prediction results and performance evaluation, refer to the `results` folder (if available).

## Contributing

Contributions, issues, and feature requests are welcome! Please fork the repository and submit a pull request with your improvements.

## Acknowledgments

- **PyTorch** for providing a powerful deep learning framework.
- **CIFAR-100** for the challenging dataset.
- **Apple Silicon** for the performance boost via MPS.

---

This README provides clear guidance on setting up, running, and understanding the project, ensuring that anyone can build and run the code optimized for MPS-based chipsets.
