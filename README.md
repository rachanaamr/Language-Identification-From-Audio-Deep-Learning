# Spoken Language Identification with DANN
This folder contains the codebase for classifying 22 Indian languages. The project uses a pre-trained mHuBERT-147 model and introduces a Domain-Adversarial Neural Network architecture to remove speaker bias, forcing it to learn speaker-invariant features rather than memorizing individual voices.

# Code files
Included are three .ipynb files along with their respective requirements:

Baseline Model

Improved HuBert

DANN Architecture

# How to Run on Kaggle (GPU P100)
This is optimized for Tesla 16GB hardware. The training loop utilizes a physical batch size of 16 combined with 4 gradient accumulation steps for an effective batch size without triggering CUDA Out-Of-Memory (OOM) errors.

Step 1: Environment Setup

Import as a new notebook.

In the menu, under settings, select Accelerator. Manage scripts for Hugging Face and Weights & Biases (W&B) authentication. Paste your API key.

Execute the pip cells sequentially. The first handles all dependency installations, downloads the dataset, and ensures compatibility:
!pip install evaluate transformers==3.6.0 --upgrade wandb
