# Spoken Language Identification with DANN

This folder contains the codebase for classifying 22 Indian languages. The project uses the pre-trained mHuBERT-147 model and introduces a Domain-Adversarial Neural Network architecture to remove speaker bias, forcing the network to learn speaker-invariant features rather than memorizing individual voices.

# Code files
The project contained three .ipynb files along with their respective requirement files:

Baseline Model - train_model_facebook_baseline.ipynb 
Improved Baseline - train_model_HuBert_improved.ipynb 
DANN Architecture - train_model_Dann.ipynb

# How to Run on Kaggle (GPU P100)
This codebase is optimized specifically for Kaggle's Tesla P100-PCIE-16GB hardware. The training loop utilizes a physical batch size of 16 combined with 4 gradient accumulation steps to simulate an effective batch size of 64 without triggering CUDA Out-Of-Memory (OOM) errors.

# Step 1: Environment Setup
1.⁠ ⁠Import the code in Kaggle as a new notebook.
2.⁠ ⁠In the menu, under settings, set the Accelerator to GPU P100.

# Step 2: Manage Tokens
The script requires Hugging Face and Weights & Biases (W&B) authentication. 
Paste your Hugging Face and Weights & Biases API key tokens respectively.

# Step 3: Execute the Pipeline
Run the cells sequentially. The first cell handles all dependency installations and downgrades datasets to ensure compatibility:
!pip install evaluate transformers accelerate
!pip install datasets==3.6.0
!pip install --upgrade wandb
