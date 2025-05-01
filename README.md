<!-- # Vision Transformer for Urban Sound Classification

This project applies a pre-trained Vision Transformer (ViT) model, originally designed for image tasks, to classify environmental sounds from the UrbanSound8K dataset. By treating audio Mel-spectrograms as images, the project demonstrates cross-domain transfer learning and fine-tunes the ViT model using PyTorch. It includes experiments with fine-tuning strategies, data augmentation, hyperparameter tuning, and optimization for HPC environments.

**Current Best Result:** ~68.6% Test Accuracy on UrbanSound8K Fold 10.

## Project Overview

The objective is to adapt a vision model (ViT) for audio classification and achieve strong performance through careful experimentation. Challenges include optimizing data pipelines, managing overfitting, and tuning hyperparameters in an HPC environment using Slurm and NVIDIA A100 GPUs.

## Features & Techniques

- **Model:** Fine-tuned Vision Transformer (ViT) from Hugging Face `transformers`.
- **Framework:** PyTorch.
- **Data:** UrbanSound8K dataset via Hugging Face `datasets`.
- **Preprocessing:** Mel-spectrogram generation using `torchaudio` (Torch Audio).
- **Fine-Tuning:** Unfreezing the final N encoder layers (best with N=2) and differential learning rates.
- **Data Augmentation:** SpecAugment (frequency/time masking).
- **Optimization:** AdamW, Cosine Annealing, Weight Decay (`3e-2`), Label Smoothing, AMP.
- **HPC Usage:** SLURM-managed cluster with NVIDIA A100 GPUs.
- **Experiment Tracking:** TensorBoard logging and robust checkpoint management.

## Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face `datasets` & `transformers`
- `torchaudio`
- TensorBoard
- SLURM (HPC)

## Dataset

The **UrbanSound8K** dataset contains 8732 labeled sound excerpts (<4 seconds) across 10 urban sound classes. The project uses the standard split:
- **Training:** Folds 1-8
- **Validation:** Fold 9
- **Testing:** Fold 10

For HPC environments, pre-download the dataset using a script like `download_dataset.py` on a login node and set the cache directory in your SLURM script.

## File Structure

```
RootProject/
├── checkpoints/               # Saved model checkpoints
├── scripts/
│   ├── data.py                # Dataset loading and processing
│   ├── model.py               # ViT model setup
│   └── engine.py              # Training/evaluation loop
├── runs/                      # TensorBoard logs
├── download_dataset.py        # Pre-download script for HPC
├── train.py                   # Main training script
├── requirements.txt           # Dependencies
├── slurm.sh                   # SLURM submission script
└── README.md                  # This file
```

## Usage

1. **Clone the Repository:**
    ```bash
    git clone [Link to Your Repo]
    cd audio-classification-using-ViT
    ```

2. **Set Up Environment:**
    ```bash
    python -m venv venv  # Create a virtual environment
    source venv/bin/activate  # Activate the virtual environment (Linux/macOS)
    pip install -r requirements.txt
    ```

3. **Run Training:**
    - **Locally:**
        ```bash
        python train.py --num_epochs 30 --batch_size 32 --num_workers 16
        ```
    - **On SLURM:**
        ```bash
        sbatch slurm.sh  # Submit the job to SLURM
        ```

4. **Monitor Training:**
    ```bash
    tensorboard --logdir=runs  # Start TensorBoard
    ```

## Results Summary

The best configuration achieved **~68.6% accuracy** on the test set (Fold 10) using:
- Fine-tuning the last 2 encoder layers.
- Differential learning rates (Head: `3e-4`, Encoder: `3e-5`).
- SpecAugment and Weight Decay (`3e-2`).
- Batch Size 64 with AMP.

## Challenges & Learnings

- **Data Loading:** Optimized `num_workers` for HPC.
- **Overfitting:** Addressed with Weight Decay and Label Smoothing.
- **Checkpointing:** Ensured reliable saving/loading during experiments.

## Future Work

- Build an interactive demo with Gradio or Streamlit.
- Experiment with unfreezing additional layers.
- Explore audio-specific pre-trained models (e.g., AST, PANNs).
- Conduct deeper error analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Acknowledgements

I would like to thank the creators of the UrbanSound8K dataset for providing an invaluable resource for audio classification research. Additionally, I extend our gratitude to the Mississippi State University's High Performance Computing Collaboratory (HPCC) for granting access to the Ptolemy Supercomputer, which was instrumental in conducting the experiments. -->

# 🎵 Vision Transformer for Urban Sound Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) This project explores the **cross-domain application** of a pre-trained **Vision Transformer (ViT)** for classifying environmental sounds from the **UrbanSound8K** dataset. By treating audio Mel-spectrograms as images, the model is fine-tuned using PyTorch, incorporating various optimization techniques and leveraging High-Performance Computing (HPC) resources. The primary goal was to adapt a state-of-the-art vision architecture for audio tasks and achieve robust performance through systematic experimentation.

**Current Best Result:** **~68.6% Test Accuracy** on UrbanSound8K Fold 10.

---

## 🚀 Key Features & Techniques

* **Model:** Fine-tuning of a pre-trained **Vision Transformer (ViT)** from Hugging Face `transformers`.
* **Framework:** **PyTorch**.
* **Data Processing:** Audio loading, resampling, padding/cropping, and **Mel-spectrogram** generation via `torchaudio`.
* **Fine-Tuning:** Optimized strategy using **2 unfrozen encoder layers** with **differential learning rates**.
* **Data Augmentation:** **SpecAugment** (frequency and time masking). *Time-domain augmentations (via `torch-audiomentations`) were explored but omitted from the final best model.*
* **Optimization:** **AdamW** optimizer, **Cosine Annealing** LR Scheduler, **Weight Decay tuning** (optimal `3e-2`), **Label Smoothing**, **Automatic Mixed Precision (AMP)**.
* **HPC Utilization:** Trained and managed experiments on a **Slurm** cluster using **NVIDIA A100 (`1g.10gb` MIG)** GPUs.
* **Data Loading:** Optimized parallel data loading using map-style datasets (`streaming=False`) and tuned `DataLoader` **`num_workers`** based on allocated CPUs (32 workers optimal).
* **Experimentation:** Addressed **overfitting** and **data loading bottlenecks** through systematic testing; implemented reliable **checkpointing** (unique filenames per run) and **TensorBoard** logging.

---

## ⚙️ Tech Stack

* Python (3.10+)
* PyTorch
* Hugging Face `datasets`
* Hugging Face `transformers`
* `torchaudio`
* `torch-audiomentations` (explored)
* TensorBoard
* `numpy`
* `tqdm`
* SLURM (for HPC execution)
* Git

---

## 📊 Dataset

This project uses the **UrbanSound8K** dataset, containing 8732 labeled sound excerpts (<4 seconds) across 10 classes: `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music`.

The standard 10-fold split is used:
* **Training:** Folds 1-8
* **Validation:** Fold 9
* **Testing:** Fold 10

*(Note for HPC users: Data should be pre-downloaded to a shared cache via `download_dataset.py` or similar. Ensure `$HF_HOME` or `$HF_DATASETS_CACHE` points to this cache in your Slurm script.)*

---

## 📂 File Structure

```
RootProject/
├── scripts/
│   ├── data.py                # Dataset loading and processing
│   ├── model.py               # ViT model setup
│   └── engine.py              # Training/evaluation loop
├── runs/                      # TensorBoard logs
├── download_dataset.py        # Pre-download script for HPC
├── train.py                   # Main training script
├── requirements.txt           # Dependencies
├── slurm.sh                   # SLURM submission script
└── README.md                  # This file
```

## 🛠️ Setup & Installation

1.  **Clone Repository:**
    ```bash
    git clone [Link to Your Repo]
    cd audio-classification-using-ViT
    ```
2.  **Create & Activate Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # .\venv\Scripts\activate # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *(Requires Python 3.10+. A CUDA-enabled PyTorch installation is recommended for GPU acceleration.)*

---

## ▶️ Usage

1.  **(HPC First Time):** Ensure the UrbanSound8K dataset is cached in a shared location accessible by compute nodes.
2.  **Configure `slurm.sh`:** Adjust Slurm directives (`--cpus-per-task`, `--mem-per-cpu`, `--gres`, `--time`), paths, and module loads for your cluster.
3.  **Run Training:**
    * **Locally (Example):**
        ```bash
        python train.py --num_epochs 30 --batch_size 32 --num_workers 16
        ```
    * **On Slurm (Example using optimal settings):**
        * Ensure `slurm.sh` requests e.g., `--cpus-per-task=32` and sufficient memory.
        * Ensure the command inside `slurm.sh` uses `--num_workers 32` and `--batch_size 64`.
        ```bash
        sbatch slurm.sh
        ```
4.  **Monitor with TensorBoard:**
    ```bash
    tensorboard --logdir=runs
    ```
    *(Access via the provided URL, potentially requires SSH tunneling on HPC).*

---

## 📈 Results Summary

The best configuration achieved **~68.6% accuracy** on the UrbanSound8K test set (Fold 10). Key settings included:
* Fine-tuning the **last 2 encoder layers** + head of ViT.
* Differential LRs (`head=3e-4`, `encoder=3e-5`) with **Cosine Annealing**.
* **SpecAugment** + **Mild Time-Domain Augmentation**.
* **Weight Decay = `3e-2`**.
* **Batch Size 64** with AMP.
* Optimized data loading (`streaming=False`, **32 workers** with 32 allocated CPUs).

---

## 💡 Challenges & Learnings

* **Data Loading:** Overcame significant HPC data loading bottlenecks by switching to map-style datasets (`streaming=False`) and carefully tuning `num_workers` based on allocated Slurm CPUs (found 32 optimal for speed/accuracy). Identified potential I/O contention issues.
* **Overfitting:** Managed model overfitting through systematic experimentation with Weight Decay values, Label Smoothing, and Data Augmentation strategies.
* **Experiment Management:** Implemented reliable checkpointing with unique run identifiers (Slurm Job ID / Timestamps) for accurate model saving and evaluation across multiple experiments.

---

## 🔮 Future Work

* Develop an interactive demo using **Gradio** or **Streamlit** (deployable on Hugging Face Spaces).
* Experiment with fine-tuning **3 encoder layers** using the optimized regularization settings.
* Explore alternative **audio-specific pre-trained models** (e.g., AST, PANNs) for comparison.
* Conduct **detailed error analysis** (e.g., confusion matrix) on the best model's predictions.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

* UrbanSound8K dataset creators.
* **Mississippi State University's High Performance Computing Collaboratory (HPCC)** for access to the Ptolemy Supercomputer.
