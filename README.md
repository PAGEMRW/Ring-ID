# Ring-ID: Lightweight and Open-Set PPG Authentication for Smart Rings

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu124-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview

This project implements a **PPG-based identity authentication system using a Siamese neural network**. By utilizing Photoplethysmography (PPG) signals collected from wearable devices, the model can distinguish between different users and achieve **non-contact, physiological biometric authentication**.

**Main Features:**

- Uses a **Siamese Neural Network** for feature extraction and similarity measurement, supporting end-to-end training.
- Supports **zero-shot authentication for new users**, enabling recognition of unseen users without retraining the model.

---

## Recommended Environment

The project is recommended to run on **Linux** with the following environment:

- **Python** >= 3.8
- **PyTorch** == 2.5.1+cu124

Other dependencies can be installed according to `environment.yml` (the author's full environment configuration) or based on your own system setup to avoid version conflicts.

---

## Dataset Download

Please download the original datasets from the following links and preprocess them according to the instructions below:

- [BIDMC Dataset](https://physionet.org/content/bidmc/1.0.0/) (PhysioNet)
- [CapnoBase Dataset](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT) (Borealis Data)

---

## Configuration

All model and training parameters are configured in the `config.yaml` file and can be modified directly. Key parameters are explained below.

### Dataset Structure

The dataset folder should be organized in the following format:

The structure of the `"dataset_path"` folder is as follows:
```python
- dataset_path
	- character01
		- 01.npy
		- 02.npy
		- ……
	- character02
	- character03
	- ……
```

Example data has been placed in `datasets/hilbert/bidmc/train` for reference.

### Supported Preprocessing Methods and Model Selection

Depending on the signal preprocessing method, the shape of the `.npy` files in `dataset_path` and the available `backbone` options will differ:

| Preprocessing Method | Input Shape (channels, length) | Available Backbone (must be specified in `config.yaml`) | Recommended `flat_shape` |
|----------------------|---------------------------------|----------------------------------------------------------|--------------------------|
| **EEMD**             | (6, segment_length)             | `'Conv1D'`, `'GRU'`, `'LSTM'`, `'TCN'`                   | 64                       |
| **Hilbert**          | (3, segment_length)             | `'Conv1Dhilbert'`, `'GRUhilbert'`, `'LSTMhilbert'`, `'TCNhilbert'` | 64 |
| **Raw Signal**       | (1, segment_length)             | `'Conv1Dseq'`, `'GRUseq'`, `'LSTMseq'`, `'TCNseq'`       | 32                       |

- `segment_length` can be set in the configuration file, with a default value of **256** (it is recommended to adjust it according to the sampling rate).
- Other hyperparameters such as **learning rate, batch size, and number of training epochs** should also be modified in `config.yaml`.

## Training the Model

The training procedure is as follows:

1. **Prepare the Data**  
   Place the raw dataset in the `dataset_path` folder under the project root directory following the structure described above (or modify `dataset_path` in `config.yaml` to point to the actual path).

2. **Modify the Configuration**  
   According to your preprocessing method and network architecture requirements, edit parameters such as `preprocess` and `backbone` in `config.yaml`.

3. **Start Training**  
   Run the following command in the terminal:

```bash
python train.py

## Model Evaluation

After training, the trained model can be used to evaluate identity authentication performance on the test dataset. This project provides an example script to compute multiple authentication metrics, including **Accuracy, Precision, Recall, F1-score, FAR, FRR, EER, AUC, and TAR@FAR=1%**, and automatically generates **ROC curves** and **DET curves**.

Modify the following parameters in the evaluation script:

```python
data_dir = "datasets/hilbert/bidmc/test"   # Test dataset path
model_path = "path/to/your/model.pth"      # Path to the trained model weights
output_dir = "evaluation/results"          # Directory to save evaluation results

model_name = "GRUhilbert"                  # Backbone type
flat_shape = 64
```

## Reference

The Siamese neural network component in this project is based on the implementation from Bubbliiiing's Siamese-PyTorch repository:  
https://github.com/bubbliiiing/Siamese-pytorch.git