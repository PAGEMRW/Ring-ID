# Ring-ID: Lightweight and Open-Set PPG Authentication for Smart Rings

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu124-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 项目简介

本项目实现了一种基于光体积描记（Photoplethysmography, PPG）信号的**孪生神经网络身份认证系统**。

通过可穿戴设备采集的 PPG 信号，实现**非接触式、生理特征级的身份认证**。

### ✨ 核心特点

* 🔗 基于 **孪生神经网络（Siamese Network）** 的特征提取与相似度学习
* 🧠 支持 **开放集（零样本）认证**，能够识别未见过的新用户
* ⚡ **轻量化模型设计**

  * 参数量：**0.0139M**
  * CPU 推理延迟：**103.4 ms**
* 📱 适用于 **资源受限的可穿戴设备场景**

<p align="center">
  <img src="assets/framework.jpg" width="600"/>
</p>

---

## 🧪 环境配置

推荐运行环境：

* Python ≥ 3.8
* PyTorch = 2.5.1 + cu124

### 安装依赖

```bash
conda env create -f environment.yml
# 或根据实际情况手动安装依赖
```

---

## 📂 数据集

请从以下链接下载数据集：

* [BIDMC Dataset](https://physionet.org/content/bidmc/1.0.0/)
* [CapnoBase Dataset](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT)

---

## ⚙️ 参数配置

所有模型与训练参数均在：

```bash
config.yaml
```

中进行配置。

---

## 📁 数据集结构

```text
dataset_path/
├── character01/
│   ├── 01.npy
│   ├── 02.npy
│   └── ...
├── character02/
├── character03/
└── ...
```

示例数据：

```bash
datasets/hilbert/bidmc/train
```

---

## 🔄 预处理与模型选择

根据信号预处理方式不同，输入数据形状及可选模型如下：

| 预处理方式   | 输入形状 (channels, length) | 可选 backbone                                           | 推荐 flat_shape |
| ------- | ----------------------- | ----------------------------------------------------- | ------------- |
| EEMD    | (6, segment_length)     | Conv1D / GRU / LSTM / TCN                             | 64            |
| Hilbert | (3, segment_length)     | Conv1Dhilbert / GRUhilbert / LSTMhilbert / TCNhilbert | 64            |
| 原始信号    | (1, segment_length)     | Conv1Dseq / GRUseq / LSTMseq / TCNseq                 | 32            |

### 说明

* `segment_length` 默认值为 **256**（建议根据采样率调整）
* 学习率、batch size、训练轮数等参数均在 `config.yaml` 中配置

---

## 🚀 模型训练

### Step 1：准备数据

将数据集按上述结构放入 `dataset_path`，或在 `config.yaml` 中修改路径。

若使用 Hilbert 预处理：

```bash
python hilbert.py
```

---

### Step 2：修改配置

编辑 `config.yaml`：

* preprocess
* backbone
* 训练参数

---

### Step 3：开始训练

```bash
python train.py
```

---

## 📊 模型评估

训练完成后，可在测试集上评估模型性能。

### 修改评估参数

```python
data_dir = "datasets/hilbert/bidmc/test"
model_path = "path/to/your/model.pth"
output_dir = "evaluation/results"

model_name = "GRUhilbert"
flat_shape = 64
```

---

### 支持指标

* Accuracy
* Precision / Recall / F1-score
* FAR / FRR
* EER
* AUC
* TAR @ FAR = 1%

---

### 可视化结果

* ROC 曲线
* DET 曲线

---

## 📚 Reference

本项目中的孪生神经网络实现基于：

https://github.com/bubbliiiing/Siamese-pytorch.git

---

## 📄 License

本项目基于 MIT License 开源。
