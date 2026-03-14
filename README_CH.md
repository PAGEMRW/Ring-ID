# Ring-ID: Lightweight and Open-Set PPG Authentication for Smart Rings

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu124-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 项目简介

本项目实现了一种基于光体积描记（Photoplethysmography, PPG）信号的**孪生神经网络身份认证系统**。通过可穿戴设备采集到的 PPG 信号，模型能够区分不同用户，实现非接触式、生理特征级的身份验证。

**主要功能：**

- 使用**孪生神经网络（Siamese Network）** 进行特征提取与相似度度量，支持端到端训练。
- 支持**新用户零样本认证**，无需重新训练模型即可对未见用户进行识别。
- 提供**数据处理脚本**，支持 [BIDMC](https://physionet.org/content/bidmc/1.0.0/) 和 [CapnoBase](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT) 数据集，方便复现。

## 推荐环境

项目运行操作系统为 Linux，推荐使用以下环境配置：

- **Python** >= 3.8
- **PyTorch** == 2.5.1+cu124
- 其他依赖包请参考 `environment.yml`（作者完整环境配置）或根据实际情况安装，避免版本冲突。

## 数据集下载

请从以下链接获取原始数据，并按后续说明预处理：

- [BIDMC Dataset](https://physionet.org/content/bidmc/1.0.0/) （PhysioNet）
- [CapnoBase Dataset](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT) （Borealis Data）


## 参数配置

所有模型和训练参数均在 `config.yaml` 文件中配置，直接修改即可。关键参数说明如下：

### 数据集结构

数据集文件夹应按照以下格式组织：

"dataset_path"文件夹的格式如下：
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


示例数据已放置在 `datasets/hilbert/bidmc/train` 中，可供参考。

### 支持的预处理方式与模型选择

根据信号预处理方式，`dataset_path` 中的 `.npy` 文件形状和可选的 `backbone` 有所不同：

| 预处理方式 | 输入形状 (channels, length) | 可选 backbone (需在 config.yaml 中指定) | `flat_shape` 建议值 |
|-----------|-----------------------------|----------------------------------------|---------------------|
| **EEMD**  | (6, segment_length)         | `'Conv1D'`, `'GRU'`, `'LSTM'`, `'TCN'` | 64                  |
| **Hilbert** | (3, segment_length)       | `'Conv1Dhilbert'`, `'GRUhilbert'`, `'LSTMhilbert'`, `'TCNhilbert'` | 64   |
| **原始信号** | (1, segment_length)       | `'Conv1Dseq'`, `'GRUseq'`, `'LSTMseq'`, `'TCNseq'` | 32                  |

- `segment_length` 可在配置文件中设置，默认为 256（建议根据采样率调整）。
- 其他超参数（如学习率、批次大小、训练轮数等）也请在 `config.yaml` 中修改。

## 训练模型

训练流程如下：

1. **准备数据**  
   将原始数据集按上述结构放置在项目根目录下的 `dataset_path` 文件夹中（或修改 `config.yaml` 中的 `dataset_path` 指向实际路径）。

2. **修改配置**  
   根据你的预处理方式和网络需求，编辑 `config.yaml` 中的 `preprocess`、`backbone` 等参数。

3. **启动训练**  
   在终端执行以下命令：
   ```bash
   python train.py

## 模型评估

训练完成后，可以使用训练好的模型对测试数据进行身份认证评估。本项目提供了示例脚本用于计算多种身份认证指标，包括 **Accuracy、Precision、Recall、F1-score、FAR、FRR、EER、AUC 以及 TAR@FAR=1%**，并自动绘制 **ROC 曲线** 和 **DET 曲线**。

在评估脚本中修改以下参数：

```python
data_dir = "datasets/hilbert/bidmc/test"   # 测试数据路径
model_path = "path/to/your/model.pth"      # 训练好的模型权重
output_dir = "evaluation/results"          # 评估结果保存路径

model_name = "GRUhilbert"                  # backbone 类型
flat_shape = 64
```
## Reference
本项目中的孪生神经网络部分基于 Bubbliiiing 的 Siamese-pytorch 实现（https://github.com/bubbliiiing/Siamese-pytorch.git）