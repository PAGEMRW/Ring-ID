import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

from scipy.interpolate import interp1d
from scipy.optimize import brentq

from nets.siamese import Siamese

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

############################################
# 参数
############################################

THRESHOLD = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
method = 'method'
flat_shape = 64

############################################
# 生成图像对
############################################

def load_image_pairs(data_dir,
                     max_pos_per_id=200,
                     max_neg_per_pair=200,
                     seed=42):

    random.seed(seed)
    np.random.seed(seed)

    image_pairs = []
    labels = []

    folders = [
        os.path.join(data_dir, folder)
        for folder in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, folder))
    ]

    print(f"Found {len(folders)} folders.")

    ########################################
    # 正样本（每个个体限制数量）
    ########################################

    pos_pairs = []

    for folder in folders:
        images = [
            os.path.join(folder, img)
            for img in os.listdir(folder)
            if img.endswith('npy')
        ]

        all_combinations = []

        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                all_combinations.append((images[i], images[j]))

        # 每个ID最多取 max_pos_per_id 个
        if len(all_combinations) > max_pos_per_id:
            all_combinations = random.sample(all_combinations, max_pos_per_id)

        for pair in all_combinations:
            pos_pairs.append(pair)

    ########################################
    # 负样本（每两个ID限制数量）
    ########################################

    neg_pairs = []

    for i in range(len(folders)):
        for j in range(i + 1, len(folders)):

            imgs1 = [
                os.path.join(folders[i], img)
                for img in os.listdir(folders[i]) if img.endswith('npy')
            ]
            imgs2 = [
                os.path.join(folders[j], img)
                for img in os.listdir(folders[j]) if img.endswith('npy')
            ]

            all_cross = []

            for img1 in imgs1:
                for img2 in imgs2:
                    all_cross.append((img1, img2))

            # 每对ID最多取 max_neg_per_pair
            if len(all_cross) > max_neg_per_pair:
                all_cross = random.sample(all_cross, max_neg_per_pair)

            for pair in all_cross:
                neg_pairs.append(pair)

    ########################################
    # 类别均衡
    ########################################

    min_samples = min(len(pos_pairs), len(neg_pairs))

    pos_pairs = random.sample(pos_pairs, min_samples)
    neg_pairs = random.sample(neg_pairs, min_samples)

    image_pairs = pos_pairs + neg_pairs
    labels = [1] * min_samples + [0] * min_samples

    combined = list(zip(image_pairs, labels))
    random.shuffle(combined)

    image_pairs, labels = zip(*combined)

    return list(image_pairs), np.array(labels)

############################################
# 评估
############################################

def evaluate_model(image_pairs, labels, output_dir):

    scores = []
    predictions = []

    with torch.no_grad():
        for idx, (img1_path, img2_path) in enumerate(image_pairs):

            if idx % 500 == 0:
                print(f"Processing {idx}/{len(image_pairs)}")

            img1 = np.load(img1_path)
            img2 = np.load(img2_path)

            if img1.ndim == 1:
                img1 = np.expand_dims(img1, 0)
            if img2.ndim == 1:
                img2 = np.expand_dims(img2, 0)

            if img1.ndim == 2:
                img1 = np.expand_dims(img1, 0)
            if img2.ndim == 2:
                img2 = np.expand_dims(img2, 0)

            img1 = torch.from_numpy(img1).float().to(device)
            img2 = torch.from_numpy(img2).float().to(device)

            prob = torch.sigmoid(model((img1, img2))).item()

            scores.append(prob)
            predictions.append(1 if prob >= THRESHOLD else 0)

    scores = np.array(scores)

    ################################
    # 基础指标
    ################################

    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    conf = confusion_matrix(labels, predictions)

    ################################
    # ROC
    ################################

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    roc_auc = auc(fpr, tpr)

    ################################
    # EER
    ################################

    eer = brentq(lambda x: 1. - interp1d(fpr, tpr)(x) - x, 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)

    ################################
    # TAR @ FAR = 1%
    ################################

    target_far = 0.01
    if np.max(fpr) >= target_far:
        tar_at_far = interp1d(fpr, tpr)(target_far)
    else:
        tar_at_far = 0.0

    ################################
    # FAR / FRR (基于固定阈值)
    ################################

    FP = conf[0,1]
    TN = conf[0,0]
    FN = conf[1,0]
    TP = conf[1,1]

    FAR = FP / (FP + TN)
    FRR = FN / (FN + TP)

    ################################
    # 保存指标到文件
    ################################

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("\n===== Confusion Matrix =====\n")
        f.write(f"TN: {TN}  FP: {FP}\n")
        f.write(f"FN: {FN}  TP: {TP}\n")
        f.write("\nMatrix Form:\n")
        f.write(f"{conf}\n")
        f.write("===== Basic Metrics =====\n")
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1-score : {f1:.4f}\n\n")

        f.write("===== Biometric Metrics =====\n")
        f.write(f"FAR (thr=0.5): {FAR:.4f}\n")
        f.write(f"FRR (thr=0.5): {FRR:.4f}\n")
        f.write(f"EER: {eer:.4f}\n")
        f.write(f"EER Threshold: {eer_threshold:.4f}\n")
        f.write(f"TAR@FAR=1%: {tar_at_far:.4f}\n")
        f.write(f"AUC: {roc_auc:.4f}\n")

    print(f"Metrics saved to {metrics_file}")

    ################################
    # 保存 ROC 图（标出EER）
    ################################

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")

    # 标出EER点
    plt.scatter(eer, 1 - eer, color='red', zorder=5,
                label=f"EER = {eer:.4f}")

    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    roc_file = os.path.join(output_dir, "ROC_curve.png")
    plt.savefig(roc_file, dpi=300)
    plt.close()

    print(f"ROC curve saved to {roc_file}")

    ################################
    # 保存 DET 图（标出EER）
    ################################

    plt.figure()
    plt.plot(fpr, fnr, label="DET Curve")

    # 标出EER点
    plt.scatter(eer, eer, color='red', zorder=5,
                label=f"EER = {eer:.4f}")

    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("DET Curve")
    plt.legend()
    plt.grid()

    det_file = os.path.join(output_dir, "DET_curve.png")
    plt.savefig(det_file, dpi=300)
    plt.close()

    print(f"DET curve saved to {det_file}")


############################################
# 主程序
############################################

if __name__ == "__main__":

    data_dir = "datasets/hilbert/bidmc/test"
    output_dir = "evalutation/seqs/ring/basic_GRU"
    model_name = 'GRUhilbert'
    model_path = ''
    flat_shape = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 输出保存路径
    ############################################
    # 加载模型
    ############################################

    model = Siamese(
        model_name=model_name,
        method=method,
        flat_shape=flat_shape
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"{model_path} model loaded.")
    os.makedirs(output_dir, exist_ok=True)
    print("Loading pairs...")
    image_pairs, labels = load_image_pairs(data_dir)

    print("Evaluating...")
    evaluate_model(image_pairs, labels, output_dir)
