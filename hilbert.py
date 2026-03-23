import os
import numpy as np
from scipy.signal import hilbert


def hilbert_features(npy_path, fs):
    signal = np.load(npy_path)

    if signal.ndim != 1:
        signal = signal.flatten()

    analytic_signal = hilbert(signal)

    amplitude = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(phase) * fs / (2.0 * np.pi)
    inst_freq = np.insert(inst_freq, 0, inst_freq[0])

    return amplitude, phase, inst_freq


def process_all(input_root, output_root, fs):
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".npy"):

                input_path = os.path.join(root, file)

                # ===== 计算相对路径 =====
                rel_path = os.path.relpath(root, input_root)
                save_dir = os.path.join(output_root, rel_path)

                os.makedirs(save_dir, exist_ok=True)

                # ===== 构造新文件名 =====
                # bidmc02_2.npy → bidmc02_hilbert2.npy
                name, ext = os.path.splitext(file)
                parts = name.split("_")

                if len(parts) == 2:
                    new_name = f"{parts[0]}_hilbert{parts[1]}.npy"
                else:
                    new_name = f"{name}_hilbert.npy"

                save_path = os.path.join(save_dir, new_name)

                # ===== 计算 Hilbert 特征 =====
                amp, phase, freq = hilbert_features(input_path, fs)

                # ===== 组合为3通道 =====
                three_channel = np.stack([amp, phase, freq], axis=0)

                # ===== 保存 =====
                np.save(save_path, three_channel)

                print(f"Saved: {save_path}")


if __name__ == "__main__":

    input_root = "datasets/seqdatas/Cap/Cap_normal_train"
    output_root = "datasets/hilbert/Cap/Cap_normal_train"

    fs = 125  # BIDMC PPG采样率

    process_all(input_root, output_root, fs)

    print("全部处理完成")