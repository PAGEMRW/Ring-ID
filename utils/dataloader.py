import random

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from scipy.fft import fft, ifft

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class SiameseDataset(Dataset):
    def __init__(self, lines, labels, noise):
        self.train_lines    = lines
        self.train_labels   = labels
        self.types          = max(labels)
        self.noise          = noise
        

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, index):
        batch_images_path = []
        #------------------------------------------#
        #   首先选取三张类别相同的图片
        #------------------------------------------#
        c               = random.randint(0, self.types - 1)
        selected_path   = self.train_lines[self.train_labels[:] == c]
        while len(selected_path)<3:
            c               = random.randint(0, self.types - 1)
            selected_path   = self.train_lines[self.train_labels[:] == c]

        image_indexes = random.sample(range(0, len(selected_path)), 3)
        #------------------------------------------#
        #   取出两张类似的图片
        #   对于这两张图片，网络应当输出1
        #------------------------------------------#
        batch_images_path.append(selected_path[image_indexes[0]])
        batch_images_path.append(selected_path[image_indexes[1]])

        #------------------------------------------#
        #   取出两张不类似的图片
        #------------------------------------------#
        batch_images_path.append(selected_path[image_indexes[2]])
        #------------------------------------------#
        #   取出与当前的小类别不同的类
        #------------------------------------------#
        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.train_lines[self.train_labels == current_c]

        image_indexes = random.sample(range(0, len(selected_path)), 1)
        batch_images_path.append(selected_path[image_indexes[0]])
        
        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
        return images, labels

    def _convert_path_list_to_images_and_labels(self, path_list):
        #-------------------------------------------#
        #   len(path_list)      = 4
        #   len(path_list) / 2  = 2
        #-------------------------------------------#
        number_of_pairs = int(len(path_list) / 2)
        #-------------------------------------------#
        #   定义网络的输入图片和标签
        #-------------------------------------------#
        pairs_of_images = [[], []]
        labels          = np.zeros((number_of_pairs, 1))

        #-------------------------------------------#
        #   对图片对进行循环
        #   0,1为同一种类，2,3为不同种类
        #-------------------------------------------#
        for pair in range(number_of_pairs):
            #读取图片
            image1 = np.load(path_list[pair * 2])
            image2 = np.load(path_list[pair * 2 + 1])
            # 🧩 保证输入至少为2D (C, L)
            if image1.ndim == 1:
                image1 = np.expand_dims(image1, axis=0)
            if image2.ndim == 1:
                image2 = np.expand_dims(image2, axis=0)
            
            if self.noise:
                image1 = self.add_noise(image1)
                image2 = self.add_noise(image2)

            pairs_of_images[0].append(image1)
            pairs_of_images[1].append(image2)

            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        #-------------------------------------------#
        #   随机的排列组合
        #-------------------------------------------#
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0] = [pairs_of_images[0][i] for i in random_permutation]
        pairs_of_images[1] = [pairs_of_images[1][i] for i in random_permutation]
        return pairs_of_images, labels

    def add_noise(self, imfs, shift_max=50, scale_range=(0.8, 1.2),
                   noise_std=0.01, jitter_std=0.005, energy_redistrib_strength=0.1,
                   mask_prob=0.1, mask_len_ratio=0.05, random_state=None):
        """
        对一个 6×len 的 IMF 信号矩阵执行多种数据增强操作（顺序固定）
        
        参数:
            imfs: np.ndarray, shape=(6, len)
            shift_max: 最大时移样本数
            scale_range: 幅值缩放范围 (min, max)
            noise_std: 加性噪声标准差（相对于信号标准差）
            jitter_std: 频谱扰动标准差（频率方向）
            energy_redistrib_strength: IMF间能量扰动幅度
            mask_prob: 掩蔽起始概率
            mask_len_ratio: 每次掩蔽的长度占总长度比例
            random_state: 随机种子
            
        返回:
            augmented: np.ndarray, shape=(6, len)
        """
        rng = np.random.default_rng(random_state)
        imfs = imfs.copy()
        n_channels, n_samples = imfs.shape

        # 1️⃣ Time Shift
        shift = rng.integers(-shift_max, shift_max + 1)
        imfs = np.roll(imfs, shift, axis=1)

        # 2️⃣ Amplitude Scaling
        scale = rng.uniform(*scale_range, size=(n_channels, 1))
        imfs = imfs * scale

        # 3️⃣ Additive Noise
        stds = np.std(imfs, axis=1, keepdims=True)
        noise = rng.normal(0, noise_std * stds, size=imfs.shape)
        imfs = imfs + noise

        # 4️⃣ Spectral Jittering
        for i in range(n_channels):
            sig = imfs[i]
            spec = fft(sig)
            freqs = np.arange(len(spec))
            phase_jitter = rng.normal(0, jitter_std, size=freqs.shape)
            jittered_spec = spec * np.exp(1j * 2 * np.pi * phase_jitter)
            imfs[i] = np.real(ifft(jittered_spec))

        # 5️⃣ Energy Redistribution
        energies = np.sum(imfs ** 2, axis=1)
        total_energy = np.sum(energies)
        noise_energy = rng.normal(0, energy_redistrib_strength, size=energies.shape)
        new_energies = energies + noise_energy * energies
        new_energies = np.maximum(new_energies, 1e-8)
        new_energies /= np.sum(new_energies) / total_energy
        for i in range(n_channels):
            imfs[i] *= np.sqrt(new_energies[i] / energies[i])

        # 6️⃣ Time Masking
        for i in range(n_channels):
            if rng.random() < mask_prob:
                mask_len = int(n_samples * mask_len_ratio)
                start = rng.integers(0, n_samples - mask_len)
                imfs[i, start:start + mask_len] = 0

        return imfs


    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    

# DataLoader中collate_fn使用
def dataset_collate(batch):
    left_images     = []
    right_images    = []
    labels          = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])
            
    images = torch.from_numpy(np.array([left_images, right_images])).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)
    return images, labels