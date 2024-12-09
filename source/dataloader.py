import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from utils.audio_tools import gvad


def get_meta(dataset_dir):
    """
    获取原始音频集，并将文件名中的features进行分割，最后返回一个DataFrame，内容为数据集中每个文件的features
    """
    # 使用glob查找指定路径下所有子目录内的所有.wav，并返回每一个wav的路径
    speech_list = glob.glob(os.path.join(dataset_dir, '*/*.wav'))
    # 定义.wav文件名中的features
    features = ['modality', 'vocal_channel', 'emotion', 'emotion_intensity', 'statement', 'repetition', 'actor']
    # 创建一个空的dataframe，列名为features
    meta_info = pd.DataFrame(columns=features)
    for fpath in speech_list:
        # 分离路径名，并保留.wav的filename，without file extension
        fn = fpath.split("/")[-1].split(".")[0]
        # 创建一个字典，将文件名中的各部分与features对应起来
        speech_info = {f: [c] for c, f in zip(fn.split("-"), features)}
        # 将当前的元数据存进meta_info
        meta_info = pd.concat((meta_info, pd.DataFrame(speech_info)), axis=0)
    # 在DataFrame的第一列插入'speech_list'列，内容为所有文件路径
    meta_info.insert(0, "speech_list", speech_list)
    return meta_info


def split_meta(meta, kfcv=False, n_splits=5, test_ratio=0.2, stratify=True, target=None, verbose=True,
               random_state=None):
    """
    拆分数据集，默认使用StratifiedKFold,StratifiedKFold 是一种用于交叉验证的技术，特别适用于分类问题。
    它在分割数据集时，确保每个折中各类别的比例与原始数据集中相同。
    这对于不平衡数据集特别重要，因为它可以保证每个类别在每个折中的分布相似，从而避免因数据不平衡带来的偏差。
    """
    if meta.empty:
        raise ValueError("The input DataFrame 'meta' is empty. Please provide a non-empty DataFrame.")

    if target not in meta.columns:
        raise ValueError(f"The target column '{target}' does not exist in the DataFrame.")

    if kfcv:
        # 使用默认的StratifiedKFold对数据集进行拆分
        skfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        if verbose:
            # 输出数据集大小和比例信息
            N = len(meta)
            print('size: {}, {}'.format(int(N * (1 - 1 / n_splits)), int(N * 1 / n_splits)))
            print('ratio: {}, {}'.format(1 - 1 / n_splits, 1 / n_splits))
        # 返回K折交叉验证的分割结果
        return skfold.split(meta[['speech_list', target]], meta[target])
    else:
        # 否则，进行常规的训练/测试分割
        if len(meta) < 2:
            raise ValueError("The input DataFrame 'meta' does not have enough samples to perform a train/test split.")

        train_meta, test_meta = train_test_split(meta[['speech_list', target]],
                                                 stratify=meta[target] if stratify else None,
                                                 shuffle=True,
                                                 test_size=test_ratio,
                                                 random_state=random_state)
        if verbose:
            # 输出数据集大小和比例信息
            N = len(meta)
            print('size: {}, {}'.format(len(train_meta), len(test_meta)))
            print('ratio: {}, {}'.format(len(train_meta) / N, len(test_meta) / N))
        return train_meta, test_meta


def mels2batch(batch):
    mel_specs = []  # 存储每个信号的梅尔频谱图
    info_list = []  # 存储每个信号的信息
    l_list = []  # 存储每个梅尔频谱图的长度
    for mel_spec, sr, info, fn in batch:
        info_list.append(info)
        l_list.append(mel_spec.shape[1])
        mel_specs.append(mel_spec.transpose(0, 1).contiguous())
    padded_mel_specs = nn.utils.rnn.pad_sequence(mel_specs, batch_first=True, padding_value=0).unsqueeze(1).transpose(2,
                                                                                                                      3).contiguous()
    # 返回填充后的Mel频谱图，情感标签，和每个样本的长度
    return padded_mel_specs, torch.Tensor(
        pd.DataFrame(info_list).emotion.values.astype('int16') - 1).long(), torch.tensor(l_list)


def sig2batch(batch, mel_specs_kwargs=None):
    if mel_specs_kwargs is None:
        mel_specs_kwargs = {}
    mel_specs = []
    info_list = []
    l_list = []
    for sig, sr, info, fn in batch:
        info_list.append(info)
        mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(sig, sr=sr, **mel_specs_kwargs))
        va_point = gvad(librosa.db_to_amplitude(mel_spec) ** 2)
        mel_spec = mel_spec[:, va_point[0]:va_point[1]]
        l_list.append(mel_spec.shape[1])
        mel_specs.append(torch.Tensor(mel_spec).transpose(0, 1))
    padded_mel_specs = nn.utils.rnn.pad_sequence(mel_specs, batch_first=True).unsqueeze(1).transpose(2, 3)
    return padded_mel_specs, torch.Tensor(pd.DataFrame(info_list).emotion.values.astype('int16') - 1).long(), l_list


if __name__ == "__main__":
    # 初始化时需要在env.py里同时将数据集地址换掉
    meta = get_meta('F:/RAVDESS_original/Audio_Speech_Actors_01-24')
