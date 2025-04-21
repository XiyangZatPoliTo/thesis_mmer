# dataset处理
# 主要作用：将音频文件转换为mels并存储在tensor中，方便后续模型输入
# torch.__version__
# Out[4]: '2.2.2+cpu'
# numpy.__version__
# Out[6]: '2.0.2'
# import torch时报错
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import glob
import librosa
from utils.audio_tools import gvad
import numpy as np
import os
from collections import defaultdict


class RAVDESSMelS(Dataset):
    """
    override the RAVDESS dataset and make it suitable for the mel spectrogram process
    """
    def __init__(self, dataset_dir, mel_specs_kwargs=None, speech_list=None, out_dir=None):
        if mel_specs_kwargs is None:
            mel_specs_kwargs = {}
        self.dataset_dir = dataset_dir
        self.mel_specs_kwargs = mel_specs_kwargs if mel_specs_kwargs else {}

        self.out_dir = out_dir

        if speech_list is None:
            self.speech_list = glob.glob(f"{dataset_dir}*/*.wav")
        else:
            self.speech_list = speech_list

    def __getitem__(self, index):
        fpath = self.speech_list[index]
        fn = fpath.split("/")[-1].split(".")[0]

        sig, sr = librosa.load(self.speech_list[index])

        features = ['modality', 'vocal_channel', 'emotion', 'emotion_intensity', 'statement', 'repetition', 'actor']
        info = {f: c for c, f in zip(fn.split("-"), features)}
        mel_spec = truncate_melspecs(sig, return_va_point=False, sr=sr, mel_specs_kwargs=self.mel_specs_kwargs)
        return torch.Tensor(mel_spec), sr, info, fn

    def __len__(self):
        return len(self.speech_list)


def truncate_melspecs(sig, return_va_point=False, sr=22050, mel_specs_kwargs=None):
    """
    Convert the spectrogram to an amplitude map
    and intercept the main part with the gvad function
    """
    if mel_specs_kwargs is None:
        mel_specs_kwargs = {}
    mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=sig, sr=sr, **mel_specs_kwargs))
    va_point = gvad(librosa.db_to_amplitude(mel_spec) ** 2)  # 将db谱图转为普通振幅谱图
    # 用VAD检测到的有效音频片段起始帧和结束帧用在截取mel
    mel_spec = mel_spec[:, va_point[0]:va_point[1]]
    if return_va_point:
        return mel_spec, va_point
    else:
        return mel_spec


emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


class RAVDESSMultimodalDataset(Dataset):
    """
    新版本的数据集处理，将音频和视频的处理统一封装在一个class
    多模态数据集：从 .wav 提取 Mel 频谱图，同时加载已预处理好的人脸帧（.npy）
    你的视频预处理脚本应已生成了 _facecropped.npy 文件（每条样本约 15 帧）
    """
    def __init__(self, dataset_dir, mel_specs_kwargs=None, frames=15, transform=None, verbose=True, debug=False):
        self.dataset_dir = dataset_dir
        self.mel_specs_kwargs = mel_specs_kwargs if mel_specs_kwargs else {}
        self.frames = frames
        self.transform = transform
        self.verbose = verbose  # 增加一个开关，打印缺失文件
        self.debug = debug

        # RAVDESS 标签映射表（情绪编号 → label）
        self.label_map = {k: int(k) for k in emotion_dict.keys()}  # 01 -> 1, ..., 08 -> 8

        # 获取所有 .wav 文件路径（音频主控）
        self.wav_paths = []
        for root, _, files in os.walk(dataset_dir):
            for f in files:
                if f.endswith(".wav") and f.startswith("03"):
                    self.wav_paths.append(os.path.join(root, f))

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        label_id = base_name.split("-")[2]
        label = torch.tensor(self.label_map[label_id], dtype=torch.long)
        label_name = emotion_dict[label_id]

        # === 音频特征提取（audio-only） ===
        try:
            sig, sr = librosa.load(wav_path, sr=22050)
            mel_spec = librosa.power_to_db(
                librosa.feature.melspectrogram(y=sig, sr=sr, **self.mel_specs_kwargs)
            )
            va = gvad(librosa.db_to_amplitude(mel_spec) ** 2)
            mel_spec = mel_spec[:, va[0]:va[1]]

            max_len = 128  # 统一时间长度，后续可搭配self.n_mels和self.max_len变为可调整的参数
            if mel_spec.shape[1] > max_len:
                mel_spec = mel_spec[:, :max_len]
            else:
                pad_width = max_len - mel_spec.shape[1]
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')

            audio_feat = torch.tensor(mel_spec, dtype=torch.float32)  # shape: [128, T]
        except Exception as e:
            if self.verbose:
                print(f"[AUDIO ERROR] Failed to process {wav_path}: {e}")

            #audio_feat = torch.zeros(self.n_mels, self.max_len)
            audio_feat = torch.zeros(128, 128)

        # === 视频模态：通过后缀部分匹配对应的.npy ===
        # 更正了在查找视频模态文件时的命名问题，之前是通过wav文件查找，但是忽略了因模态不同
        # 而出现的名称替换问题，仅音频以03开头，仅视频以02开头
        video_suffix = "-".join(base_name.split("-")[1:])
        actor_dir = os.path.dirname(wav_path)
        candidate_npy_files = [f for f in os.listdir(actor_dir) if f.startswith("02-") and f.endswith("_facecropped.npy")]

        npy_path = None
        for fname in candidate_npy_files:
            if video_suffix in fname:
                npy_path = os.path.join(actor_dir, fname)
                break

        if npy_path is None or not os.path.exists(npy_path):
            if self.verbose:
                print(f"[VIDEO WARNING] Missing .npy file for: {base_name}")
            # 若找不到对应视频，使用全 0 占位帧
            video_faces = torch.zeros(self.frames, 3, 224, 224)
        else:
            try:
                faces_np = np.load(npy_path)  # shape: [T, 224, 224, 3]
                faces_np = faces_np[:self.frames]  # 裁剪/限制帧数
                faces = torch.tensor(faces_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # [T, 3, 224, 224]
                if self.transform:
                    faces = self.transform(faces)
                video_faces = faces
            except Exception as e:
                if self.verbose:
                    print(f"[VIDEO ERROR] Failed to load {npy_path}: {e}")
                video_faces = torch.zeros(self.frames, 3, 224, 224)

        if self.debug:
            # 在train的时候出现IndexError: Target 8 is out of bounds
            # 意味着传给CrossEntropyLoss的labels中，出现了值为 8 的标签
            # 需要将标签编号做-1处理
            # label -> int(label_id) - 1
            return audio_feat, video_faces, int(label_id) - 1, label_name
        else:
            return audio_feat, video_faces, int(label_id) - 1

    def stats_per_actor(self):
        """
        统计每个演员的样本数量
        """
        counts = defaultdict(int)  # 每次访问不存在的key时，不会报错，而是自动初始化为0
        for path in self.wav_paths:
            actor_folder = os.path.basename(os.path.dirname(path))  # Actor_01, Actor_02...
            counts[actor_folder] += 1

        print("\n📊 样本统计（每位演员）：")
        for actor, count in sorted(counts.items()):
            print(f"{actor}: {count} samples")


# #dataset使用实例
# dataset = RAVDESSMultimodalDataset(
#     dataset_dir="F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24/",
#     mel_specs_kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512},
#     frames=15,
#     verbose=True  # 打印缺失文件
# )

# # 统计每位演员的样本数量（可选）
# dataset.stats_per_actor()