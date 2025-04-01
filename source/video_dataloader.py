import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from print_time import print_checkpoint_message

def get_video_meta(dataset_dir):
    """
    这个部分和源代码有所差别，目的是为了让代码风格和audio部分的get_meta相似
    获取原始音频集，并将文件名中的identifier进行分割，最后返回一个DataFrame，内容为数据集中每个文件的features
    """
    # 使用glob查找指定路径下所有子目录内的.npy文件，此类型文件由extract_face.py提取获得
    video_list = glob.glob(os.path.join(dataset_dir, '*/*.npy'))
    print(f"videos list: {video_list}")
    # 定义文件名中的features
    features = ['modality', 'vocal_channel', 'emotion',
                'emotion_intensity', 'statement', 'repetition', 'actor']
    # 创建一个空的dataframe，列名为features
    meta_info = pd.DataFrame(columns=features)
    for fpath in video_list:
        # 分离路径名，并保留.wav的filename，without file extension
        fn = fpath.split("\\")[-1].split("_")[0]
        print(fn)
        # 创建一个字典，将文件名中的各部分与features对应起来
        video_info = {f: [c] for c, f in zip(fn.split("-"), features)}
        print(video_info)
        # 将当前的元数据存进meta_info
        meta_info = pd.concat((meta_info, pd.DataFrame(video_info)), axis=0)
        print(meta_info)
    # 在DataFrame的第一列插入'speech_list'列，内容为所有文件路径
    meta_info.insert(0, "video_list", video_list)
    return meta_info


class VideoDataset(Dataset):
    def __init__(self, meta_info, transform=None):
        """
        构造自定义Dataset类
        Args:
            meta_info: 包含文件路径和特征信息的DataFrame
            transform: 用于数据预处理的变换函数
        """
        self.meta_info = meta_info
        self.transform = transform

    def __len__(self):
        return len(self.meta_info)

    def __getitem__(self, idx):
        npy_file = self.meta_info.iloc[idx]['npy_list']
        # 加载视频特征（假设.npy文件中包含的视频特征）
        video_features = np.load(npy_file)

        # 获取相关的标签信息
        emotion = int(self.meta_info.iloc[idx]['emotion']) - 1  # 假设情感标签从1开始，转换为0开始
        actor = int(self.meta_info.iloc[idx]['actor']) - 1  # 假设演员ID从1开始，转换为0开始
        info = self.meta_info.iloc[idx]

        # 选择数值类型的列
        numeric_info = pd.to_numeric(info, errors='coerce')  # 将非数值类型转换为NaN
        numeric_info = numeric_info.dropna()

        info_tensor = torch.tensor(numeric_info.values, dtype=torch.float32)  # 转换为Tensor

        # 将info（pandas.Series）转换为numpy数组或Tensor
        # info_tensor = torch.tensor(info.values, dtype=torch.float32)  # 这里是转换步骤

        if self.transform:
            video_features = self.transform(video_features)

        # 返回Tensor类型的数据
        return torch.tensor(video_features, dtype=torch.float32), torch.tensor(emotion, dtype=torch.long), info_tensor


def get_train_test_split(meta, test_ratio=0.2, random_state=None):
    """
    按照指定的比例划分训练集和测试集。
    """
    from sklearn.model_selection import train_test_split
    train_meta, test_meta = train_test_split(meta, test_size=test_ratio, random_state=random_state,
                                             stratify=meta['emotion'])
    return train_meta, test_meta


if __name__ == "__main__":
    # 获取视频数据的元数据
    dataset_dir = 'F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24'  # 替换为你的视频数据路径
    print_checkpoint_message("m1")
    meta_info = get_video_meta(dataset_dir)
    print_checkpoint_message("m2")


    # # 获取训练集和测试集
    # train_meta, test_meta = get_train_test_split(meta_info)
    # print_checkpoint_message("m3")
    # # 创建Dataset对象
    # train_dataset = VideoDataset(train_meta)
    # test_dataset = VideoDataset(test_meta)
    #
    # # 创建DataLoader
    # batch_size = 4
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #
    # # 打印第一个批次数据
    # for video_data, labels, info in train_loader:
    #     print(video_data.shape)  # 打印视频特征的形状
    #     print(labels)  # 打印情感标签
    #     break  # 打印一个批次的数据
