import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.ndimage import gaussian_filter1d


def minmax(x):
    """
    normalization process
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def vad(spec, plot=False):
    mag = spec.squeeze().sum(axis=0)
    mag_ = minmax(mag)

    threshold = np.quantile(mag_, 0.5)

    va_ind1 = np.where(mag_ >= threshold, 1, 0)
    va_point = np.where(va_ind1 == 1)[0][[0, -1]]
    if va_point[0] < 0:
        va_point[0] = 0

    if plot:
        va_ind2 = np.zeros_like(mag)
        va_ind2[va_point[0]:va_point[1]] = 1
        plt.plot(mag_)
        plt.hlines(xmin=0, xmax=len(mag_), y=threshold)
        plt.plot(np.where(mag_ >= threshold, 1, 0))
        plt.plot(va_ind2)
    return va_point


def gvad(spec, plot=False, n_spare=3):
    """
    自定义的语音活动检测（VAD）函数。

    参数：
    - spec: 输入的幅度谱，形状为 [频率, 时间]
    - plot: 是否绘制能量图和检测到的语音段
    - n_spare: 在检测到的语音段前后增加的帧数，用于防止边界截断

    返回值：
    - va_point: 检测到的语音活动的起始和结束帧索引
    """
    # 将频谱从振幅转换成分贝尺度，并对每一帧做归一化处理，最后.sum对每一帧的频率能量做求和，得到一个时间轴上的能量表示
    time_mag = minmax(librosa.amplitude_to_db(spec)).sum(axis=0)
    # 将时间能量大小进行排序，找出变化剧烈的位置
    sorted_time_mag = np.sort(time_mag)
    # 用高斯滤波器对排序后的能量做平滑，sigma决定平滑程度
    sorted_time_mag_smoothed = gaussian_filter1d(sorted_time_mag, sigma=10)
    # 求平滑曲线的梯度（斜率），用于找出能量快速上升的位置
    sorted_time_mag_smoothed_grad = np.gradient(sorted_time_mag_smoothed)
    # argmax找到梯度最大的点（也就是能量突增点）
    threshold = sorted_time_mag_smoothed[np.argmax(sorted_time_mag_smoothed_grad)]
    va_point = (np.where(time_mag > threshold)[0][[0, -1]] + np.array([-n_spare, n_spare])).tolist()
    if va_point[0] < 0:
        va_point[0] = 0
    if plot:
        plt.plot(time_mag)
        vad_line = np.zeros_like(time_mag)
        vad_line[va_point[0]: va_point[1]] = np.max(time_mag)
        plt.plot(vad_line)
    return va_point
