# # 目前测试未通过
# # ImportError: numpy.core.multiarray failed to import
# # (auto-generated because you didn't call 'numpy.import_array()' after cimporting numpy;
# # use '<void>numpy._import_array' to disable if you are certain you don't need it).
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
#
#
# # 定义 minmax 归一化函数
# def minmax(x):
#     return (x - x.min()) / (x.max() - x.min() + 1e-8)
#
#
# # 定义 gvad 函数
# def gvad(spec, plot=False, n_spare=3):
#     """
#     自定义的语音活动检测（VAD）函数。
#
#     参数：
#     - spec: 输入的幅度谱，形状为 [频率, 时间]
#     - plot: 是否绘制能量图和检测到的语音段
#     - n_spare: 在检测到的语音段前后增加的帧数，用于防止边界截断
#
#     返回值：
#     - va_point: 检测到的语音活动的起始和结束帧索引
#     """
#     # 将幅度谱转换为分贝尺度并归一化
#     time_mag = minmax(librosa.amplitude_to_db(spec)).sum(axis=0)
#     # 对时间维度的能量进行排序和平滑处理
#     sorted_time_mag = np.sort(time_mag)
#     sorted_time_mag_smoothed = gaussian_filter1d(sorted_time_mag, sigma=10)
#     # 计算梯度并找到最大梯度对应的阈值
#     sorted_time_mag_smoothed_grad = np.gradient(sorted_time_mag_smoothed)
#     threshold = sorted_time_mag_smoothed[np.argmax(sorted_time_mag_smoothed_grad)]
#     # 找到超过阈值的时间帧索引，并在前后添加 n_spare 帧
#     va_point = (np.where(time_mag > threshold)[0][[0, -1]] + np.array([-n_spare, n_spare])).tolist()
#     # 防止索引越界
#     va_point[0] = max(va_point[0], 0)
#     va_point[1] = min(va_point[1], len(time_mag) - 1)
#     # 可视化
#     if plot:
#         plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
#         plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
#         plt.figure(figsize=(12, 4))
#         plt.plot(time_mag, label="帧能量")
#         vad_line = np.zeros_like(time_mag)
#         vad_line[va_point[0]: va_point[1]] = np.max(time_mag)
#         plt.plot(vad_line, label="检测到的语音段", linestyle='--', color='red')
#         plt.title("语音活动检测（GVAD）")
#         plt.xlabel("时间帧")
#         plt.ylabel("归一化能量")
#         plt.legend()
#         plt.grid(True)
#         plt.show()
#     return va_point
#
#
# # 加载示例音频文件
# audio_path = '03-01-01-01-01-01-01.wav'  # 请替换为你的音频文件路径
# y, sr = librosa.load(audio_path, sr=None)
#
# # 计算短时傅里叶变换（STFT）的幅度谱
# spec = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
#
# # 应用 gvad 函数并可视化结果
# vad_points = gvad(spec, plot=True, n_spare=5)
# print("检测到的语音活动区间（帧索引）:", vad_points)


import os

# 设置根文件夹路径
root_folder = 'F:/RAVDESS_original'

# 递归遍历所有文件夹和文件
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith("_facecroppad.avi"):
            old_path = os.path.join(dirpath, filename)
            new_filename = filename.replace("_facecroppad.avi", "_facecropped.avi")
            new_path = os.path.join(dirpath, new_filename)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
