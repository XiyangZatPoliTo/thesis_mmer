from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
from dataset.dataset import RAVDESSMultimodalDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 设置字体为 SimHei，以支持中文显示
matplotlib.rcParams['font.family'] = 'SimHei'
# 解决负号 '-' 显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False


# 初始化 Dataset
dataset = RAVDESSMultimodalDataset(
    dataset_dir="F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24/",
    mel_specs_kwargs={"n_mels": 128, "n_fft": 2048, "hop_length": 512},
    frames=15,
    verbose=True,
    debug=True
)

print(f"🔍 样本总数: {len(dataset)}")

# 手动取一条样本
sample = dataset[0]
print("\n📦 第一个样本内容：")
print("音频特征 shape:", sample[0].shape)        # [128, T]
print("视频帧 shape:", sample[1].shape)          # [15, 3, 224, 224]
print("标签编号:", sample[2].item())
print("标签名", sample[3])

# 构建 dataloader
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 打印前两批数据
print("\n📦 查看前两批样本的结构：")
for i, batch in enumerate(loader):
    if len(batch) == 4:
        audio_batch, video_batch, label_batch, label_names = batch
    else:
        audio_batch, video_batch, label_batch = batch
        label_names = ["N/A"] * len(label_batch)

    print(f"\nBatch {i+1}")
    print("  audio batch shape:", audio_batch.shape)  # [B, 128, T]
    print("  video batch shape:", video_batch.shape)  # [B, 15, 3, 224, 224]
    print("  labels:", label_batch.tolist())
    print("  label names:", label_names)
    if i == 1:
        break

# 统计标签分布
# NOTE: 'neutral' emotion (01) only has normal intensity (01),
#       so each actor has 4 neutral samples instead of 8,
#       leading to total 96 instead of 192.
print("\n📊 正在统计标签分布...")
labels = [dataset[i][2].item() for i in range(len(dataset))]
counter = Counter(labels)

# 打印标签分布
for k in sorted(counter.keys()):
    print(f"类 {k}: {counter[k]} 个样本")

# 可视化标签分布
plt.bar([str(k) for k in sorted(counter.keys())], [counter[k] for k in sorted(counter.keys())])
plt.xlabel("情绪类别标签")
plt.ylabel("样本数")
plt.title("标签分布统计")
plt.grid(True)
plt.show()





# import os
# from collections import defaultdict
#
# root_dir = "F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24"
# video_counts = defaultdict(int)
# audio_counts = defaultdict(int)
#
# # 遍历每个演员文件夹
# for actor in sorted(os.listdir(root_dir)):
#     actor_path = os.path.join(root_dir, actor)
#     if not os.path.isdir(actor_path):
#         continue
#
#     for fname in os.listdir(actor_path):
#         if fname.startswith("02-") and fname.endswith("_facecropped.npy"):
#             video_counts[actor] += 1
#         elif fname.startswith("03-") and fname.endswith(".wav"):
#             audio_counts[actor] += 1
#
# # 打印结果
# print("🎬 每个演员的 video-only (.npy) 文件数 和 audio-only (.wav) 文件数：\n")
# total_video, total_audio = 0, 0
# for actor in sorted(audio_counts.keys() | video_counts.keys()):
#     v = video_counts.get(actor, 0)
#     a = audio_counts.get(actor, 0)
#     total_video += v
#     total_audio += a
#     print(f"{actor}:  video-only .npy = {v:3d}    audio-only .wav = {a:3d}")
#
# print("\n📊 总计:")
# print(f"video-only .npy 文件总数:  {total_video}")
# print(f"audio-only .wav 文件总数:  {total_audio}")

