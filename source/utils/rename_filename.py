import os
# This file is for rename all the wrong filename in the specific directory

# 指定要遍历的目录
directory = "F:/RAVDESS_dataset/Audio&Video_Speech_Actors_01-24"  # 请替换为你的目标文件夹路径

flag = 0

# 遍历目录及其子目录中的所有文件
for root, dirs, files in os.walk(directory):
    for filename in files:
        # 检查文件名是否包含 'facecroppad'
        flag += 1
        if 'facecroppad' in filename:
            # 构造新文件名
            new_filename = filename.replace('facecroppad', 'facecropped')

            # 获取文件的完整路径
            old_file_path = os.path.join(root, filename)
            new_file_path = os.path.join(root, new_filename)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f'Renamed: {old_file_path} -> {new_file_path}')

        print(f"Flag: {flag}")

