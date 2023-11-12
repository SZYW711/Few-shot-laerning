import shutil
import os
import h5py
import numpy as np
import scipy.io as sio
import scipy.io

# 指定文件夹路径
folder_path = '/Users/siwen/Documents/MATLAB/fewshot'

# 目标文件夹路径（PyCharm工作目录）
target_folder = '/Users/siwen/PycharmProjects/few-shot'

# 遍历文件夹及子文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 确保文件扩展名为.mat
        if file.endswith('.mat'):
            # 构造源文件路径
            source_file = os.path.join(root, file)

            # 构造目标文件夹路径
            target_subfolder = os.path.relpath(root, folder_path)
            target_dir = os.path.join(target_folder, target_subfolder)

            # 如果目标文件夹不存在，则创建它
            os.makedirs(target_dir, exist_ok=True)

            # 构造目标文件路径
            target_file = os.path.join(target_dir, file)

            # 使用shutil.copy2函数复制.mat文件到目标文件夹
            shutil.copy2(source_file, target_file)


# 定义文件夹路径和文件名
file_path = '/Users/siwen/PycharmProjects/few-shot/test'  # 文件夹路径

# 读取test.mat文件
test_data = sio.loadmat(file_path + '/test.mat')
test_array = test_data['testData']
test_array = np.transpose(test_array)  # 转置转换

# 读取test_label.mat文件
test_label_data = sio.loadmat(file_path + '/test_label.mat')
test_label_array = test_label_data['testLabel']
test_label_array = np.transpose(test_label_array)  # 转置转换

# 打印读取的数据数组大小
print('test_array shape:', test_array.shape)
print('test_label_array shape:', test_label_array.shape)

file1_path = '/Users/siwen/Documents/MATLAB/traindata.mat'
file2_path = '/Users/siwen/Documents/MATLAB/trainlabel.mat'

train_data = scipy.io.loadmat(file1_path)
train_array = train_data['train_data']
trian_array = np.transpose(train_array)  # 转置转换

train_label = scipy.io.loadmat(file2_path)
train_label_array = train_label['train_label']
# train_label_array = np.transpose(train_label_array)  # 转置转换

# 打印读取的数据数组大小
print('train_array shape:', train_array.shape)
print('train_label_array shape:', train_label_array.shape)

np.save('trianData.npy', train_array)
np.save('trainLabel.npy', train_label_array)


# 将数据保存为numpy数组
np.save('test.npy', test_array)
np.save('test_label.npy', test_label_array)