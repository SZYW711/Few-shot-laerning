import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from net import SimilarityModel
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader,TensorDataset
import scipy.io
import h5py


# 训练集文件夹路径
folder_path = '/Users/siwen/PycharmProjects/few-shot/train'
batch_size = 32

data_list = []
label_list = []

for subfolder in ['0', '1']:
    subfolder_path = os.path.join(folder_path, subfolder)
    file_list = [file for file in os.listdir(subfolder_path) if file.endswith('.mat')]

    # 遍历子文件夹中的所有.mat文件
    for file in file_list:
        # 构建文件的完整路径
        file_path = os.path.join(subfolder_path, file)

        # 读取.mat文件
        mat_data = scipy.io.loadmat(file_path)

        # 访问特定变量的值
        data = mat_data['combined_data']

        # 将数据的形状从(600, 2)调整为(2, 600)
        data = np.transpose(data)

        # 将数据切分成两个600*1的片段，并转换为Tensor格式
        sample1 = torch.Tensor(data[0,:].reshape(1, -1))
        sample2 = torch.Tensor(data[1,:].reshape(1, -1))

        # 解析文件名，获取标签（文件夹名称）
        label = int(subfolder)

        # 将数据和标签添加到列表中
        data_list.append((sample1, sample2))
        label_list.append(label)

# 将数据和标签转换为Tensor格式
data_tensor = torch.stack([torch.cat((sample1.unsqueeze(1), sample2.unsqueeze(1)), dim=1) for sample1, sample2 in data_list])
data_tensor = data_tensor.squeeze(1)  # 去除第二个维度为1的维度
label_tensor = torch.tensor(label_list)

train_ids = TensorDataset(data_tensor,label_tensor)
train_dataloader = DataLoader(train_ids,batch_size=32,shuffle=True)

model = SimilarityModel()

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()

def train(model, train_dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    best_accuracy = 0.0

    # 创建日志文件
    log_file = open("training_log.txt", "w")
    log_file.write("Epoch\tAccuracy\n")

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, data in enumerate(train_dataloader, 1):
            x_data, labels = data
            sample1 = x_data[:, 0, :]
            sample2 = x_data[:, 1, :]
            sample1 = sample1.unsqueeze(1)
            sample2 = sample2.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(sample1, sample2)
            a = outputs.data
            # 计算损失
            loss = loss_fn(outputs.squeeze(1), labels.float())

            # 反向传播和优化
            loss.backward()
            optimizer.step()


            # 统计预测结果
            a = outputs.data
            # 根据a的值进行预测调整
            threshold = 0.8
            predicted = torch.full_like(labels, 2)
            predicted = torch.where(a[:, 0] > threshold, torch.tensor(1), predicted)
            predicted = torch.where(a[:, 0] < (1 - threshold), torch.tensor(0), predicted)

            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)


            running_loss += loss.item()


        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = correct_predictions / total_predictions

        # 打印损失和准确率
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.3f}, Accuracy: {epoch_accuracy:.3f}')

        # 保存最好的模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            save_model(model, "best_model.pt")

        # 记录准确率到日志文件
        log_file.write(f"{epoch+1}\t{epoch_accuracy}\n")

    # 保存最后一次的模型
    save_model(model, "final_model.pt")

    # 关闭日志文件
    log_file.close()

    print('训练完成。')

# 使用示例
train_ids = TensorDataset(data_tensor, label_tensor)
train_dataloader = DataLoader(train_ids, batch_size=32, shuffle=True)
model = SimilarityModel()

epoch = 100
learning_rate = 0.0006
train(model, train_dataloader, epoch, learning_rate)