import torch
from torch import nn

class SimilarityModel(nn.Module):

    def __init__(self):
        super(SimilarityModel, self).__init__()

        # 定义一个卷积层，用于特征提取。
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Conv1d(128, 512, kernel_size=3),
        )

        # 定义一个线性层，用来判断两张图片是否为同一类别
        self.sim = nn.Sequential(
            nn.Flatten(),
            nn.Linear(13184, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, sample1, sample2):
        # 使用卷积层提取sample1的特征
        sample1_features = self.conv(sample1)
        # 使用卷积层提取sample2的特征
        sample2_features = self.conv(sample2)
        # 将 |sample1-sample2| 的结果送给线性层，判断它们的相似度。
        return self.sim(torch.abs(sample1_features - sample2_features))
