import torch.nn as nn

from models import video_cnn
from data import consts


class FlowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = video_cnn.VideoCNN(input_dim=2, se=False)
        self.gru = nn.GRU(consts.EMBED_SIZE, consts.EMBED_SIZE, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.dropout = nn.Dropout(0.5)
        self.liner = nn.Linear(consts.EMBED_SIZE * 2, consts.EMBED_SIZE)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        self.gru.flatten_parameters()
        x = self.backbone(x)
        x, _ = self.gru(x)
        x = self.liner(x)
        x = self.relu(x)
        x = x.mean(1)
        x = self.dropout(x)
        return x
