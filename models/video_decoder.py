import torch
from torch import nn

from data import consts


class AdIN3d(nn.Module):
    def __init__(self, num_feature, embed_size=consts.EMBED_SIZE):
        super().__init__()
        self.mlp = nn.Linear(embed_size, num_feature * 2)
        self.relu = nn.ReLU(inplace=True)
        self.instance_norm = nn.InstanceNorm3d(num_feature, affine=False)

    def forward(self, x, latent):
        style = self.mlp(latent)
        style = self.relu(style)
        shape = [-1, 2, x.shape[1], 1, 1, 1]
        style = style.view(shape)
        x = self.instance_norm(x)
        x = x * (style[:, 0] + 1.0) + style[:, 1]
        return x


class Norm3d(nn.Module):
    def __init__(self, num_feature, mode="adin"):
        super().__init__()
        self.mode = mode
        if mode == "batch":
            self.norm = nn.BatchNorm3d(num_feature)
        elif mode == "adin":
            self.norm = AdIN3d(num_feature)
        elif mode == "instance":
            self.norm = nn.InstanceNorm3d(num_feature)
        else:
            self.norm = None

    def forward(self, x, latent=None):
        if self.mode == "batch" or self.mode == "instance":
            return self.norm(x)
        elif self.mode == "adin":
            return self.norm(x, latent)
        else:
            return x


class VideoDecoder2(nn.Module):
    def __init__(self, norm_mode="adin", embed_size=512):
        super().__init__()
        self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.conv1 = nn.Conv3d(embed_size, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), padding_mode="replicate")
        self.norm1 = Norm3d(96, mode=norm_mode)

        self.upsample2 = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.conv2 = nn.Conv3d(96, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), padding_mode="replicate")
        self.norm2 = Norm3d(64, mode=norm_mode)

        self.upsample3 = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.conv3 = nn.Conv3d(64, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2), padding_mode="replicate")
        self.norm3 = Norm3d(32, mode=norm_mode)

        self.upsample4 = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.conv4 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), padding_mode="replicate")
        self.norm4 = Norm3d(16, mode=norm_mode)

        self.conv5 = nn.Conv3d(16, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 1, 2), padding_mode="replicate")
        self.dropout = nn.Dropout3d(p=0.5)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, latent=None):

        out = self.conv1(x)
        out = self.norm1(out, latent)
        out = self.relu(out)
        out = self.upsample1(out)

        out = self.conv2(out)
        out = self.norm2(out, latent)
        out = self.relu(out)
        out = self.upsample2(out)

        out = self.conv3(out)
        out = self.norm3(out, latent)
        out = self.relu(out)
        out = self.upsample3(out)

        out = self.conv4(out)
        out = self.norm4(out, latent)
        out = self.relu(out)
        out = self.upsample4(out)
        out = self.conv5(out)
        out = self.sigmoid(out)
        return out
