import h5py
import torch
import torch.nn as nn
from utility.batch_test import *
import torch.nn as nn

class UserTextEncoder(nn.Module):
    def __init__(self, dim=768, dropout_rate=0.2):
        """
        结构: 768 → 512 → 768（保留原维度，增强表达）
        """
        super(UserTextEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim, 512),
            # nn.ReLU(),
            nn.GELU(),
            # nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, dim)
        )


    def forward(self, text_vec):
        """
        :param text_vec: (batch_size, 768)
        :return: (batch_size, 768)
        """
        return self.encoder(text_vec)