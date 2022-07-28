import torch
import torch.nn.functional as F
from torch import nn

from model.atten import AnomalyAttention
from model.embed import DataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, windows_size, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = AnomalyAttention(d_k, d_v, d_model, n_heads, windows_size, dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=(1,))
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=(1,))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        x, series, prior = self.attention(x, x, x)

        residual = x.clone()
        x = self.dropout(self.activation(self.conv1(x.permute(0, 2, 1))))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        return self.norm(residual + x), series, prior


class AnomalyTransformer(nn.Module):
    def __init__(self, d_k=64, d_v=64, d_model=512, d_ff=2048, n_heads=8, n_layer=3, windows_size=100, d_feature=55, dropout=0.2):
        super(AnomalyTransformer, self).__init__()

        self.embedding = DataEmbedding(d_feature, d_model, dropout)

        self.encoder = nn.ModuleList()
        for _ in range(n_layer):
            self.encoder.append(
                EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, windows_size, dropout)
            )

        self.norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, d_feature, bias=True)

    def forward(self, x):
        series_list, prior_list = [], []
        x = self.embedding(x)

        for layer in self.encoder:
            x, series, prior = layer(x)
            series_list.append(series)
            prior_list.append(prior)

        x = self.norm(x)
        x = self.projection(x)

        series = torch.stack(series_list).transpose(0, 1)
        prior = torch.stack(prior_list).transpose(0, 1)

        return x, series, prior
