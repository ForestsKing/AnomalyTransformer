import math

import numpy as np
import torch
from torch import nn


class AnomalyAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, windows_size, dropout):
        super(AnomalyAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.W_sigma = nn.Linear(self.d_model, self.n_heads, bias=False)

        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        distances = torch.zeros((windows_size, windows_size))
        for i in range(windows_size):
            for j in range(windows_size):
                distances[i][j] = (i - j) ** 2
        self.register_buffer("distances", distances)

    def forward(self, input_Q, input_K, input_V):
        residual = input_Q.clone()
        Q = self.W_Q(input_Q).view(input_Q.size(0), input_Q.size(1), self.n_heads, self.d_k)
        K = self.W_K(input_K).view(input_K.size(0), input_K.size(1), self.n_heads, self.d_k)
        V = self.W_V(input_V).view(input_V.size(0), input_V.size(1), self.n_heads, self.d_v)
        sigma = self.W_sigma(input_Q).view(input_V.size(0), input_V.size(1), self.n_heads)

        Q, K, V, sigma = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), sigma.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        series = attn
        context = torch.matmul(attn, V)

        context = context.transpose(1, 2)
        context = context.reshape(input_Q.size(0), input_Q.size(1), self.n_heads * self.d_v)
        output = self.fc(context)
        output = self.dropout(self.norm(output + residual))

        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, input_Q.size(1))
        sigma = torch.sigmoid(sigma * 5) + 1e-5  # 不加 1e-5 会 nan
        sigma = torch.pow(3, sigma) - 1
        prior = self.distances.unsqueeze(0).unsqueeze(0).expand(sigma.shape).to(input_Q.device)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior / 2 / (sigma ** 2))
        prior = prior / torch.unsqueeze(torch.sum(prior, dim=-1), dim=-1).expand(prior.shape)

        return output, series, prior
