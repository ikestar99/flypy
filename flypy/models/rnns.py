#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 08 09:00:00 2024
@author: ike
https://github.com/asilvaalex4/bravo_util_fncs/blob/main/bravo_util_fncs/
models.py#L59
"""
import torch
import torch.nn as nn


class LangClassifierConvLSTM(torch.nn.Module):
    def __init__(
            self,
            n_features: int,
            n_layers: int,
            n_hidden: int,
            p_dropout_lstm,
            p_dropout_conv,
            n_outputs: int,
            kernel_size: int = 2
    ):
        super(LangClassifierConvLSTM, self).__init__()

        # LSTM output is (batch_size, seq_len, num_directions * hidden_size)
        self.conv_1d = torch.nn.Conv1d(
            in_channels=n_features, out_channels=n_features,
            kernel_size=kernel_size, stride=kernel_size)
        self.gru_lstm = torch.nn.GRU(
            input_size=n_features, hidden_size=n_hidden, num_layers=n_layers,
            batch_first=True, bidirectional=True, dropout=p_dropout_lstm)
        self.dropout = nn.Dropout(p_dropout_conv)
        self.linear = nn.Linear(2 * n_hidden, n_outputs)

    def forward(self, x):
        # x.shape = (trials, time, features)
        # permute (0, 2, 1) --> trials, electrodes, time = temporal convolution
        x = self.conv_1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        x, _ = self.gru_lstm(x)
        x = self.linear(x)
        return x[:, -1]
