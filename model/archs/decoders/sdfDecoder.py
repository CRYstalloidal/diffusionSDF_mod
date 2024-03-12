#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import sys
import torch.nn.init as init
import numpy as np


class SdfDecoder(nn.Module):
    def __init__(self, latent_size=256, hidden_dim=512,input_size=None,dropout=0.5
                 ):
        super().__init__()
        self.latent_size = latent_size
        self.input_size = latent_size+3 if input_size is None else input_size
        self.dp = nn.Dropout(p=dropout)

        self.block1 = nn.Sequential(
                nn.Linear(self.input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                self.dp,
                nn.ReLU(),
            )

        self.block2=nn.Linear(hidden_dim+self.input_size, hidden_dim)
        self.actn=nn.ReLU()

        self.block3 = nn.Linear(hidden_dim, 1)




    def forward(self, x):

        out = self.block1(x)
        out = torch.cat([out,x],dim=-1)
        out = self.block2(out)
        out = self.block3(self.actn(out))

        return out.squeeze()

