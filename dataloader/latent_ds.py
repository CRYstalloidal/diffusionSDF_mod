#!/usr/bin/env python3

import numpy as np
import time 
import logging
import os
import random
import torch
import torch.utils.data

import pandas as pd 
import csv

class LatentDS(torch.utils.data.Dataset):
    def __init__(self,numpy_file,pc_file):
        tmp_data = np.load(numpy_file)
        self.data = torch.from_numpy(tmp_data).float()
        tmp_pc = np.load(pc_file)
        self.pc = torch.from_numpy(tmp_pc).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):    
        data_dict = {
            "latent":self.data[idx],
            "point_cloud":self.pc[idx]
            }
        return data_dict
