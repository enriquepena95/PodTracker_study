#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:27:44 2024

@author: enrique
"""

import torch
import torch.nn as nn

class PeanutClassifier(nn.Module):
    def __init__(self):
        super(PeanutClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 32)  # Input size 3 (length, width, area), output size 32
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 3)  # Output size 3 (number of classes)
        

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x