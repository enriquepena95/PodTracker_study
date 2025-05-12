#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:29:44 2024

@author: enrique
"""

import os
import pandas as pd
import numpy as np

# Base folder
base_folder = os.path.expanduser("~/PodTracker_study/data/decision_tree_data")

# Read CSV files

red_df = pd.read_csv(os.path.join(base_folder, 'red_train.csv'))
white_df = pd.read_csv(os.path.join(base_folder,'white_train.csv'))
blue_df = pd.read_csv(os.path.join(base_folder, 'blue_train.csv'))

# Add 'class' column
red_df['class'] = 'Jumbo'
white_df['class'] = 'Fancy'
blue_df['class'] = 'No1'

# Find the class with the smallest number of observations
min_samples = min(len(red_df), len(white_df), len(blue_df))

# Sample balanced datasets
red_balanced = red_df.sample(n=min_samples, random_state=42)
white_balanced = white_df.sample(n=min_samples, random_state=42)
blue_balanced = blue_df.sample(n=min_samples, random_state=42)

# Concatenate balanced datasets
balanced_df = pd.concat([red_balanced, white_balanced, blue_balanced], ignore_index=True)

# Shuffle the concatenated DataFrame
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

#balanced_df = balanced_df.drop(columns=['Unnamed: 0'])

# Output balanced DataFrame
print(balanced_df)

balanced_df.to_csv(os.path.join(base_folder,'balanced_df.csv'))
