#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:55:24 2024

@author: enrique
"""
#%% Import Packages
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
code_dir = os.path.expanduser("~/PodTracker_study/code")
sys.path.append(code_dir)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from random_seed import random_seed
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from sklearn.model_selection import KFold
from model_architecture import PeanutClassifier

#%% Set Random Seed
seed = 42
x = random_seed(seed=seed, deterministic=True)

#%% Set Device
device = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print("Using device", device)

#%% Prepare the dataset
base_folder = os.path.expanduser("~/PodTracker_study")
path_to_dataset = os.path.join(base_folder, "data/decision_tree_data/balanced_df.csv")
df = pd.read_csv(path_to_dataset)
df = df.drop(columns=['Unnamed: 0'])

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.path.join(base_folder, "model_checkpoints")
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

#%% Split the data into training and testing
# Extract features (X) and labels (y)
X = df.drop('class', axis=1)  # Assuming 'class' is the column containing labels
y = df['class']

# Split the data into 80% training and 20% val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# Assuming I have class labels 'red', 'white', and 'blue'
class_mapping = {'Jumbo': 0, 'Fancy': 1, 'No1': 2}

# Map class labels to numerical labels
y_train_mapped = y_train.map(class_mapping)
y_val_mapped = y_val.map(class_mapping)

# Assuming X_train, y_train, X_test, y_test are my training and testing data
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train_mapped.values, dtype=torch.long)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val_mapped.values, dtype=torch.long)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader =  DataLoader(val_dataset, batch_size=1, shuffle = False)


#%% Define the Optimizer Class

class OptimizerTemplate:
    
    def __init__(self, params: nn.ParameterList, lr: float)->None:
        self.params = list(params)
        self.lr = lr
        
    def zero_grad(self)->None:
        ## Set gradients of all parameters to zero
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_() # For second-order optimizers important
                p.grad.zero_()
    
    @torch.no_grad()
    def step(self)->None:
        ## Apply update step to all parameters
        for p in self.params:
            if p.grad is None: # We skip parameters without any gradients
                continue
            self.update_param(p)
            
    def update_param(self, p: nn.Parameter)->None:
        raise NotImplementedError

class SGDMomentum(OptimizerTemplate):
    
    def __init__(self, params: nn.ParameterList, lr: float, momentum: float=0.9)->None:
        super().__init__(params, lr)
        self.momentum = momentum # Corresponds to beta_1 in the equation above
        self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params} # Dict to store m_t
        
    def update_param(self, p:nn.Parameter)->None:
        self.param_momentum[p] = self.momentum*self.param_momentum[p]+(1-self.momentum)*p.grad.data # Update momentum
        p.data -= self.lr*self.param_momentum[p] # Update parameter

#%% Define the Training and Validation Functions 

def train_one_epoch(model: nn.Module, optimizer: OptimizerTemplate, loss_module, data_loader)->Tuple[float, int]:
    true_preds, count = 0.0, 0
    device = next(model.parameters()).device 
    model.train()
    

    for data, labels in data_loader:

        data, labels = data.to(device), labels.to(device)


        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_module(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Record statistics during training
        o, predicted = torch.max(outputs, 1)  # Predicted class is the max value in outputs
        true_preds += (predicted == labels).sum().item()  # Count the number of correct preds
        count += labels.size(0)  # Increase count by counts # in a batch 
        
    train_acc = true_preds / count
    #print(f"Train Accuracy: {train_acc}")
    return train_acc

@torch.no_grad()
def test_model(model, data_loader):
    true_preds, count = 0.0, 0
    model.eval()
    
    for data, labels in data_loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)

        # Record statistics during testing
        _, predicted = torch.max(outputs, 1)  
        true_preds += (predicted == labels).sum().item()
        count += labels.size(0)

    test_acc = true_preds / count
    #print(f"Test Accuracy: {test_acc}")
    return test_acc

def save_model(model, model_name, root_dir=CHECKPOINT_PATH):
    model.to(device)
    modelpath = os.path.join(root_dir, model_name)
    torch.save(model.state_dict(), modelpath)
    

def load_model(model, model_name, root_dir=CHECKPOINT_PATH):
    model.to(device)
    modelpath = os.path.join(root_dir, model_name)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    return model


# Define a function for training and validating for one fold
def train_and_validate_fold(model, optimizer, loss_module, train_loader, val_loader, num_epochs=25):
    best_val_acc = -1.0
    
    for epoch in range(1, num_epochs + 1):
        train_acc = train_one_epoch(model, optimizer, loss_module, train_loader)
        
        if epoch % 10 == 0 or epoch == num_epochs:
            acc = test_model(model, val_loader)
            if acc > best_val_acc:
                best_val_acc = acc
                save_model(model, "MyModel", CHECKPOINT_PATH)

            print(
                f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {acc*100.0:05.2f}%, Best validation accuracy: {best_val_acc*100.0:05.2f}%"
            )
    
    return best_val_acc

#%% Define The Neural Network Architecture

#class PeanutClassifier(nn.Module):
#    def __init__(self):
#        super(PeanutClassifier, self).__init__()
#        self.fc1 = nn.Linear(3, 32)  # Input size 3 (length, width, area), output size 32
#        self.bn1 = nn.BatchNorm1d(32)
#        self.fc2 = nn.Linear(32, 32)
#        self.bn2 = nn.BatchNorm1d(32)
#        self.fc3 = nn.Linear(32, 3)  # Output size 3 (number of classes)
        

#    def forward(self, x):
#        x = torch.relu(self.bn1(self.fc1(x)))
#        x = torch.relu(self.bn2(self.fc2(x)))
#        x = self.fc3(x)
#        return x



#%% Training with k-fold cross-validation

# Define the number of folds
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

# Define the loss module
loss_module = nn.CrossEntropyLoss().to(device)

# Initialize best models list
best_models = []  # List to store the best models of each fold
fold_accuracies = []
fold = 0
for train_index, val_index in kf.split(X):
    fold += 1
    print(f"Fold {fold}/{k_folds}")
    
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y[train_index], y[val_index]
    
    # Map class labels to numerical labels
    y_train_fold_mapped = y_train_fold.map(class_mapping)
    y_val_fold_mapped = y_val_fold.map(class_mapping)

    # Convert to tensors
    X_train_fold = torch.tensor(X_train_fold.values, dtype=torch.float32)
    y_train_fold = torch.tensor(y_train_fold_mapped.values, dtype=torch.long)
    X_val_fold = torch.tensor(X_val_fold.values, dtype=torch.float32)
    y_val_fold = torch.tensor(y_val_fold_mapped.values, dtype=torch.long)

    # Create DataLoader for batch processing
    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
    train_loader_fold = DataLoader(train_dataset_fold, batch_size=20, shuffle=True)
    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=1, shuffle=False)

    # Create a new model for each fold
    peanut_model_fold = PeanutClassifier().to(device)
    optimizer_fold = torch.optim.Adam(peanut_model_fold.parameters(), lr=0.0005, betas=(0.2, 0.9))
    
    # Train and validate the model for the current fold
    best_val_acc_fold = train_and_validate_fold(peanut_model_fold, optimizer_fold, loss_module, train_loader_fold, val_loader_fold, num_epochs=500)
    fold_accuracies.append(best_val_acc_fold)
    
    # Save the parameters of the best model
    best_model_path = os.path.join(CHECKPOINT_PATH, f"best_model_fold_{fold}.pt")
    save_model(peanut_model_fold, f"best_model_fold_{fold}.pt", CHECKPOINT_PATH)
    best_models.append(best_model_path)


best_model_path = None
best_val_acc = 0.0

# Find the best model from the list of best models
for model_path in best_models:
    model = PeanutClassifier().to(device)
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate the model's performance
    val_acc = test_model(model, val_loader)
    print(f"Test Accuracy: {val_acc}")
    
    # Update the best model if this model has higher validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = model_path

# Load the best model
final_model = PeanutClassifier().to(device)
final_model.load_state_dict(torch.load(best_model_path))

    

# Create the final model with averaged parameters
#final_model = PeanutClassifier().to(device)
#final_model.load_state_dict(avg_params)
final_model_path = os.path.join(CHECKPOINT_PATH, "final_model.pt")
save_model(final_model, final_model_path)

#save_model(model, "Final_Model", CHECKPOINT_PATH)
