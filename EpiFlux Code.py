# SECTION 1: DATA PROCESSING 

# import data pipline library
import torch
# Dataloader is Pytorch utility to load data in batches
from torch.utils.data import DataLoader, Dataset        # Dataset is Pytorch abstract class to represent dataset
from sklearn.model_selection import train_test_split    # utility from scikit-learn to split databases
import pandas as pd



# SECTION 2: DEFINE NEURAL NET

# import torch module
import torch.nn as nn
# import submodule
import torch.nn.functional as F


# create neural net, inherits nn module 
class EpiNet1(nn.Module):
    
    # constructor initializes NN layers
    def __init__(self, input_size):
      
      # call constructor of parent class/initialize nn.Module
      super(EpiNet1, self).__init__()
      
      # 3 fully connected layers (affine operations: y = Wx + b)
      self.fc1 = nn.linear(input_size, 128)
      self.fc2 = nn.linear(128, 64)
      self.fc3 = nn.linear(64, 1)
      self.sigmoid = nn.sigmoid()
    
    # create forward method, defines forward pass of NN. 
    def forward (self, x):
       x = torch.relu(self.fc1(x))
       x = torch.relu(self.fc2(x))
       x = torch.sigmoid(self.fc3(x))
       return x

Moral_data_model = EpiNet1(100)



# SECTION 3: TRAINING LOOP
    











