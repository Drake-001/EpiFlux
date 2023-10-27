# import library
import torch

#import torch module
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



