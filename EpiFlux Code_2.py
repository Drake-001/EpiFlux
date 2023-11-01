### Project Description ###
# MY EXPLANATION
# This project creates multiple neural network model(s) that is fed ethical propositions AND raw data that from a human's 
# evluation might prove contradictory to the ethical claims. Thus the project is largely at the intersection of 
# formal epistemology and formal ethics, because it requires the machine to make "beliefs" (i.e., output) based off
# ethical propositions. 


# TO-DO
# copy outline from ChatGPT chat "Final Pytorch Project," should be the last outline/conversation
# gather ethical information / look more into the MIT The Moral Machine 
#### IMPORTANT ####
#    Bayesian probability to assign probabilities to the beliefs of the neural network for complex information, e.g., given 
#    propositions x, y, & z, what is the liklihood that humans need to be eliminated from earth, etc.   














# OFFICIAL STRUCTURE 

# Starting with the practical experiment from the subproject, we can first focus on 
# creating a simple neural network that learns from a dataset of moral or 
# epistemological statements. Here's a basic outline of the coding process and the 
# starting code to get you going:

# Data prep: 
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv("path_to_your_data.csv")

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2)

# Convert sentences to embeddings (this is a simple placeholder, more advanced techniques like word embeddings should be used)
def statement_to_vector(statement):
    return torch.tensor([ord(c) for c in statement[:100]]).float()

class MoralDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        statement = self.data.iloc[idx]['statement']
        label = self.data.iloc[idx]['label']
        return statement_to_vector(statement), torch.tensor(label).float()

train_loader = DataLoader(MoralDataset(train_data), batch_size=32, shuffle=True)
test_loader = DataLoader(MoralDataset(test_data), batch_size=32)


# NN Model
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = SimpleNN(100)  # Assuming each statement is truncated/padded to 100 characters



# Training loop
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for statements, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(statements)
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")








