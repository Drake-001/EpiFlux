

a = [1.0, 2.0, 1.0]
a = [0]

# %%
# import library
import torch 



a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a

# %%
points = torch.zeros(10)
points

# %%
points = torch.zeros(10,5)
points

# %%
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points

# %%
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# prints the Y-coordinate of the zeroth point in dataset
points[0,1]

# %%
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# prints the X-coordinate of the second point in dataset
points[2,0]

# %%
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# prints Y-coordinate of zeroth point AND Y-coordinate of first point
print([points[0,1], points[1,1]])

# %%
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# prints zeroth point in dataset
points[0]

# %%
some_list = list(range(6))
# from 1 inclusive to 4 exclusive, in steps of 2 
some_list[1:4]
# output:
# [1, 4]


# %%
some_list = list(range(6))
# from 1 inclusive to 4 exclusive, in steps of 2 
some_list[1:4:2]
# output:
# [1, 4]

# %%
import torch
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
x_data

# %%
import torch
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
x_data[1]

# %%
import torch
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
x_data[1]

# %%
import random

coordinate_pairs = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(50)]
print(coordinate_pairs)


# %%
import torch
data = [(97, 32), (70, 67), (78, 50), (32, 0), (79, 26), (61, 40), (99, 76), (84, 42), (92, 75), (86, 46), (70, 97), (47, 96), (34, 28), (75, 76), (61, 6), (99, 38), (36, 30), (47, 44), (33, 66), (86, 8), (61, 22), (20, 50), (89, 93), (83, 55), (63, 33), (38, 80), (62, 68), (73, 84), (62, 53), (5, 4), (88, 24), (23, 23), (85, 35), (10, 58), (48, 60), (52, 15), (49, 72), (79, 5), (84, 82), (74, 79), (94, 91), (30, 56), (28, 17), (89, 86), (7, 7), (62, 49), (1, 9), (88, 38), (11, 8), (68, 72)]
x_data = torch.tensor(data)
x_data[1:43:11]

# %%
import torch
data = [(97, 32), (70, 67), (78, 50), (32, 0), (79, 26), (61, 40), (99, 76), (84, 42), (92, 75), (86, 46), (70, 97), (47, 96), (34, 28), (75, 76), (61, 6), (99, 38), (36, 30), (47, 44), (33, 66), (86, 8), (61, 22), (20, 50), (89, 93), (83, 55), (63, 33), (38, 80), (62, 68), (73, 84), (62, 53), (5, 4), (88, 24), (23, 23), (85, 35), (10, 58), (48, 60), (52, 15), (49, 72), (79, 5), (84, 82), (74, 79), (94, 91), (30, 56), (28, 17), (89, 86), (7, 7), (62, 49), (1, 9), (88, 38), (11, 8), (68, 72)]
x_data = torch.tensor(data)
x_data[32]

# %%
# We move our tensor to the GPU if available
import torch
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")

# %%





#--------------------------#




# BEGIN EpiFlux

#   IMPORT necessary PyTorch libraries
  
#   DEFINE dataset 
#     This could be a collection of statements, facts, or experiences that the primary network uses to form beliefs
  
#   DEFINE primary_network 
#     This network will take input data and produce some form of belief or knowledge representation
  
#   DEFINE secondary_network
#     This network will take the outputs (beliefs) of the primary network and evaluate/reflect upon them

#   DEFINE training_loop
#     FOR each epoch
#       FOR each data_point in dataset
#         primary_belief = primary_network(data_point)
#         evaluation = secondary_network(primary_belief)
        
#         CALCULATE loss using the evaluation and desired outcomes
#         BACKPROPAGATE through both networks to optimize weights

#   EXECUTE training_loop

#   EVALUATE final beliefs and evaluations
#     Assess how primary network's beliefs have stabilized and what the secondary network's reflections reveal about them

#   DRAW insights 
#     Analyze and interpret results in context of Rorty’s philosophical ideas and implications for AI model interpretability

# END EpiFlux

