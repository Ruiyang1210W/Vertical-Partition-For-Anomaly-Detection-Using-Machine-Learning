import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sharedParameter import generate_shared_parameters
from data_preprocessing import load_and_preprocess

# === Setup ===
torch.manual_seed(55)
np.random.seed(55)

csv_path = "C:/Users/wangr/OneDrive/Desktop/Senior project resourses/Vertical Partition/BalancedTestData.csv"
desired_feature = 100
depth = 200
num_ranks = 32  # Must match MPI run (e.g., mpiexec -n 4)

# === Load and preprocess input ===
X_np, y_labels, mean_vals, std_vals = load_and_preprocess(csv_path, label_column="Label", desired_features=desired_feature)
N = X_np.shape[0]
print("Input shape after preprocessing:", X_np.shape)

# === Load shared parameters ===
shared_weights, shared_biases = generate_shared_parameters(desired_feature, depth, seed=55)

# === Partition Simulation ===
partition_outputs = []

for rank in range(num_ranks):
    base = desired_feature // num_ranks
    rem = desired_feature % num_ranks

    if rank < rem:
        local_width = base + 1
        start = rank * local_width
    else:
        local_width = base
        start = rem * (base + 1) + (rank - rem) * base
    end = start + local_width

    X_part = X_np[:, start:end]
    weights_part = [w[start:end, start:end] for w in shared_weights]
    biases_part = [b[start:end] for b in shared_biases]

    # Build a matching submodel
    class SubModel(nn.Module):
        def __init__(self, width, depth, weights, biases):
            super().__init__()
            layers = []
            for i in range(depth - 1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(width, width))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)

            with torch.no_grad():
                linear_layers = [l for l in self.network if isinstance(l, nn.Linear)]
                for i, layer in enumerate(linear_layers):
                    layer.weight.copy_(torch.tensor(weights[i].T, dtype=torch.float64))
                    layer.bias.copy_(torch.tensor(biases[i], dtype=torch.float64))

        def forward(self, x):
            return self.network(x)

    model_part = SubModel(width=local_width, depth=depth, weights=weights_part, biases=biases_part).double()
    output_part = model_part(torch.tensor(X_part, dtype=torch.float64)).detach().numpy()
    partition_outputs.append(output_part)

# === Combine all partition outputs ===
torch_vpartition_output = np.concatenate(partition_outputs, axis=1)
print(torch_vpartition_output[:5])
np.save("pytorch_output.npy", torch_vpartition_output)



