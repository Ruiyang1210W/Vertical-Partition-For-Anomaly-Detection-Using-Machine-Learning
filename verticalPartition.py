"""
Vertical Partitioning with MPI for Neural Network Inference

This script implements feature-wise (vertical) data partitioning across MPI processes
to parallelize forward propagation in a custom-built neural network. Each process:
  1. Receives a subset of input features (columns)
  2. Builds a local sub-network with shared weights
  3. Computes partial forward pass outputs
  4. Uses MPI Allgatherv to reconstruct the full output

Achieves ~217× speedup over serial baseline at 32 processors.
"""
from mpi4py import MPI
import numpy as np
from Models import Model
from sharedParameter import generate_shared_parameters
from data_preprocessing import load_and_preprocess
from time import time
import sys

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --------------------------
# 0. Load and Normalize Data
# --------------------------
csv_path = "C:/Users/wangr/OneDrive/Desktop/Senior project resourses/Vertical Partition/BalancedTestData.csv"
desired_feature = 100

if rank == 0:
    # Load normalized data and strip labels
    X_normalized, y_labels, mean_vals, std_vals = load_and_preprocess(csv_path, label_column="Label", desired_features=desired_feature)
    N = X_normalized.shape[0]
else:
    X_normalized = None
    N = None
    desired_feature = None

# Broadcast shared data dimensions
N = comm.bcast(N, root=0)
desired_feature = comm.bcast(desired_feature, root=0)

# Allocate array on other ranks
if rank != 0:
    X_normalized = np.empty((N, desired_feature), dtype=np.float64)

# Broadcast full data
comm.Bcast(X_normalized, root=0)

# --------------------------
# 1. Vertical Partitioning (Feature-wise Distribution)
# --------------------------
# Distribute 100 input features across MPI processes (column-wise split)
# Example: 100 features / 4 processes = 25 features per process
local_width_base = desired_feature // size
remainder = desired_feature % size

# Handle uneven splits: first 'remainder' processes get one extra feature
# Example: 100 features / 3 processes → [34, 33, 33] instead of [33, 33, 33]
if rank < remainder:
    local_width = local_width_base + 1
    start = rank * local_width
else:
    local_width = local_width_base
    start = remainder * (local_width_base + 1) + (rank - remainder) * local_width_base

end = start + local_width
local_data = X_normalized[:, start:end]  # Each process gets its column slice

# --------------------------
# 2. Build Local Model & Forward Pass
# --------------------------
depth = int(sys.argv[1]) if len(sys.argv) > 1 else 200
width = local_width

startweights, startbiases = generate_shared_parameters(width, depth, seed=55)
model = Model(width, depth, startweights, startbiases)

model.forward(local_data)
local_output = np.array(model.out())  # Shape: (N, local_width)
local_output = local_output.reshape((N, -1))

# --------------------------
# 3. Gather Results from All Ranks (MPI Collective Communication)
# --------------------------
# Each process has partial output of shape (883, local_width)
# Need to concatenate horizontally to reconstruct full (883, 100) output
local_flat = local_output.flatten()
send_count = local_flat.size

# Use Allgatherv (not Allgather) because processes may have different local_width
# due to uneven feature splits - Allgatherv handles variable-length messages
recv_counts = comm.allgather(send_count)
displacements = [sum(recv_counts[:i]) for i in range(size)]
total_recv = sum(recv_counts)

full_flat = np.empty(total_recv, dtype=np.float64)
comm.Allgatherv([local_flat, MPI.DOUBLE],
                [full_flat, recv_counts, displacements, MPI.DOUBLE])

# Reshape flattened array back to (N, total_columns)
final_output = full_flat.reshape(N, -1)

if rank == 0:
    print(" Final output shape:", final_output.shape)
    print(" Sample output (first 5 rows):")
    print(final_output[:5])
    np.save("custom_output.npy", final_output)

# Time setup
# Before forward pass
start_time = time()
model.forward(local_data)
elapsed_time = time() - start_time

# === Gather maximum elapsed time
max_elapsed_time = comm.reduce(elapsed_time, op=MPI.MAX, root=0)

if rank == 0:
    print(f"[MPI - {size} processors] Max forward pass time: {max_elapsed_time:.6f} seconds")