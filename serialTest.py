# serialTest.py
import time
import numpy as np
from Models import Model
from sharedParameter import generate_shared_parameters
from data_preprocessing import load_and_preprocess

# === Load Data ===
csv_path = "balancedTestData.csv"
X_np, _, _, _ = load_and_preprocess(csv_path, label_column="Label", desired_features=100)

# === Build Full Serial Model ===
depth = 200
width = 100
startweights, startbiases = generate_shared_parameters(width, depth, seed=55)
model = Model(width, depth, startweights, startbiases)

# === Time Forward Pass ===
start_time = time.time()
model.forward(X_np)
end_time = time.time()

serial_time = end_time - start_time
print(f"[Serial] Forward pass time: {serial_time:.6f} seconds")
