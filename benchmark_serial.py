# benchmark_serial.py
# Comprehensive benchmarking script for serial model performance
import time
import numpy as np
from Models import Model
from sharedParameter import generate_shared_parameters
from data_preprocessing import load_and_preprocess

# === Load Full Dataset ===
csv_path = "balancedTestData.csv"
X_np, _, _, _ = load_and_preprocess(csv_path, label_column="Label", desired_features=100)

# === Build Full Serial Model ===
depth = 200
width = 100
startweights, startbiases = generate_shared_parameters(width, depth, seed=55)
model = Model(width, depth, startweights, startbiases)

# === Sample sizes to test ===
sample_sizes = [5, 10, 100, 500, 881]  # 881 is the full dataset
num_iterations = 5  # Run each test 5 times for reliability

print("=" * 70)
print("SERIAL MODEL BENCHMARK - Multiple Sample Sizes")
print("=" * 70)
print(f"Model: depth={depth}, width={width}")
print(f"Iterations per test: {num_iterations}")
print("=" * 70)
print()

results = []

for n in sample_sizes:
    if n > len(X_np):
        print(f"Warning: Sample size {n} exceeds dataset size {len(X_np)}. Skipping.")
        continue

    # Get subset of data
    X_subset = X_np[:n]

    # Warm-up run (not counted)
    model.forward(X_subset)

    # Timed runs
    times = []
    for i in range(num_iterations):
        start_time = time.time()
        model.forward(X_subset)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)

    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    results.append({
        'n': n,
        'avg': avg_time,
        'std': std_time,
        'min': min_time,
        'max': max_time
    })

    print(f"Samples: {n:4d} | Avg: {avg_time:.6f}s | Std: {std_time:.6f}s | Min: {min_time:.6f}s | Max: {max_time:.6f}s")

print()
print("=" * 70)
print("SUMMARY TABLE FOR README")
print("=" * 70)
print()
print("| Samples (n) | Serial Avg (s) | Std Dev (s) |")
print("|-------------|----------------|-------------|")
for r in results:
    print(f"| {r['n']:<11d} | {r['avg']:<14.6f} | {r['std']:<11.6f} |")

print()
print("=" * 70)
print("NOTES:")
print("- Use 'Avg' column for README comparison")
print(f"- Each timing averaged over {num_iterations} runs")
print("- First run was warm-up (not counted)")
print("- Std Dev shows timing variance")
print("=" * 70)
