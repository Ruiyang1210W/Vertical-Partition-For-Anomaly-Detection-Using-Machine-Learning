import numpy as np
import matplotlib.pyplot as plt

# === Load outputs ===
custom_output = np.load("custom_output.npy")
torch_output = np.load("pytorch_output.npy")  

# === Check shapes ===
print("Shape match:", custom_output.shape == torch_output.shape)

# === Error Metrics ===
abs_diff = np.abs(custom_output - torch_output)
print("Mean Absolute Error:", np.mean(abs_diff))
# print("Max Absolute Error:", np.max(abs_diff))
# print("Are all values close (atol=1e-2)?", np.allclose(custom_output, torch_output, atol=1e-2))

# === Visualization ===
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title("Custom Output (MPI)")
plt.hist(custom_output.flatten(), bins=50, alpha=0.7, label='Custom')
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
plt.title("PyTorch Output")
plt.hist(torch_output.flatten(), bins=50, alpha=0.7, label='PyTorch', color='orange')
plt.xlabel("Value")

"""
plt.subplot(1, 3, 3)
plt.title("Absolute Difference")
plt.hist(abs_diff.flatten(), bins=50, alpha=0.7, label='Abs Diff', color='red')
plt.xlabel("Absolute Error")
"""

plt.tight_layout()
plt.show()
