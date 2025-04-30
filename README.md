# Vertical-Partition-For-Anomaly-Detection-Using-Machine-Learning

- Implement a deep neural network from scratch (forward pass only)
- Apply **vertical partitioning** using MPI (feature-wise splitting)
- Measure performance across various processor counts
- Compare with a serial version to calculate **speedup**
- Analyze scalability trends and connection to **Amdahl’s Law**

- ## Dataset
- **CIC-IDS 2017** (subset)
- 800+ rows and 10 features (including label)
- Used to simulate realistic input for performance testing

- ## 📈 Results Summary

- Achieved up to **5.5× speedup** using 8 processors
- Best performance observed at **16–32 processors**
- Results aligned with **Amdahl’s Law**: more processors helped until communication overhead increased

## Future Work
- Improve custom model accuracy by aligning outputs with PyTorch
- Compare vertical partitioning with teammates’ horizontal/block partitioning
- Extend the model for training and anomaly detection
- Deploy on HPC cluster for large-scale scalability testing

## How to Run
### Serial test:
python serialTest.py
### Parallel run (example with 8 processors):
mpiexec -n 8 python verticalPartition.py 100

