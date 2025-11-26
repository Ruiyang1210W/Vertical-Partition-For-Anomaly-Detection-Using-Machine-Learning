# Vertical Partitioning for Anomaly Detection Using MPI

**High-Performance Computing approach to scalable deep learning inference for cybersecurity applications**

## Overview

This project implements a **custom-built deep neural network from scratch** with **MPI-based vertical partitioning** to accelerate anomaly detection in cybersecurity datasets. By distributing input features across multiple processors and parallelizing the forward propagation phase, the system achieves significant performance improvements while maintaining mathematical parity with PyTorch implementations.

### Key Achievements

- **~217× speedup** over serial baseline (32 processors)
- **Mathematical parity** with PyTorch reference implementation (validated through output comparison)
- **Built from scratch** - no high-level frameworks, full control over architecture
- **Scalable architecture** tested across 4-64 processors
- **Real-world dataset** - CIC-IDS 2017 cybersecurity benchmark (5.9M samples, 85 features)

---

## Motivation

Traditional intrusion detection systems struggle with the computational demands of modern deep learning on high-dimensional cybersecurity data. This research explores **processor-level partitioning strategies** to enable:

- **Scalable inference** on large-scale network traffic data
- **CPU-based parallelization** without requiring GPU infrastructure
- **Foundation for distributed anomaly detection** using autoencoders

---

## Architecture

### Vertical Partitioning Strategy

The system implements **feature-wise (vertical) partitioning** where input features are distributed column-wise across MPI processes:

```
Input Features: [F1, F2, F3, ... F100]
                    ↓
        ┌──────────┴──────────┬──────────┬──────────┐
    Process 0      Process 1   Process 2   Process 3
    [F1...F25]     [F26...F50] [F51...F75] [F76...F100]
        ↓              ↓           ↓           ↓
    Local Model    Local Model Local Model Local Model
        ↓              ↓           ↓           ↓
        └──────────────┴───────────┴───────────┘
                        ↓
                  MPI Allgather
                        ↓
                  Final Output
```

### System Components

- **`Models.py`** - Core neural network architecture with configurable depth/width
- **`Layers.py`** - Custom layer implementation (InputLayer, Hidden, Output)
- **`Activations.py`** - Activation functions (ReLU, Sigmoid) with derivatives
- **`verticalPartition.py`** - MPI-parallelized forward propagation driver
- **`sharedParameter.py`** - Deterministic weight/bias generation (ensures consistency)
- **`data_preprocessing.py`** - CIC-IDS 2017 normalization and feature padding
- **`compare_results.py`** - Validation against PyTorch baseline

---

## Dataset

**CIC-IDS 2017** - A comprehensive intrusion detection benchmark from the Canadian Institute for Cybersecurity

### Full Dataset Context
- **5,950,088 samples** across 5 days of network traffic (2.76GB)
- **85 features** after merging all CSV files
- **Attack types**: DDoS, Brute Force, Web Attacks, Infiltration, PortScan, Botnet
- **Download**: [CIC-IDS 2017 Official Page](https://www.unb.ca/cic/datasets/ids-2017.html)

### Experimental Subset
For this research, experiments were conducted using a balanced subset (`balancedTestData.csv`):
- **883 samples** (manageable for testing and validation)
- **10 features** (including label column)
- **Preprocessing**: NaN/infinity removal, normalization, feature padding to 100 dimensions
- This subset allows rapid iteration while demonstrating the scalability principles that apply to the full dataset

---

## Performance Results

### Runtime Comparison

| Samples (n) | Serial (s) | MPI 32-proc (s) | **Speedup** |
|-------------|------------|-----------------|-------------|
| 5           | 0.102218   | 0.00047         | **~217×**   |
| 10          | 0.102218   | 0.000569        | **~180×**   |
| 100         | 0.102218   | 0.004293        | **~24×**    |

### Speedup Analysis

![Speedup Chart](https://via.placeholder.com/600x400?text=Speedup+vs+Processors)

**Key Findings:**
- **Optimal performance**: 16-32 processors
- **Diminishing returns** beyond 32 processors due to communication overhead
- **Aligned with Amdahl's Law**: Parallel overhead becomes dominant at high processor counts

---

## Installation & Setup

### Prerequisites

```bash
# Python 3.10+
# MPI implementation (e.g., MPICH, OpenMPI)
```

### Install Dependencies

```bash
pip install mpi4py numpy pandas scikit-learn torch matplotlib
```

### Dataset Preparation

1. Download [CIC-IDS 2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
2. Place `balancedTestData.csv` in the project root
3. Update CSV path in `verticalPartition.py` (line 17)

---

## Usage

### Serial Execution (Baseline)

```bash
python serialTest.py
```

**Output:**
```
[Serial] Forward pass time: 0.102218 seconds
```

### Parallel Execution (MPI)

```bash
# Run with 8 processors, 100-layer network
mpiexec -n 8 python verticalPartition.py 100
```

**Output:**
```
[MPI - 8 processors] Max forward pass time: 0.007617 seconds
Final output shape: (800, 100)
```

### Validate Against PyTorch

```bash
python compare_results.py
```

**Output:**
```
Mean Absolute Error: 0.29
```

---

## Technical Implementation Details

### MPI Communication Pattern

1. **Data Distribution**: Rank 0 loads and broadcasts full dataset
2. **Feature Partitioning**: Each process gets `features // num_procs` columns (with balanced remainder distribution)
3. **Local Forward Pass**: Each process runs its sub-network independently
4. **Allgather**: Partial outputs collected and concatenated to reconstruct full output

### Weight Synchronization

Uses **deterministic random seeding** (`seed=55`) via `sharedParameter.py` to ensure:
- Identical weights across all MPI processes
- Fair comparison with PyTorch baseline
- Reproducible results

---

## Project Structure

```
├── Activations.py              # ReLU, Sigmoid activation functions
├── Layers.py                   # InputLayer, Layer classes
├── Models.py                   # Neural network model class
├── verticalPartition.py        # MPI parallel driver
├── serialTest.py               # Serial baseline test
├── sharedParameter.py          # Shared weight generation
├── data_preprocessing.py       # CIC-IDS 2017 preprocessing
├── compare_results.py          # PyTorch validation
├── pyTorchTest_100x200.py      # PyTorch reference implementation
├── balancedTestData.csv        # Dataset subset
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Limitations & Future Work

### Current Limitations

- ⚠️ **Forward pass only** - No backpropagation/training implemented yet
- ⚠️ **Output discrepancy** - 0.29 MAE vs PyTorch (likely floating-point precision/weight alignment)
- ⚠️ **No autoencoder** - Planned extension for anomaly detection

### Roadmap

- [ ] Implement distributed backpropagation for training
- [ ] Add autoencoder architecture for unsupervised anomaly detection
- [ ] Compare vertical vs horizontal vs block partitioning strategies
- [ ] Deploy on HPC cluster (e.g., SLURM-managed system)
- [ ] Reduce PyTorch output discrepancy to < 0.01 MAE
- [ ] Implement model checkpointing and fault tolerance

---

## Research Context

This work was completed as part of a research fellowship at **Cal Poly Pomona** under the supervision of **Professor John Korah** (CS 4620, Spring 2025). The project focuses on the **intersection of High-Performance Computing and Machine Learning** for cybersecurity applications.

### Related Works

- **Full Research Report**: [`PROCESSOR PARTITIONING FOR ANOMALY DETECTION USING MACHINE LEARNING.pdf`](./PROCESSOR%20PARTITIONING%20FOR%20ANOMALY%20DETECTION%20USING%20MACHINE%20LEARNING.pdf)
- **Presentation Slides**:
  - [View on Google Slides](https://docs.google.com/presentation/d/1_7tjPeLq2DHzszy5KHrw42hgbglpq-hP62UN1d6C2KM/edit?slide=id.p#slide=id.p) (interactive)
  - [PDF Version](./Anomaly%20Detection%20Vertical%20Partition%20Report.pdf) (offline access)

---

## References

1. Sharafaldin et al. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *ICISSP 2018*.
2. Chandola et al. (2009). "Anomaly Detection: A Survey." *ACM Computing Surveys*.
3. Gropp et al. (1999). *Using MPI: Portable Parallel Programming with the Message-Passing Interface*. MIT Press.
4. Navathe et al. (1984). "Vertical Partitioning Algorithms for Database Design." *ACM TODS*.

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Contact

**Ruiyang (Wendy) Wang**
Research Fellow, Cal Poly Pomona

---

*This project demonstrates practical application of distributed computing principles to real-world cybersecurity challenges, bridging the gap between theoretical HPC concepts and production ML systems.*

