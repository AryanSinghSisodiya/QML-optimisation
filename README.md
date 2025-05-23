# ⚛️ Quantum Machine Learning Models (BTP – IIT Patna)

This repository contains implementations of Three quantum machine learning models developed as part of my B.Tech Final Year Project at IIT Patna. The goal is to explore quantum-enhanced learning using **Quantum Support Vector Machines (QSVM)** and **Quantum Neural Networks (QNN)** for binary classification tasks.

---

## 📁 Files

- `VQC and QSVM.ipynb` – Implementation of a Quantum SVM classifier using Qiskit Machine Learning.
- `QCL_model_btp.ipynb` – Implementation of a quantum neural network using a hybrid quantum-classical approach.

---

## 🧠 Project Motivation

Classical machine learning methods often struggle with complex feature spaces or require significant computational resources. Quantum computing offers a potential paradigm shift by enabling kernel-based methods and neural networks to operate in exponentially large Hilbert spaces, with potentially fewer parameters.

This work draws inspiration from foundational papers in quantum ML:

> - Havlíček et al. (2019): Supervised learning with quantum-enhanced feature spaces  
> - Abbas et al. (2021): The power of quantum neural networks  
> - Wu et al. (2021): Event Classification with Quantum Machine Learning in High-Energy Physics

---

## 📊 Dataset

Both notebooks use the [SUSY and HIGGS dataset](https://archive.ics.uci.edu/ml/datasets/SUSY), a binary classification dataset commonly used in high-energy physics. The data is normalized and split into training and test sets.

---

## 🔬 1. Quantum Support Vector Machine – `QSVM.ipynb`

### 🔧 Approach

- Uses the `QuantumKernel` class from `qiskit_machine_learning.kernels`.
- A `ZZFeatureMap` encodes classical features into a quantum state.
- The kernel matrix is computed using inner products of quantum states.
- A classical SVM is trained using this quantum kernel.

### 📈 Evaluation

- Accuracy
- Confusion matrix
- Quantum circuit visualization

---

## 🧠 2. Quantum Circuit Learning – `QCL_model_btp.ipynb`

### 🏗 Model Overview

- Implements a **Quantum Neural Network (QNN)** using Qiskit's `EstimatorQNN`.
- Circuit is built from:
  - **Encoding circuit** (feature map)
  - **Variational circuit** (parameterized rotation gates)
- Integrated into PyTorch using `TorchConnector`.

### ⚙ Training Details

- Loss Function: Mean Squared Error (MSE)
- Optimizer: L-BFGS (PyTorch)
- Backend: Qiskit Aer Simulator

### 📊 Evaluation Metrics

- Accuracy: ~0.764 (For SUSY) ~0.96 (for HIGGS)
- ROC-AUC Score
- F1 Score, Precision, Recall
- Confusion Matrix
- Output Distribution (Plotted)
- Intermediate raw and mapped outputs of the QNN

---

## 📚 Key Results

| Model | Accuracy | ROC-AUC | Highlights |
|-------|----------|---------|------------|
| QSVM  | *depends on kernel size* | Kernel heatmap shown | Quantum kernel-based classifier |
| QCL   | 96.5%     | Visualized | End-to-end trainable QNN |

---

## 📚 References

1. **Havlíček, V., Córcoles, A. D., Temme, K., et al. (2019).**  
   *Supervised learning with quantum-enhanced feature spaces*.  
   Nature Physics, 15(6), 633–640. [DOI:10.1038/s41567-019-0648-8](https://doi.org/10.1038/s41567-019-0648-8)

2. **Abbas, A., Sutter, D., Zoufal, C., et al. (2021).**  
   *The power of quantum neural networks*.  
   Nature Computational Science, 1, 403–409. [DOI:10.1038/s43588-021-00084-1](https://doi.org/10.1038/s43588-021-00084-1)

3. **Wu, J., Koh, D., Lidar, D. A., & Nachman, B. (2021).**  
   *Event Classification with Quantum Machine Learning in High-Energy Physics*.  
   Physical Review D, 104(5), 052003. [arXiv:2105.12256](https://arxiv.org/abs/2105.12256)

---

