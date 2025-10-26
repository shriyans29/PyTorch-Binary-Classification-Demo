# 🌀 PyTorch Circle Binary Classifier

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)

This project demonstrates **binary classification** on non-linear circular data using **PyTorch**.  
It shows how deep neural networks with **ReLU activations** can learn complex decision boundaries.

---

## 📖 Overview

We generate a **toy dataset** of circles using `make_circles()` from **Scikit-learn**, where:

- Class `0` = inner circle  
- Class `1` = outer circle  

A simple linear model fails to separate the classes, but a multi-layer model (`CircleModelV2`) with non-linear activation can classify them effectively.

---

## ⚙️ Features

- 🔹 Binary classification on circular data  
- 🔹 Device-agnostic code: CPU/GPU support  
- 🔹 Multi-layer neural network with **ReLU**  
- 🔹 Decision boundary visualization  
- 🔹 Training & evaluation with **accuracy tracking**  
- 🔹 Model saving/loading support  

---

## 🧠 Model Architecture

### CircleModelV2
```python
nn.Linear(2, 10) → ReLU
nn.Linear(10, 10) → ReLU
nn.Linear(10, 10) → ReLU
nn.Linear(10, 1)
Activation: ReLU

Loss Function: BCEWithLogitsLoss

Optimizer: SGD (lr=0.1)

Output: Single neuron for binary classification

🧩 Requirements
Install dependencies:

bash
Copy code
pip install torch numpy matplotlib scikit-learn pandas requests
▶️ Usage
Train the Model:
bash
Copy code
python circle_classification.py
Visualize Decision Boundary:
Decision boundaries show how the model separates class 0 and class 1.

📊 Results
Linear model → ~50% accuracy

Multi-layer ReLU model → ~95%+ accuracy

Non-linear activation crucial for circular separation

💾 Saving and Loading
Models can be saved with torch.save(model.state_dict(), PATH)
Load models with:

python
Copy code
model.load_state_dict(torch.load(PATH))
model.eval()
