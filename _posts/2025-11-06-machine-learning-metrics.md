---
title: "📗 DSML: Machine Learning, Deep Learning & Data Science Metrics — Comprehensive Revision"
layout: post
description: "Concise, clear, and validated revision notes on Machine Learning, Deep Learning, and Data Science Metrics — structured for beginners and practitioners. Chirpy-ready markdown."
categories: [DSML, Machine-Learning, Metrics]
tags: [DSML, Machine-Learning, Deep-Learning, Metrics, Data-Science, Notes]
author: Kalyan Narayana
date: 2025-11-06 19:30:00 +0530
toc: true
math: true
---

# DSML: Machine Learning, Deep Learning & Data Science Metrics ✅

Comprehensive **revision notes** covering the essentials of **Machine Learning (ML)**, **Deep Learning (DL)**, and **Data Science metrics**, written clearly, concisely, and precisely — ideal for quick review or structured study.

---

## 1. Machine Learning — Core Concepts

### 🔹 Definition
Machine Learning is the study of algorithms that **learn from data** to identify patterns and make predictions or decisions without explicit programming.

### 🔹 Objective
The goal is **generalisation** — achieving strong performance on unseen data, not just the training set.

### 🔹 Workflow
1. Problem definition  
2. Data collection  
3. Preprocessing & feature engineering  
4. Model selection & training  
5. Evaluation & validation  
6. Hyperparameter tuning  
7. Deployment & monitoring  

### 🔹 Key Trade-offs
| Concept | Explanation |
|----------|-------------|
| **Bias–Variance Tradeoff** | High bias → underfitting; high variance → overfitting |
| **Model Complexity vs Data Size** | Complex models require larger datasets to generalise well |

---

## 2. Deep Learning — Core Ideas

### 🔹 Definition
Deep Learning is a subfield of ML that uses **multi-layered neural networks** to automatically learn features from data such as images, text, or sound.

### 🔹 Key Components
- **Layers**: Linear transformations + activations  
- **Loss Function**: Measures prediction error  
- **Optimizer**: Adjusts parameters (SGD, Adam)  
- **Regularisation**: Dropout, weight decay, batch norm  

### 🔹 Common Architectures
| Type | Typical Use |
|------|--------------|
| MLP | Tabular or small-scale data |
| CNN | Image and spatial pattern recognition |
| RNN / LSTM / GRU | Sequential data or time series |
| Transformer | Modern NLP and vision tasks |
| Autoencoder / VAE | Dimensionality reduction, generative models |
| GAN | Synthetic data generation |

### 🔹 Training Flow
1. Normalise inputs and batch data  
2. Choose loss + optimizer  
3. Forward → backward pass  
4. Apply regularisation and early stopping  
5. Monitor loss and metrics  
6. Save the best model checkpoint  

---

## 3. Common Machine Learning Algorithms

| Category | Algorithms | Ideal For |
|-----------|-------------|-----------|
| Linear Models | Linear Regression, Logistic Regression | Simple, interpretable problems |
| Tree-Based | Decision Tree, Random Forest, XGBoost | Tabular data, high accuracy |
| Kernel Methods | SVM | Small/medium datasets |
| Instance-Based | k-NN | Simple baselines |
| Neural Networks | MLP, CNN, RNN, Transformer | Complex or unstructured data |
| Clustering | K-Means, DBSCAN, Hierarchical | Grouping unlabeled data |
| Dim. Reduction | PCA, t-SNE, UMAP | Visualization, noise removal |
| Reinforcement | Q-Learning, PPO | Sequential decision-making |

---

## 4. Data Science Metrics — Essentials

### 🔹 Classification Metrics
Let **TP, FP, TN, FN** represent the standard confusion matrix entries.

- **Accuracy**  
  $$
  Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
  $$
- **Precision**  
  $$
  Precision = \frac{TP}{TP + FP}
  $$
- **Recall (Sensitivity)**  
  $$
  Recall = \frac{TP}{TP + FN}
  $$
- **F1 Score**  
  $$
  F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
  $$
- **Specificity**  
  $$
  Specificity = \frac{TN}{TN + FP}
  $$

Use:
- **Precision** when false positives are costly.  
- **Recall** when false negatives are costly.  
- **F1** when balancing both is important.  

---

### 🔹 Regression Metrics

| Metric | Formula | Interpretation |
|---------|----------|----------------|
| **MSE** | $\text{MSE} = \frac{1}{n}\sum (y - \hat{y})^2$ | Penalises large errors |
| **RMSE** | $\text{RMSE} = \sqrt{\text{MSE}}$ | Same units as target |
| **MAE** | $\text{MAE} = \frac{1}{n}\sum |y - \hat{y}|$ | Robust to outliers |
| **R² (Coefficient of Determination)** | $R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | Fraction of variance explained |

---

### 🔹 Ranking & Recommendation Metrics
| Metric | Description |
|---------|-------------|
| **Precision@k** | Relevant items in top-k recommendations |
| **Recall@k** | Fraction of total relevant items found |
| **MAP** | Mean average precision across queries |
| **NDCG** | Rank-aware relevance metric |

---

## 5. Practical Python Snippets

### 🧩 Logistic Regression (Scikit-Learn)
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
````

### 🧠 Minimal PyTorch Network

```python
import torch
from torch import nn

class SimpleNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.layers(x)

model = SimpleNet(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

---

## 6. Best Practices & Common Pitfalls

| ✅ Best Practices                                 | ⚠️ Common Pitfalls                           |
| ------------------------------------------------ | -------------------------------------------- |
| Start simple, add complexity only when justified | Data leakage (using target info in features) |
| Use proper cross-validation                      | Ignoring feature scaling for ML models       |
| Track experiments & reproducibility              | Overfitting small datasets                   |
| Align metrics with business goals                | Over-reliance on accuracy                    |
| Monitor deployed model drift                     | Poor validation strategy (esp. time series)  |

---

## 7. Quick Comparison Tables

### 🔸 ML vs DL

| Aspect              | Machine Learning            | Deep Learning                   |
| ------------------- | --------------------------- | ------------------------------- |
| Data Requirement    | Low–Medium                  | High                            |
| Feature Engineering | Manual                      | Automatic                       |
| Interpretability    | High                        | Low                             |
| Compute Demand      | Low                         | High                            |
| Use Cases           | Tabular data, small samples | Images, text, large-scale tasks |

---

## 8. Deployment Checklist

* ✅ Save trained model artifact with metadata
* ✅ Validate input schema & pipeline consistency
* ✅ Monitor performance & data drift
* ✅ Automate retraining or alerts
* ✅ Secure access and governance

---

## 9. Analogy & Plain Explanation

* **Machine Learning**: like teaching a student with examples and grading them.
* **Deep Learning**: like a student discovering sub-skills automatically (edges → shapes → objects).
* **Metrics**: your grading system — tells you if the student is learning correctly.

---

## 10. Quick Upgrade Path

1. Study **transformers** and **attention mechanisms**
2. Learn **MLOps**: CI/CD, feature stores, versioning
3. Explore **Explainable AI** (SHAP, LIME)
4. Implement **ethical AI** and bias mitigation
5. Practice with **real projects** — Kaggle, open datasets

---

### 📚 References (clickable)

1. <a href="https://en.wikipedia.org/wiki/Machine_learning" target="_blank" rel="noopener">Machine Learning — Wikipedia</a>
2. <a href="https://www.geeksforgeeks.org/machine-learning/introduction-machine-learning/" target="_blank" rel="noopener">Introduction to Machine Learning — GeeksforGeeks</a>
3. <a href="https://www.geeksforgeeks.org/machine-learning/machine-learning/" target="_blank" rel="noopener">Machine Learning Overview — GeeksforGeeks</a>
4. <a href="https://www.ibm.com/think/topics/machine-learning" target="_blank" rel="noopener">What Is Machine Learning (ML)? — IBM</a>
5. <a href="https://www.ibm.com/think/topics/machine-learning-algorithms" target="_blank" rel="noopener">Machine Learning Algorithms — IBM</a>
6. <a href="https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained" target="_blank" rel="noopener">Machine Learning Explained — MIT Sloan</a>
