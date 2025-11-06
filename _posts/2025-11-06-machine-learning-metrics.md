---
title: "📗 DSML: Machine Learning Metrics"
layout: post
description: "Concise, clear, and validated revision notes on Machine Learning, Deep Learning, and Data Science Metrics — structured for beginners and practitioners."
categories: [Notes, DSML Metrics]
tags: [DSML, Machine-Learning, Deep-Learning, Metrics, Data-Science, Notes]
author: Kalyan Narayana
date: 2025-11-06 19:30:00 +0530
toc: true
math: true
---

# DSML: Machine Learning Metrics ✅

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
| **MAE** | $\text{MAE} = \frac{1}{n}\sum \|y - \hat{y}\|$ | Robust to outliers |
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

# 📊 DSML Metrics — Comprehensive Guide ✅

Understanding metrics is at the **heart of Data Science and Machine Learning**.  
Metrics tell you *how well your model is performing* — whether it predicts correctly, generalizes well, and aligns with the real-world objective.

This guide explains key metrics for:
- **Classification**
- **Regression**
- **Clustering**
- **Ranking / Recommendation**
- **Time Series / Forecasting**

Each metric includes:
- 💡 A beginner-friendly **description**
- 🧮 The **mathematical formula**
- 🧩 A **Python method or code snippet**

---

## 🧠 1. Classification Metrics

Used when the model predicts discrete labels (e.g., *spam / not spam*, *disease / no disease*).

| Metric | Description | Formula | Python Method |
|:--|:--|:--|:--|
| **Accuracy** | Measures the overall percentage of correct predictions. Best for balanced datasets. | $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$ | `accuracy_score(y_true, y_pred)` |
| **Precision** | Out of all predicted positives, how many are actually positive. Useful when false positives are costly. | $$Precision = \frac{TP}{TP + FP}$$ | `precision_score(y_true, y_pred)` |
| **Recall (Sensitivity / TPR)** | Out of all actual positives, how many did we correctly predict. Useful when missing positives is costly. | $$Recall = \frac{TP}{TP + FN}$$ | `recall_score(y_true, y_pred)` |
| **F1-Score** | Harmonic mean of precision and recall — balances both. | $$F_1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$ | `f1_score(y_true, y_pred)` |
| **Specificity (TNR)** | Out of all actual negatives, how many were correctly predicted. | $$Specificity = \frac{TN}{TN + FP}$$ | via `confusion_matrix` |
| **ROC-AUC** | Measures the model’s ability to distinguish classes. Closer to 1 = better. | *Area under ROC curve* | `roc_auc_score(y_true, y_prob)` |
| **PR-AUC** | Precision–Recall tradeoff, ideal for imbalanced datasets. | *Area under PR curve* | `average_precision_score(y_true, y_prob)` |
| **Log Loss** | Penalizes incorrect probabilities heavily. Ideal for probabilistic classifiers. | $$L = -\frac{1}{n}\sum [y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$ | `log_loss(y_true, y_prob)` |

🧩 **Example**
```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec  = recall_score(y_true, y_pred)
f1   = f1_score(y_true, y_pred)
````

---

## 📈 2. Regression Metrics

Used when predicting continuous values (e.g., house price, temperature, revenue).

| Metric                                    | Description                                                                                              | Formula                                                             | Python Method                                       |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------ | :-------------------------------------------------- |
| **MAE (Mean Absolute Error)**             | Average absolute difference between predicted and actual values. Easy to interpret.                      | $$MAE = \frac{1}{n}\sum \| y_i - \hat{y}_i \| $$  | `mean_absolute_error(y_true, y_pred)`           |
| **MSE (Mean Squared Error)**              | Squares errors, penalizing large deviations. Sensitive to outliers.                                      | $$MSE = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$$                       | `mean_squared_error(y_true, y_pred)`          |
| **RMSE (Root Mean Squared Error)**        | Square root of MSE, same unit as the target.                                                             | $$RMSE = \sqrt{MSE}$$                                               | `mean_squared_error(y_true, y_pred, squared=False)` |
| **R² (Coefficient of Determination)**     | Proportion of variance in target explained by the model (1 = perfect).                                   | $$R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2}$$ | `r2_score(y_true, y_pred)`                          |
| **MAPE (Mean Absolute Percentage Error)** | Measures average percentage difference between prediction and actual. Easy to explain to non-tech users. | $$MAPE = \frac{100}{n}\sum \left \| \frac{y_i - \hat{y}_i}{y_i}\right \| $$  | `np.mean(np.abs((y_true - y_pred)/y_true))*100` |
| **SMAPE (Symmetric MAPE)**                | Handles zeros better by averaging actuals & predictions in denominator.                                  | $$SMAPE = \frac{100}{n}\sum \frac{ \| y_i - \hat{y}_i \| }{( \| y_i \| + \| \hat{y}_i \| )/2}$$ | `2*np.mean(np.abs(y-y_hat)/(np.abs(y)+np.abs(y_hat)))*100` |

🧩 **Example**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2   = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

---

## 📦 3. Clustering Metrics (Unsupervised)

Used when no labels exist; these measure the **compactness and separation** of clusters.

| Metric                      | Description                                                                        | Formula / Idea                                                                             | Python Method                        |
| :-------------------------- | :--------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- | :----------------------------------- |
| **Silhouette Score**        | How close each point is to its own cluster vs. others (−1 to +1). Higher = better. | $$s = \frac{b - a}{\max(a,b)}$$ where *a* = intra-cluster, *b* = nearest cluster distance. | `silhouette_score(X, labels)`        |
| **Calinski-Harabasz Index** | Ratio of between-cluster variance to within-cluster variance. Higher = better.     | $$CH = \frac{Tr(B_k)/(k-1)}{Tr(W_k)/(n-k)}$$                                               | `calinski_harabasz_score(X, labels)` |
| **Davies-Bouldin Index**    | Average similarity between clusters; lower = better.                               | —                                                                                          | `davies_bouldin_score(X, labels)`    |

---

## 🔍 4. Ranking / Recommendation Metrics

Used for recommendation systems, search ranking, or top-k predictions.

| Metric                                           | Description                                                                | Formula / Concept                                                            | Python Method                 |
| :----------------------------------------------- | :------------------------------------------------------------------------- | :--------------------------------------------------------------------------- | :---------------------------- |
| **Precision@k**                                  | Fraction of top-k recommended items that are relevant.                     | $$P@k = \frac{\text{Relevant items in top }k}{k}$$                           | Custom function               |
| **Recall@k**                                     | Fraction of total relevant items retrieved in top-k list.                  | $$R@k = \frac{\text{Relevant items in top }k}{\text{Total relevant items}}$$ | Custom function               |
| **MAP (Mean Average Precision)**                 | Average of precision values across recall levels. Higher = better ranking. | $$MAP = \frac{1}{Q}\sum_q AvgPrecision(q)$$                                  | `average_precision_score()`   |
| **NDCG (Normalized Discounted Cumulative Gain)** | Measures ranking quality considering order and relevance.                  | $$NDCG@k = \frac{DCG@k}{IDCG@k}$$                                            | `ndcg_score(y_true, y_score)` |

🧩 **Example**

```python
from sklearn.metrics import ndcg_score, average_precision_score
ndcg = ndcg_score(y_true, y_score)
map_score = average_precision_score(y_true, y_score)
```

---

## ⏱️ 5. Time Series / Forecasting Metrics

Used when predicting values over time (sales, temperature, stock prices).
Emphasis on **directional accuracy** and **scale-independent errors**.

| Metric    | Description                                                            | Formula                                                                          | Python Method                                      |
| :-------- | :--------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------- |
| **MAE**   | Average of absolute errors; robust and easy to interpret.              | $$MAE = \frac{1}{n}\sum \| y_t - \hat y_t \| $$  | `mean_absolute_error(y_true, y_pred)`           |
| **RMSE**  | Penalises larger errors; sensitive to outliers.                        | $$RMSE = \sqrt{\frac{1}{n}\sum (y_t - \hat y_t)^2}$$  | `mean_squared_error(y_true, y_pred, squared=False)` |
| **MAPE**  | Average percentage error; intuitive but unstable for near-zero values. | $$MAPE = \frac{100}{n}\sum \left \| \frac{y_t - \hat y_t}{y_t}\right \| $$  | `np.mean(np.abs((y_true - y_pred)/y_true))*100` |
| **SMAPE** | Symmetric version of MAPE; bounded between 0–200%.                     | $$SMAPE = \frac{100}{n}\sum \frac{ \| y_t - \hat y_t \| }{( \| y_t \| + \| \hat y_t \| )/2}$$ | Custom NumPy formula |
| **RMSPE** | Root mean square percentage error; scale-free measure.                 | $$RMSPE = 100 \sqrt{\frac{1}{n}\sum \left(\frac{y_t - \hat y_t}{y_t}\right)^2}$$ | Custom NumPy formula  |

---

## ⚙️ 6. Summary: Choosing the Right Metric

| Problem Type              | Typical Metrics        | Use When                                |
| ------------------------- | ---------------------- | --------------------------------------- |
| **Classification**        | Accuracy, F1, ROC-AUC  | Balanced or imbalanced label prediction |
| **Regression**            | MAE, RMSE, R², MAPE    | Predicting continuous targets           |
| **Clustering**            | Silhouette, CH, DBI    | Evaluating cluster quality              |
| **Ranking / Recommender** | MAP, NDCG, Precision@k | Personalized recommendations, search    |
| **Forecasting**           | MAPE, SMAPE, RMSE      | Time series or sequential predictions   |

---

## ✅ 7. Key Takeaways

* Always **choose a metric aligned with your business goal**.
  (e.g., Recall for fraud detection, MAPE for forecast accuracy).
* Use **cross-validation** for robust metric estimation.
* Track **multiple metrics** — one metric rarely tells the full story.
* In production, monitor **metrics drift** to detect model decay.

--- 

## 🧭 ML Categories, Problem Types & Evaluation Metrics (Validated & Beginner-Friendly)

| **ML Category** | **Problem Type** | **Sub-Category / Nature** | **Typical Metrics** | **Use When / Description** |
|-----------------|------------------|-----------------------------|----------------------|-----------------------------|
| **Supervised Learning (SL)** | **Classification** | Binary / Multi-class / Multi-label / Imbalanced | Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Log-Loss | When you have labeled data (known outputs). The model learns to map input → output. Example: spam detection, disease diagnosis. |
|  | **Regression** | Linear / Nonlinear / Robust / Percentage-based | MAE, MSE, RMSE, R², MAPE, SMAPE | When predicting continuous values (e.g., prices, temperature). MAPE and SMAPE express accuracy as % error. |
| **Unsupervised Learning (USL)** | **Clustering** | Partition-based, Hierarchical, Density-based | Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index | When no labels exist; model groups similar data. Example: customer segmentation. |
|  | **Dimensionality Reduction** | PCA, t-SNE, UMAP | Reconstruction Error, Explained Variance | When simplifying features while preserving structure. Example: visualization, compression. |
| **Semi-Supervised Learning (SSL)** | **Hybrid Tasks** | Few labeled + many unlabeled samples | Accuracy, F1-Score, Log-Loss, ROC-AUC (on labeled subset) | When labeled data is expensive or scarce. Example: labeling a few medical images, learning from rest automatically. |
| **Self-Supervised Learning (Self-SL)** | **Representation Learning** | Contrastive, Masked Prediction, Autoencoding | Pretext-task accuracy, Transfer Learning performance | Model generates pseudo-labels from data itself. Example: predicting masked words in text (BERT), missing pixels in images. |
| **Reinforcement Learning (RL)** | **Decision / Control Tasks** | Single-agent / Multi-agent / Continuous / Discrete | Reward, Cumulative Return, Average Reward, Success Rate | Model learns by interacting with environment via trial-and-error. Example: robotics, game AI. |
| **Deep Learning (DL)** | **Supervised DL** | CNNs (images), RNNs (sequences), Transformers (text) | Accuracy, F1-Score, IoU, BLEU, ROUGE | Applies neural networks to complex data. Example: image classification, translation, text summarization. |
|  | **Unsupervised DL** | Autoencoders, GANs, VAEs | Reconstruction Loss, FID, Inception Score | Learns structure or generates new samples without labels. Example: synthetic image generation. |
|  | **Self-Supervised DL** | Pretraining models using pseudo-tasks | Linear Probe Accuracy, Fine-tune Accuracy | Foundation models trained on large unlabeled corpora; adapted to downstream tasks. |
| **Specialized Domains (Cross-Cutting)** | **Ranking / Recommender Systems** | Search, Recommendation, Personalization | Precision@k, Recall@k, MAP, NDCG | Measures top-k relevance and ordering of results. Example: Netflix, Google Search. |
|  | **Forecasting (Time-Series)** | Univariate / Multivariate / Multi-step | MAPE, SMAPE, RMSE, MAE | Predicts future values from past data. Example: sales, demand, energy load. |
|  | **Anomaly Detection** | Supervised / Unsupervised | Precision@k, Recall@k, F1, ROC-AUC, PR-AUC | Detects rare abnormal events. Example: fraud, system failures. |
|  | **Natural Language Processing (NLP)** | Text Classification / NER / Translation | Accuracy, F1, BLEU, ROUGE, METEOR | Evaluates language model outputs. Example: sentiment analysis, summarization. |
|  | **Computer Vision (CV)** | Object Detection / Segmentation / Recognition | mAP, IoU, Dice, Pixel Accuracy | Measures visual recognition accuracy. Example: detecting cars or faces in images. |
|  | **Generative Models** | GANs, Diffusion Models | FID, Inception Score, Perplexity | Evaluates realism and diversity of generated outputs. Example: image or text synthesis. |

---

### ✅ Simplified View (At a Glance)

| **ML Category** | **Typical Problems** | **Typical Metrics** | **Example Use-Case** |
|------------------|----------------------|----------------------|----------------------|
| **Supervised Learning (SL)** | Classification, Regression | Accuracy, F1, ROC-AUC, MAE, RMSE | Spam detection, price prediction |
| **Unsupervised Learning (USL)** | Clustering, Dimensionality Reduction | Silhouette, CH, DBI, Explained Variance | Customer segmentation, visualization |
| **Semi-Supervised Learning (SSL)** | Hybrid labeled/unlabeled learning | Accuracy, F1, ROC-AUC | Learning from few labeled samples |
| **Self-Supervised Learning (Self-SL)** | Pretext predictive tasks | Representation quality, Transfer Accuracy | Pretraining models on unlabeled data |
| **Reinforcement Learning (RL)** | Sequential decision-making | Reward, Cumulative Return | Robotics, games, trading bots |
| **Deep Learning (DL)** | Vision, NLP, Audio, Generative tasks | Accuracy, IoU, BLEU, FID | Image, speech, and text understanding |
| **Forecasting / Time-Series** | Predicting future trends | MAPE, SMAPE, RMSE | Sales, weather, demand forecasts |
| **Ranking / Recommendation** | Search / recommender systems | Precision@k, Recall@k, NDCG | Netflix, YouTube, Google Search |
| **Anomaly Detection** | Rare-event detection | F1, ROC-AUC, PR-AUC | Fraud, defect detection |

---

### 🧩 Plain-Language Summary for Beginners

| **Category** | **Plain Meaning** | **What It Learns** | **Example** |
|---------------|------------------|--------------------|--------------|
| **Supervised Learning** | Learns from examples with correct answers (labeled data). | Input → Output mapping. | Predicting loan default, image classification. |
| **Unsupervised Learning** | Learns from unlabeled data to find hidden patterns. | Data structure or clusters. | Grouping customers by purchase behavior. |
| **Semi-Supervised Learning** | Learns from a few labeled + many unlabeled examples. | Extends labeled data usefulness. | Medical image analysis (few labeled scans). |
| **Self-Supervised Learning** | Creates labels automatically from raw data. | Learns general representations. | Predict masked words in sentences (BERT). |
| **Reinforcement Learning** | Learns through trial and error to maximize reward. | Policy or strategy. | Self-driving cars, game agents. |
| **Deep Learning** | Uses large neural networks to learn features automatically. | Complex feature hierarchies. | Face recognition, speech translation. |

---

### ⚙️ Notes & Tips

- **Supervised Learning** dominates applied ML (most business problems have labeled data).  
- **Unsupervised Learning** is used for discovery, not prediction.  
- **Semi/Self-Supervised Learning** bridges gaps where labeled data is scarce.  
- **Reinforcement Learning** is powerful but data-hungry and environment-dependent.  
- **Deep Learning** cuts across all categories — when data is large and complex.  
- Always **align metrics** with your business objective (precision for spam filters, recall for disease detection, MAPE for forecasts).  

---

### 📚 References

1. <a href="https://en.wikipedia.org/wiki/Machine_learning" target="_blank" rel="noopener">Machine Learning — Wikipedia</a>  
2. <a href="https://www.ibm.com/think/topics/machine-learning" target="_blank" rel="noopener">What Is Machine Learning — IBM Think</a>  
3. <a href="https://www.geeksforgeeks.org/machine-learning/machine-learning/" target="_blank" rel="noopener">Machine Learning Overview — GeeksforGeeks</a>
4. 3. <a href="https://www.geeksforgeeks.org/machine-learning/introduction-machine-learning/" target="_blank" rel="noopener">Introduction to Machine Learning — GeeksforGeeks</a>

5. <a href="https://www.ibm.com/think/topics/machine-learning" target="_blank" rel="noopener">What Is Machine Learning (ML)? — IBM</a>
6. <a href="https://www.ibm.com/think/topics/machine-learning-algorithms" target="_blank" rel="noopener">Machine Learning Algorithms — IBM</a>

7. <a href="https://scikit-learn.org/stable/modules/model_evaluation.html" target="_blank" rel="noopener">Scikit-Learn: Model Evaluation — Official Docs</a>
8. <a href="https://developers.google.com/machine-learning/crash-course/classification/metrics" target="_blank" rel="noopener">Google ML Crash Course — Evaluation Metrics</a>  
9. <a href="https://paperswithcode.com/datasets-and-evaluation-metrics" target="_blank" rel="noopener">Papers With Code — Evaluation Metrics by Domain</a>  
10. <a href="https://mitsloan.mit.edu/ideas-made-to-matter/machine-learning-explained" target="_blank" rel="noopener">Machine Learning Explained — MIT Sloan</a>

11. <a href="https://en.wikipedia.org/wiki/Reinforcement_learning" target="_blank" rel="noopener">Reinforcement Learning — Wikipedia</a>  
12. <a href="https://en.wikipedia.org/wiki/Self-supervised_learning" target="_blank" rel="noopener">Self-Supervised Learning — Wikipedia</a>  
13. <a href="https://en.wikipedia.org/wiki/Deep_learning" target="_blank" rel="noopener">Deep Learning — Wikipedia</a>

---
