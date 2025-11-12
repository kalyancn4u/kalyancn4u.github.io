---
layout: post
title: "🌊 Scikit-learn: Deep Dive & Best Practices"
description: "Concise, clear, and validated revision notes on Scikit-learn library for Python — practical best practices for beginners and practitioners."
author: technical_notes
date: 2025-11-11 00:06:00 +05:30
categories: [Notes, Scikit-learn]
tags: [Scikit-learn, Sklearn, Machine Learning, Data Preprocessing, Pipelines, Model Evaluation, Best Practices]
toc: true
math: true
mermaid: true
---

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [The Estimator API](#the-estimator-api)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Pipelines](#pipelines)
7. [Model Selection and Evaluation](#model-selection-and-evaluation)
8. [Cross-Validation](#cross-validation)
9. [Hyperparameter Tuning](#hyperparameter-tuning)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Jargon Tables](#jargon-tables)

---

## Introduction

Scikit-learn (sklearn) is a comprehensive, open-source machine learning library for Python. Built on top of NumPy, SciPy, and Matplotlib, it provides simple and efficient tools for data mining, data analysis, and predictive modeling. Scikit-learn offers a consistent interface across diverse machine learning algorithms including classification, regression, clustering, dimensionality reduction, model selection, and preprocessing.

### Key Features
- **Unified API**: Consistent interface across all estimators
- **Extensive Algorithms**: Classification, regression, clustering, dimensionality reduction
- **Built-in Preprocessing**: Data transformation and feature engineering tools
- **Model Selection**: Cross-validation, grid search, and evaluation metrics
- **Robust Documentation**: Comprehensive guides and examples
- **Production Ready**: Efficient implementations suitable for real-world applications

---

## Core Concepts

### What is an Estimator?

An **estimator** is any object that learns from data by implementing a `fit()` method. Estimators can be classifiers, regressors, clusterers, or transformers. All scikit-learn estimators follow a consistent API pattern.

### Fundamental Principles

#### Consistency
All objects share a uniform interface with limited, well-documented methods:
- `fit(X, y)`: Learn parameters from training data
- `predict(X)`: Make predictions on new data
- `transform(X)`: Transform data (for transformers)
- `score(X, y)`: Evaluate model performance

#### Inspection
All learned parameters are accessible as public attributes with trailing underscores (e.g., `coef_`, `feature_importances_`).

#### Non-proliferation of Classes
Datasets are represented as NumPy arrays or SciPy sparse matrices. Hyperparameters are standard Python strings or numbers.

#### Composition
Machine learning algorithms are expressed as sequences of fundamental operations. Scikit-learn reuses existing building blocks whenever possible.

#### Sensible Defaults
Models provide reasonable default parameter values to enable quick prototyping.

---

## The Estimator API

### Core Methods

#### fit(X, y=None)
Trains or fits the model to data X (and target y, if applicable). Returns `self` to enable method chaining.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)  # Learns parameters from training data
```

#### predict(X)
Makes predictions on new data X using learned parameters. Used in supervised learning (classifiers and regressors).

```python
y_pred = model.predict(X_test)  # Returns predicted values
```

**Important**: `predict()` cannot be called before `fit()`. Attempting to do so raises `NotFittedError`.

#### transform(X)
Transforms input data X. Used by transformers (scalers, encoders, dimensionality reducers).

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # Fit and transform in one step
X_test_scaled = scaler.transform(X_test)  # Apply same transformation to test data
```

#### fit_transform(X, y=None)
Convenience method combining `fit()` and `transform()`. More efficient than calling them separately. Primarily used for transformers.

```python
# Instead of:
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)

# Use:
X_scaled = scaler.fit_transform(X_train)
```

#### fit_predict(X, y=None)
Fits the model and returns predictions on training data. Relevant for unsupervised learning (clustering algorithms).

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)  # Fit and predict cluster labels
```

#### score(X, y)
Evaluates model performance. Returns different metrics based on estimator type:
- **Classifiers**: Accuracy score
- **Regressors**: R² coefficient of determination
- **Clusterers**: Not typically available

```python
accuracy = classifier.score(X_test, y_test)
r2_score = regressor.score(X_test, y_test)
```

### Estimator Types

#### Classifiers
Inherit from `ClassifierMixin`. Predict discrete class labels.

**Key Methods**:
- `fit(X, y)`: Train on features X and labels y
- `predict(X)`: Return predicted class labels
- `predict_proba(X)`: Return probability estimates for each class
- `predict_log_proba(X)`: Return log-probabilities
- `decision_function(X)`: Return confidence scores
- `score(X, y)`: Return accuracy score (default metric)

**Examples**: `LogisticRegression`, `RandomForestClassifier`, `SVC`

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

#### Regressors
Inherit from `RegressorMixin`. Predict continuous target values.

**Key Methods**:
- `fit(X, y)`: Train on features X and continuous targets y
- `predict(X)`: Return predicted values
- `score(X, y)`: Return R² score (default metric)

**Examples**: `LinearRegression`, `RandomForestRegressor`, `SVR`

```python
from sklearn.linear_model import Ridge

reg = Ridge(alpha=1.0)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2 = reg.score(X_test, y_test)
```

#### Transformers
Inherit from `TransformerMixin`. Transform input data without changing the number of samples.

**Key Methods**:
- `fit(X, y=None)`: Learn transformation parameters
- `transform(X)`: Apply transformation
- `fit_transform(X, y=None)`: Fit and transform in one step
- `inverse_transform(X)`: Reverse transformation (when applicable)

**Examples**: `StandardScaler`, `PCA`, `OneHotEncoder`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_original = scaler.inverse_transform(X_test_scaled)  # Reverse scaling
```

#### Clusterers
Inherit from `ClusterMixin`. Group similar samples together.

**Key Methods**:
- `fit(X, y=None)`: Learn cluster structure (y is ignored if provided)
- `predict(X)`: Assign cluster labels to new data (when applicable)
- `fit_predict(X)`: Fit and assign labels in one step

**Learned Attributes**:
- `labels_`: Cluster labels for each training sample

**Examples**: `KMeans`, `DBSCAN`, `AgglomerativeClustering`

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(X)
new_labels = kmeans.predict(X_new)  # Assign new samples to clusters
```

---

## Data Preprocessing

Data preprocessing transforms raw data into a format suitable for machine learning algorithms. Scikit-learn provides comprehensive preprocessing tools.

### Scaling and Normalization

#### StandardScaler
Standardizes features by removing mean and scaling to unit variance (z-score normalization).

**Formula**: `z = (x - μ) / σ`

**Use When**: Features have Gaussian distribution or different scales.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Learned parameters
print(scaler.mean_)  # Mean of each feature
print(scaler.scale_)  # Standard deviation of each feature
```

#### MinMaxScaler
Scales features to a specified range (default [0, 1]).

**Formula**: `X_scaled = (X - X_min) / (X_max - X_min)`

**Use When**: Features need to be bounded within a specific range.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)
```

#### RobustScaler
Uses statistics robust to outliers (median and interquartile range).

**Use When**: Data contains many outliers.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

#### MaxAbsScaler
Scales each feature by its maximum absolute value. Preserves sparsity.

**Use When**: Data is already centered at zero or is sparse.

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_train)
```

#### Normalizer
Normalizes samples individually to unit norm (L1, L2, or max norm).

**Use When**: Interested in the direction of feature vectors, not magnitude.

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X)
```

### Encoding Categorical Variables

#### OneHotEncoder
Creates binary columns for each category (one-hot encoding/dummy variables).

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first to avoid multicollinearity
X_encoded = encoder.fit_transform(X_categorical)

# Get feature names
feature_names = encoder.get_feature_names_out()
```

#### OrdinalEncoder
Encodes categorical features as integers (preserves ordinal relationship).

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
X_encoded = encoder.fit_transform(X_ordinal)
```

#### LabelEncoder
Encodes target labels as integers (for classification tasks).

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Inverse transform to get original labels
y_original = encoder.inverse_transform(y_encoded)
```

#### TargetEncoder
Encodes categories using target statistics (supervised encoding).

```python
from sklearn.preprocessing import TargetEncoder

encoder = TargetEncoder(smooth=0.25)
X_encoded = encoder.fit_transform(X_categorical, y)
```

### Handling Missing Values

#### SimpleImputer
Replaces missing values using various strategies.

**Strategies**:
- `mean`: Replace with column mean
- `median`: Replace with column median
- `most_frequent`: Replace with mode
- `constant`: Replace with specified constant

```python
from sklearn.impute import SimpleImputer

# For numerical features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numerical)

# For categorical features
imputer_cat = SimpleImputer(strategy='most_frequent')
X_cat_imputed = imputer_cat.fit_transform(X_categorical)
```

#### KNNImputer
Imputes missing values using k-nearest neighbors.

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

#### IterativeImputer
Models each feature with missing values as a function of other features.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
X_imputed = imputer.fit_transform(X)
```

### Discretization

#### KBinsDiscretizer
Bins continuous features into discrete intervals.

```python
from sklearn.preprocessing import KBinsDiscretizer

discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_binned = discretizer.fit_transform(X)
```

### Custom Transformations

#### FunctionTransformer
Applies custom functions to features.

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Apply log transformation
log_transformer = FunctionTransformer(np.log1p, validate=True)
X_log = log_transformer.fit_transform(X)
```

---

## Feature Engineering

Feature engineering creates new features or transforms existing ones to improve model performance.

### Polynomial Features

Creates polynomial and interaction features.

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Example: [a, b] becomes [a, b, a², ab, b²]
```

### Feature Selection

#### Variance Threshold
Removes features with low variance.

```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
```

#### SelectKBest
Selects k highest scoring features based on statistical tests.

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature indices
selected_features = selector.get_support(indices=True)
```

#### Recursive Feature Elimination (RFE)
Recursively removes features and builds model on remaining features.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=10, step=1)
X_selected = selector.fit_transform(X, y)

# Get feature rankings
print(selector.ranking_)
```

#### SelectFromModel
Selects features based on importance weights from a model.

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier(n_estimators=100)
selector = SelectFromModel(estimator, threshold='median')
X_selected = selector.fit_transform(X, y)
```

### Dimensionality Reduction

#### Principal Component Analysis (PCA)
Reduces dimensionality by projecting data onto principal components.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Retain 95% variance
X_reduced = pca.fit_transform(X)

# Explained variance
print(pca.explained_variance_ratio_)
```

#### Linear Discriminant Analysis (LDA)
Supervised dimensionality reduction maximizing class separability.

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_reduced = lda.fit_transform(X, y)
```

#### t-SNE
Non-linear dimensionality reduction for visualization.

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)
```

### Text Feature Extraction

#### CountVectorizer
Converts text documents to token count matrix.

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X_counts = vectorizer.fit_transform(documents)
```

#### TfidfVectorizer
Converts text to TF-IDF features.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(documents)
```

### Time-Related Feature Engineering

#### Cyclical Encoding
Encodes periodic features (hour, day, month) preserving cyclical nature.

```python
import numpy as np

def encode_cyclical(data, period):
    """Encode cyclical feature using sine and cosine."""
    sin_feature = np.sin(2 * np.pi * data / period)
    cos_feature = np.cos(2 * np.pi * data / period)
    return sin_feature, cos_feature

# Example: Encode hour of day (24-hour period)
hour_sin, hour_cos = encode_cyclical(df['hour'], 24)
```

---

## Pipelines

Pipelines chain multiple processing steps together, ensuring consistent transformations across train/test data and preventing data leakage.

### Creating Pipelines

#### make_pipeline
Quick way to create pipeline without naming steps.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=10),
    LogisticRegression()
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

#### Pipeline
Create pipeline with named steps for better control.

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
```

### ColumnTransformer

Apply different transformations to different columns.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'income']),
        ('cat', OneHotEncoder(), ['city', 'gender'])
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

### FeatureUnion

Concatenate results of multiple transformer objects.

```python
from sklearn.pipeline import FeatureUnion

feature_union = FeatureUnion([
    ('pca', PCA(n_components=10)),
    ('select', SelectKBest(k=15))
])

pipe = Pipeline([
    ('features', feature_union),
    ('classifier', LogisticRegression())
])
```

### Pipeline Benefits

1. **Prevents Data Leakage**: Transformations fitted only on training data
2. **Cleaner Code**: Encapsulates entire workflow
3. **Easy Deployment**: Single object for entire pipeline
4. **Cross-Validation Compatible**: Works seamlessly with CV
5. **Hyperparameter Tuning**: Can optimize all steps together

### Accessing Pipeline Steps

```python
# Access specific step
scaler = pipe.named_steps['scaler']

# Get learned parameters
print(scaler.mean_)

# Access intermediate transformations
X_transformed = pipe[:-1].transform(X_test)
```

---

## Model Selection and Evaluation

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,     # Reproducibility
    stratify=y          # Maintain class distribution
)
```

### Evaluation Metrics

#### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# ROC-AUC for probability predictions
y_proba = classifier.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Detailed report
print(classification_report(y_test, y_pred))
```

**Metric Selection**:
- **Accuracy**: Overall correctness (use when classes are balanced)
- **Precision**: True positives / Predicted positives (minimize false positives)
- **Recall**: True positives / Actual positives (minimize false negatives)
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)
- **ROC-AUC**: Trade-off between true positive and false positive rates

#### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, mean_absolute_percentage_error
)

# Common metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
```

**Metric Selection**:
- **MSE/RMSE**: Penalizes large errors more heavily
- **MAE**: Robust to outliers
- **R²**: Proportion of variance explained (0 to 1)
- **MAPE**: Relative error as percentage

### Handling Imbalanced Datasets

#### Class Weights

```python
from sklearn.linear_model import LogisticRegression

# Automatically balance class weights
clf = LogisticRegression(class_weight='balanced')

# Manual weights
clf = LogisticRegression(class_weight={0: 1.0, 1: 5.0})
```

#### Resampling

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Oversampling with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Combined pipeline
pipe = ImbPipeline([
    ('smote', SMOTE()),
    ('classifier', LogisticRegression())
])
```

#### Stratified Splitting

```python
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in splitter.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

---

## Cross-Validation

Cross-validation assesses model generalization by training and testing on different data subsets.

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Stratified K-Fold

Maintains class distribution in each fold.

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(classifier, X, y, cv=skfold)
```

### cross_validate

Returns multiple metrics and timing information.

```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X, y, 
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True,
    return_estimator=True  # Return fitted models
)

print(cv_results.keys())
# ['fit_time', 'score_time', 'test_accuracy', 'train_accuracy', ...]
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

### Leave-One-Out Cross-Validation

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

---

## Hyperparameter Tuning

### Grid Search

Exhaustively searches specified parameter combinations.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
```

### Randomized Search

Samples random parameter combinations.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,          # Number of parameter combinations to try
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
```

### Pipeline Hyperparameter Tuning

Use double underscore notation to access nested parameters.

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier())
])

param_grid = {
    'pca__n_components': [5, 10, 15, 20],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
```

### Analyzing Grid Search Results

```python
import pandas as pd

# Convert results to DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# Important columns
print(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])

# Best results
print(results_df.nlargest(5, 'mean_test_score'))
```

---

## Best Practices

### 1. Always Set random_state

Ensures reproducibility of results involving randomness.

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model with randomness
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
```

### 2. Use Pipelines to Prevent Data Leakage

**Wrong Approach** (causes data leakage):
```python
# DON'T DO THIS - fits on entire dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
```

**Correct Approach**:
```python
# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Use pipeline - fits only on training data
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 3. Apply Consistent Transformations

Always use the same transformer fitted on training data for test data.

```python
# Fit on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data with same parameters
X_test_scaled = scaler.transform(X_test)  # NOT fit_transform!
```

### 4. Stratify When Splitting

Maintains class distribution in train/test splits.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    stratify=y,  # Maintains class proportions
    random_state=42
)
```

### 5. Use Cross-Validation for Model Evaluation

Single train-test split may be unreliable.

```python
# Instead of single split
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# Use cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
```

### 6. Choose Appropriate Metrics

Don't rely solely on accuracy, especially with imbalanced datasets.

```python
from sklearn.metrics import classification_report, confusion_matrix

# For imbalanced classification
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Use appropriate scoring in cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
```

### 7. Scale Features for Distance-Based Algorithms

Algorithms sensitive to feature scales: SVM, KNN, Neural Networks, Linear/Logistic Regression with regularization.

```python
# Required for these algorithms
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

pipe = Pipeline([
    ('scaler', StandardScaler()),  # Essential!
    ('classifier', SVC())
])
```

**Not required** for tree-based algorithms: Decision Trees, Random Forest, Gradient Boosting.

### 8. Handle Missing Values Explicitly

```python
# Check for missing values
print(X.isnull().sum())

# Handle in pipeline
from sklearn.impute import SimpleImputer

pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

### 9. Use n_jobs for Parallelization

Speed up computations on multi-core machines.

```python
# Model training
model = RandomForestClassifier(n_estimators=100, n_jobs=-1)  # Use all cores

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)

# Grid search
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
```

### 10. Save and Load Models

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
loaded_model = joblib.load('model.pkl')
predictions = loaded_model.predict(X_test)
```

### 11. Feature Importance Analysis

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
feature_names = X.columns  # Assuming pandas DataFrame

# Sort and visualize
indices = importances.argsort()[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
```

### 12. Monitor Training and Validation Performance

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, 
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# Plot learning curve
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend()
plt.show()
```

### 13. Use Appropriate Validation Strategy

```python
# Time series data - use TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Grouped data - use GroupKFold
from sklearn.model_selection import GroupKFold
gkfold = GroupKFold(n_splits=5)

# Stratified for classification
from sklearn.model_selection import StratifiedKFold
skfold = StratifiedKFold(n_splits=5)
```

### 14. Encode Target Variable for Classification

```python
from sklearn.preprocessing import LabelEncoder

# If target is categorical strings
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train model
model.fit(X_train, y_encoded)

# Decode predictions
y_pred = model.predict(X_test)
y_pred_original = le.inverse_transform(y_pred)
```

### 15. Optimize Memory Usage

```python
# Use sparse matrices for sparse data
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X)

# Specify dtype for memory efficiency
import numpy as np
X = X.astype(np.float32)  # Instead of float64

# Use sparse output in encoders
encoder = OneHotEncoder(sparse_output=True)
```

---

## Common Pitfalls

### 1. Data Leakage

**Problem**: Information from test set influences training.

**Examples**:
- Fitting scaler on entire dataset before split
- Using target variable for feature engineering
- Not handling time series data properly

**Solution**: Always split data first, fit only on training data.

```python
# Wrong
scaler.fit(X)  # Uses information from test set
X_train, X_test = train_test_split(X)

# Correct
X_train, X_test = train_test_split(X)
scaler.fit(X_train)  # Only learns from training data
```

### 2. Not Using Pipelines

**Problem**: Inconsistent transformations between train and test data.

**Solution**: Always use pipelines for multi-step workflows.

```python
# Use this pattern
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])
pipe.fit(X_train, y_train)
```

### 3. Overfitting

**Symptoms**:
- High training accuracy, low test accuracy
- Large gap between training and validation scores

**Solutions**:
```python
# Regularization
model = LogisticRegression(C=0.1)  # Stronger regularization

# More training data or data augmentation

# Simpler model
model = RandomForestClassifier(max_depth=5)  # Limit complexity

# Cross-validation to detect
scores = cross_val_score(model, X, y, cv=5)
```

### 4. Ignoring Class Imbalance

**Problem**: Model biased toward majority class.

**Solutions**:
```python
# Class weights
model = LogisticRegression(class_weight='balanced')

# Resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Appropriate metrics
from sklearn.metrics import f1_score, roc_auc_score
```

### 5. Not Scaling Features

**Problem**: Algorithms like SVM, KNN perform poorly with unscaled features.

**Solution**: Always scale for distance-based algorithms.

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])
```

### 6. Using predict() Instead of predict_proba()

**Problem**: Loses probability information for threshold tuning.

**Solution**: Use probabilities when available.

```python
# Get probabilities
y_proba = classifier.predict_proba(X_test)[:, 1]

# Adjust decision threshold
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Choose optimal threshold
optimal_threshold = thresholds[np.argmax(precision + recall)]
y_pred_adjusted = (y_proba >= optimal_threshold).astype(int)
```

### 7. Incorrect Cross-Validation for Time Series

**Problem**: Using standard K-Fold for temporal data causes data leakage.

**Solution**: Use TimeSeriesSplit.

```python
# Wrong for time series
kfold = KFold(n_splits=5)

# Correct for time series
tscv = TimeSeriesSplit(n_splits=5)
```

### 8. Not Handling Categorical Variables

**Problem**: Passing categorical strings directly to algorithms.

**Solution**: Encode categorical variables properly.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
```

### 9. Forgetting to Set random_state

**Problem**: Results not reproducible.

**Solution**: Always set random_state.

```python
# Everywhere randomness is involved
train_test_split(X, y, random_state=42)
RandomForestClassifier(random_state=42)
KFold(n_splits=5, shuffle=True, random_state=42)
```

### 10. Using Default Hyperparameters

**Problem**: Default parameters may not be optimal for your data.

**Solution**: Tune hyperparameters systematically.

```python
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

---

## Jargon Tables

### Table 1: Machine Learning Lifecycle Terminology

| Scikit-learn Term | Alternative Terms | Definition | Context |
|-------------------|-------------------|------------|---------|
| **fit** | train, learn, build | Learn parameters from training data | Model training phase |
| **transform** | preprocess, convert, map | Apply learned transformation to data | Data preprocessing |
| **predict** | infer, forecast, estimate | Generate predictions on new data | Model inference |
| **score** | evaluate, assess, measure | Calculate performance metric | Model evaluation |
| **fit_transform** | learn and apply | Fit transformer and transform data in one step | Efficient preprocessing |
| **fit_predict** | train and infer | Fit model and predict on same data | Clustering/unsupervised |
| **estimator** | model, learner, algorithm | Object that learns from data | Core ML concept |
| **pipeline** | workflow, chain, sequence | Series of data transformations and model | End-to-end process |
| **cross-validation** | k-fold validation, CV | Assess model on multiple train-test splits | Model validation |
| **hyperparameter** | tuning parameter, meta-parameter | Parameter set before training | Model configuration |
| **feature** | attribute, variable, predictor | Input variable for model | Data representation |
| **target** | label, response, outcome, dependent variable | Variable to predict | Supervised learning |
| **sample** | instance, observation, example, data point | Single row of data | Dataset element |
| **training set** | train data, training data | Data used to fit model | Model training |
| **test set** | holdout set, validation set | Data used to evaluate model | Model evaluation |
| **overfitting** | memorization, high variance | Model too complex for data | Model diagnostic |
| **underfitting** | oversimplification, high bias | Model too simple for data | Model diagnostic |
| **regularization** | penalization, shrinkage | Technique to reduce overfitting | Model constraint |
| **stratification** | proportional sampling | Maintain class distribution in splits | Sampling technique |

### Table 2: Hierarchical Differentiation of Estimator Types

| Level | Category | Subcategory | Examples | Primary Methods |
|-------|----------|-------------|----------|-----------------|
| **1** | **Estimator** | Base class for all | All sklearn objects | `fit()` |
| **2** | **Predictor** | Supervised learners | All classifiers/regressors | `fit()`, `predict()`, `score()` |
| | | **Classifier** | LogisticRegression, SVC | `predict_proba()`, `decision_function()` |
| | | **Regressor** | LinearRegression, SVR | (basic predict only) |
| **2** | **Transformer** | Data transformers | Scalers, encoders, PCA | `fit()`, `transform()`, `fit_transform()` |
| | | **Feature Extractor** | CountVectorizer, TfidfVectorizer | Text → numeric features |
| | | **Feature Selector** | SelectKBest, RFE | Reduce feature dimensions |
| | | **Preprocessor** | StandardScaler, OneHotEncoder | Data normalization/encoding |
| | | **Dimensionality Reducer** | PCA, LDA, t-SNE | Reduce feature space |
| **2** | **Clusterer** | Unsupervised learners | KMeans, DBSCAN | `fit()`, `fit_predict()` |
| **2** | **Meta-Estimator** | Wraps other estimators | Pipeline, GridSearchCV | Composite functionality |
| | | **Ensemble** | RandomForest, GradientBoosting | Combines multiple models |
| | | **Multioutput** | MultiOutputClassifier | Handles multiple targets |
| | | **Calibration** | CalibratedClassifierCV | Probability calibration |

### Table 3: Data Splitting Terminology

| Term | Description | Use Case | Scikit-learn Function |
|------|-------------|----------|----------------------|
| **Train-Test Split** | Single split into two sets | Quick model evaluation | `train_test_split()` |
| **K-Fold CV** | k equal-sized folds | General cross-validation | `KFold` |
| **Stratified K-Fold** | K-fold maintaining class distribution | Classification with imbalanced classes | `StratifiedKFold` |
| **Time Series Split** | Sequential splits for temporal data | Time series forecasting | `TimeSeriesSplit` |
| **Leave-One-Out** | n folds (one sample per fold) | Small datasets | `LeaveOneOut` |
| **Leave-P-Out** | All combinations leaving p samples out | Very small datasets | `LeavePOut` |
| **Group K-Fold** | K-fold respecting group boundaries | Grouped data (patients, sessions) | `GroupKFold` |
| **Shuffle Split** | Random permutation split | Flexible train/test sizes | `ShuffleSplit` |

### Table 4: Preprocessing Method Categories

| Operation Type | Technique | Purpose | Scikit-learn Class |
|----------------|-----------|---------|-------------------|
| **Scaling** | Standardization | Mean=0, Std=1 | `StandardScaler` |
| | Min-Max Scaling | Range [0,1] or custom | `MinMaxScaler` |
| | Robust Scaling | Use median and IQR | `RobustScaler` |
| | Max-Abs Scaling | Divide by max absolute value | `MaxAbsScaler` |
| **Normalization** | L1/L2 Norm | Scale samples to unit norm | `Normalizer` |
| **Encoding** | One-Hot Encoding | Binary columns per category | `OneHotEncoder` |
| | Ordinal Encoding | Integer encoding | `OrdinalEncoder` |
| | Label Encoding | Encode target labels | `LabelEncoder` |
| | Target Encoding | Use target statistics | `TargetEncoder` |
| **Imputation** | Mean/Median | Fill with central tendency | `SimpleImputer` |
| | KNN Imputation | Use k-nearest neighbors | `KNNImputer` |
| | Iterative | Model-based imputation | `IterativeImputer` |
| **Discretization** | Binning | Convert continuous to discrete | `KBinsDiscretizer` |
| **Feature Creation** | Polynomial | Generate polynomial features | `PolynomialFeatures` |

### Table 5: Model Evaluation Terminology

| Metric Type | Metric Name | Formula/Description | Best Value | Use Case |
|-------------|-------------|---------------------|------------|----------|
| **Classification** | Accuracy | (TP + TN) / Total | 1.0 | Balanced classes |
| | Precision | TP / (TP + FP) | 1.0 | Minimize false positives |
| | Recall/Sensitivity | TP / (TP + FN) | 1.0 | Minimize false negatives |
| | F1-Score | 2 × (Precision × Recall) / (Precision + Recall) | 1.0 | Balance precision/recall |
| | ROC-AUC | Area under ROC curve | 1.0 | Binary classification |
| | Log Loss | -Σ(y log(p) + (1-y)log(1-p)) | 0.0 | Probability accuracy |
| **Regression** | MSE | Mean((y - ŷ)²) | 0.0 | Penalize large errors |
| | RMSE | √MSE | 0.0 | Same units as target |
| | MAE | Mean(\|y - ŷ\|) | 0.0 | Robust to outliers |
| | R² | 1 - (SS_res / SS_tot) | 1.0 | Variance explained |
| | MAPE | Mean(\|y - ŷ\| / \|y\|) × 100 | 0.0 | Percentage error |
| **Clustering** | Silhouette Score | (b - a) / max(a, b) | 1.0 | Cluster separation |
| | Davies-Bouldin | Avg similarity of clusters | 0.0 | Cluster compactness |
| | Calinski-Harabasz | Between/within variance ratio | Higher better | Cluster density |

### Table 6: Hyperparameter Tuning Terminology

| Term | Description | Strategy | Scikit-learn Class |
|------|-------------|----------|-------------------|
| **Grid Search** | Exhaustive search over parameter grid | Try all combinations | `GridSearchCV` |
| **Random Search** | Random sampling from parameter distributions | Sample n_iter combinations | `RandomizedSearchCV` |
| **Halving Grid Search** | Successive halving with grid | Early stopping low performers | `HalvingGridSearchCV` |
| **Halving Random Search** | Successive halving with random | Progressive refinement | `HalvingRandomSearchCV` |
| **Manual Search** | User-defined parameter testing | Custom iteration | Loop with `cross_val_score` |

---

## Advanced Topics

### Custom Estimators

Create custom estimators by inheriting from base classes.

```python
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.0):
        self.factor = factor
    
    def fit(self, X, y=None):
        # Learn parameters from training data
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        return self
    
    def transform(self, X):
        # Apply transformation
        return (X - self.mean_) / (self.std_ * self.factor)
    
    def inverse_transform(self, X):
        # Reverse transformation
        return (X * self.std_ * self.factor) + self.mean_

# Use in pipeline
pipe = Pipeline([
    ('custom_scaler', CustomScaler(factor=2.0)),
    ('classifier', LogisticRegression())
])
```

### Ensemble Methods

Combine multiple models for better performance.

```python
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier,
    BaggingClassifier, AdaBoostClassifier
)

# Voting Classifier - average predictions
voting = VotingClassifier([
    ('lr', LogisticRegression()),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True))
], voting='soft')  # 'soft' uses probabilities

# Stacking Classifier - meta-model on predictions
stacking = StackingClassifier([
    ('rf', RandomForestClassifier()),
    ('svc', SVC())
], final_estimator=LogisticRegression())

# Bagging - bootstrap aggregation
bagging = BaggingClassifier(
    LogisticRegression(),
    n_estimators=10,
    max_samples=0.8,
    random_state=42
)

# Boosting - sequential learning
boosting = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
```

### Calibration

Improve probability predictions.

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate classifier probabilities
calibrated = CalibratedClassifierCV(
    SVC(),  # SVC doesn't have good probability estimates
    method='sigmoid',  # or 'isotonic'
    cv=5
)

calibrated.fit(X_train, y_train)
y_proba = calibrated.predict_proba(X_test)
```

### Multi-Output Models

Handle multiple target variables.

```python
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Multi-output classification
multi_clf = MultiOutputClassifier(RandomForestClassifier())
multi_clf.fit(X_train, y_train)  # y_train has multiple columns

# Multi-output regression
multi_reg = MultiOutputRegressor(LinearRegression())
multi_reg.fit(X_train, y_train)
```

### Handling Mixed Data Types

```python
from sklearn.compose import make_column_transformer, make_column_selector

# Automatic selection by dtype
preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(), make_column_selector(dtype_include=object))
)

# Use in pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```

### Partial Fitting for Large Datasets

Train on data that doesn't fit in memory.

```python
from sklearn.linear_model import SGDClassifier

# Models supporting partial_fit
model = SGDClassifier()

# Train in batches
for X_batch, y_batch in data_batches:
    model.partial_fit(X_batch, y_batch, classes=np.unique(y))
```

### Feature Hashing

Efficient encoding for high-dimensional categorical features.

```python
from sklearn.feature_extraction import FeatureHasher

hasher = FeatureHasher(n_features=10, input_type='string')
X_hashed = hasher.transform(raw_data)
```

---

## Performance Optimization Tips

### 1. Use Appropriate Data Structures

```python
# For sparse data, use sparse matrices
from scipy.sparse import csr_matrix
X_sparse = csr_matrix(X)

# For dense data with consistent dtype
X = np.array(X, dtype=np.float32)  # Less memory than float64
```

### 2. Parallelize Operations

```python
# Use all CPU cores
model = RandomForestClassifier(n_jobs=-1)
scores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
grid_search = GridSearchCV(model, params, n_jobs=-1)
```

### 3. Reduce Data Size

```python
# Feature selection before training
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=20)
X_reduced = selector.fit_transform(X, y)

# Sample data for prototyping
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1)
```

### 4. Use Warm Start for Iterative Models

```python
# Continue training from previous state
model = GradientBoostingClassifier(warm_start=True, n_estimators=100)
model.fit(X_train, y_train)

# Add more trees
model.n_estimators = 200
model.fit(X_train, y_train)  # Continues from 100 trees
```

### 5. Profile Your Code

```python
import time

start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(f"Training time: {end - start:.2f} seconds")

# Use cross_validate for detailed timing
cv_results = cross_validate(model, X, y, cv=5, return_train_score=True)
print(f"Avg fit time: {cv_results['fit_time'].mean():.3f}s")
```

---

## Complete Example Workflow

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# 2. Identify feature types
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# 3. Create preprocessing pipeline
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Create full pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

# 7. Perform grid search with cross-validation
grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 8. Get best model
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# 9. Evaluate on test set
y_pred = best_model.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Cross-validation on full dataset
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='f1_weighted')
print(f"\nCross-validation F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# 11. Save model
joblib.dump(best_model, 'best_model.pkl')

# 12. Load and use model later
loaded_model = joblib.load('best_model.pkl')
new_predictions = loaded_model.predict(X_new)
```

---

## References

<div style="line-height: 1.8;">

1. <a href="https://scikit-learn.org/stable/documentation.html" target="_blank">Scikit-learn Official Documentation</a>

2. <a href="https://scikit-learn.org/stable/developers/develop.html" target="_blank">Scikit-learn Development Guide</a>

3. <a href="https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html" target="_blank">Choosing the Right Estimator - Scikit-learn</a>

4. <a href="https://scikit-learn.org/stable/modules/cross_validation.html" target="_blank">Cross-validation: Evaluating Estimator Performance</a>

5. <a href="https://scikit-learn.org/stable/modules/grid_search.html" target="_blank">Tuning the Hyper-parameters of an Estimator</a>

6. <a href="https://scikit-learn.org/stable/modules/compose.html" target="_blank">Pipelines and Composite Estimators</a>

7. <a href="https://scikit-learn.org/stable/modules/preprocessing.html" target="_blank">Preprocessing Data</a>

8. <a href="https://scikit-learn.org/stable/modules/feature_selection.html" target="_blank">Feature Selection</a>

9. <a href="https://scikit-learn.org/stable/auto_examples/index.html" target="_blank">Scikit-learn Examples Gallery</a>

10. <a href="https://scikit-learn.org/stable/common_pitfalls.html" target="_blank">Common Pitfalls and Recommended Practices</a>

11. <a href="https://arxiv.org/abs/1201.0490" target="_blank">Scikit-learn: Machine Learning in Python (Research Paper)</a>

12. <a href="https://inria.hal.science/hal-00650905v2/document" target="_blank">API Design for Machine Learning Software (Research Paper)</a>

</div>
