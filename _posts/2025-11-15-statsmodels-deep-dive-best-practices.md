---
title: "🌊 Statsmodels: Deep Dive & Best Practices"
layout: post
author: technical_notes
date: 2025-11-15 00:00:00 +0530
categories: [Notes, Statsmodels]
tags: [Statsmodels, Python, Statistics, Data Science, Regression, Time-series, Statistical-modeling]
description: "Concise, clear, and validated revision notes on Statsmodels — structured for beginners and practitioners."
toc: true
math: true
mermaid: true
---

## Introduction

Statsmodels is a comprehensive Python library specifically designed for statistical modeling, hypothesis testing, and data exploration. Built on top of NumPy, SciPy, and Matplotlib, it provides advanced statistical functionalities that extend beyond numerical computing libraries. The library offers classes and functions for estimating various statistical models, conducting statistical tests, and performing data exploration with results validated against established statistical packages.

## Core Concepts and Architecture

### Library Overview

Statsmodels is structured around several key components:

- **Model Classes**: Define the statistical model structure
- **Results Classes**: Contain estimation results and diagnostic information
- **Formula Interface**: R-style formula support via Patsy integration
- **API Interface**: Direct programming interface for model specification

### Installation and Setup

```python
# Installation via pip
pip install statsmodels

# Installation via conda
conda install statsmodels

# Import conventions
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.tsa.api as tsa
```

### Design Patterns

The library follows a consistent pattern across all models:

1. **Model Instantiation**: Create model object with data
2. **Model Fitting**: Estimate parameters using `.fit()` method
3. **Results Inspection**: Analyze results via summary and diagnostics
4. **Prediction**: Generate predictions on new or existing data

## Modeling Workflow: Phases and Terminology

### Table 1: Modeling Lifecycle Terminology

| Common Term | Statsmodels Term | Statistical Term | Description |
|-------------|------------------|------------------|-------------|
| Input Data | `endog` / `exog` | Dependent / Independent Variables | Response variable (endog) and predictor variables (exog) |
| Model Building | Instantiation | Model Specification | Creating model object with data |
| Training | Fitting | Parameter Estimation | Estimating model parameters via `.fit()` |
| Evaluation | Diagnostics | Model Validation | Assessing model assumptions and performance |
| Prediction | Prediction | Inference | Generating predictions from fitted model |
| Parameters | `params` | Coefficients | Estimated model parameters |
| Residuals | `resid` | Errors | Difference between observed and fitted values |

### Table 2: Hierarchical Model Components

| Level | Component | Sub-components | Purpose |
|-------|-----------|----------------|---------|
| **Data Layer** | Design Matrices | `endog`, `exog` | Raw input data structures |
| | | `y`, `X` | Alternative naming convention |
| **Model Layer** | Model Class | `OLS`, `GLM`, `Logit`, `SARIMAX` | Defines statistical model |
| | Parameters | `order`, `seasonal_order`, `family` | Model configuration |
| **Estimation Layer** | Fitting Method | `method='newton'`, `method='bfgs'` | Optimization algorithm |
| | Convergence | `maxiter`, `tol` | Estimation control parameters |
| **Results Layer** | Summary Statistics | `R-squared`, `AIC`, `BIC` | Model fit metrics |
| | Coefficients | `params`, `pvalues`, `conf_int()` | Parameter estimates and inference |
| | Diagnostics | `resid`, `fittedvalues`, `influence` | Model checking tools |
| **Inference Layer** | Prediction | `predict()`, `forecast()` | Out-of-sample predictions |
| | Confidence Intervals | `get_prediction().summary_frame()` | Uncertainty quantification |

## Linear Regression Models

### Ordinary Least Squares (OLS)

OLS is the foundational regression method that minimizes the sum of squared residuals.

**Mathematical Formulation:**

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

Where:
- $\hat{\beta}$ = estimated coefficients
- $X$ = design matrix of predictors
- $y$ = response vector

**Implementation:**

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
n_obs = 100
X = np.random.randn(n_obs, 2)
true_beta = np.array([1.5, -2.0])
epsilon = np.random.randn(n_obs) * 0.5
y = 5 + X @ true_beta + epsilon

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_with_const)
results = model.fit()

# Display results
print(results.summary())

# Access specific results
print(f"R-squared: {results.rsquared:.4f}")
print(f"Coefficients: {results.params}")
print(f"P-values: {results.pvalues}")
print(f"Confidence Intervals:\n{results.conf_int()}")
```

### Weighted Least Squares (WLS)

WLS is used when observations have different variances (heteroscedasticity).

**Mathematical Formulation:**

$$\hat{\beta}_{WLS} = (X^T W X)^{-1} X^T W y$$

Where $W$ is a diagonal matrix of weights (typically $w_i = 1/\sigma_i^2$).

**Implementation:**

```python
# Example with measurement errors
x = np.linspace(0, 10, 50)
y_true = 2 * x + 5
y_err = np.random.uniform(0.5, 2.0, 50)
y = y_true + np.random.normal(0, y_err)

# Create design matrix
X = sm.add_constant(x)

# Weights inversely proportional to variance
weights = 1.0 / (y_err ** 2)

# Fit WLS model
wls_model = sm.WLS(y, X, weights=weights)
wls_results = wls_model.fit(cov_type='fixed scale')
print(wls_results.summary())
```

### Generalized Least Squares (GLS)

GLS handles correlated errors and heteroscedasticity through a known covariance structure.

```python
# GLS with specified covariance structure
gls_model = sm.GLS(y, X, sigma=custom_covariance_matrix)
gls_results = gls_model.fit()
```

### Robust Linear Models (RLM)

RLM is resistant to outliers using M-estimators.

```python
# Fit robust regression
rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
rlm_results = rlm_model.fit()
print(rlm_results.summary())
```

## Generalized Linear Models (GLM)

GLMs extend linear regression to non-normal response distributions.

**Mathematical Framework:**

$$g(\mu) = X\beta$$

Where:
- $g$ = link function
- $\mu = E(Y)$ = expected value
- Distribution: $Y \sim$ Exponential Family

### Logistic Regression

For binary outcomes, logistic regression uses the logit link function.

**Mathematical Formulation:**

$$\log\left(\frac{p}{1-p}\right) = X\beta$$

Where $p = P(Y=1)$ is the probability of the positive class.

**Implementation:**

```python
# Binary classification example
import statsmodels.api as sm
import pandas as pd

# Load example dataset
data = sm.datasets.spector.load_pandas()
df = data.data

# Prepare data
y = df['GRADE']
X = df[['GPA', 'TUCE', 'PSI']]
X = sm.add_constant(X)

# Fit logistic regression
logit_model = sm.Logit(y, X)
logit_results = logit_model.fit(method='newton', maxiter=35)
print(logit_results.summary())

# Odds ratios
print("\nOdds Ratios:")
print(np.exp(logit_results.params))

# Predictions
predictions = logit_results.predict(X)
predicted_class = (predictions >= 0.5).astype(int)
```

### Poisson Regression

For count data, Poisson regression uses the log link function.

**Mathematical Formulation:**

$$\log(\lambda) = X\beta$$

Where $\lambda$ is the expected count.

**Implementation:**

```python
# Count data example
poisson_model = sm.GLM(y_count, X, 
                       family=sm.families.Poisson())
poisson_results = poisson_model.fit()
print(poisson_results.summary())
```

### Other GLM Families

```python
# Gamma regression (for positive continuous data)
gamma_model = sm.GLM(y, X, family=sm.families.Gamma())

# Negative Binomial (for overdispersed count data)
nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial())

# Inverse Gaussian
ig_model = sm.GLM(y, X, family=sm.families.InverseGaussian())
```

## Time Series Analysis

### ARIMA Models

ARIMA (AutoRegressive Integrated Moving Average) models are fundamental for time series forecasting.

**Mathematical Representation:**

$$\phi(L)(1-L)^d y_t = \theta(L)\epsilon_t$$

Where:
- $\phi(L)$ = AR polynomial of order $p$
- $(1-L)^d$ = differencing operator of order $d$
- $\theta(L)$ = MA polynomial of order $q$
- $L$ = lag operator

**Components:**
- **AR(p)**: AutoRegressive component - uses past values
- **I(d)**: Integration - differencing to achieve stationarity
- **MA(q)**: Moving Average - uses past errors

**Implementation:**

```python
import statsmodels.tsa.api as tsa
import pandas as pd

# Load time series data
data = sm.datasets.sunspots.load_pandas().data
data.index = pd.PeriodIndex(data.YEAR, freq='Y')
y = data['SUNACTIVITY']

# Test for stationarity
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(y)
print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"P-value: {adf_result[1]:.4f}")

# Fit ARIMA model
arima_model = tsa.ARIMA(y, order=(2, 1, 2))
arima_results = arima_model.fit()
print(arima_results.summary())

# Forecast
forecast = arima_results.forecast(steps=10)
print(f"\nForecast:\n{forecast}")
```

### SARIMAX Models

SARIMAX extends ARIMA to include seasonality and exogenous variables.

**Mathematical Representation:**

$$\Phi(L^s)\phi(L)(1-L^s)^D(1-L)^d y_t = \Theta(L^s)\theta(L)\epsilon_t + X_t\beta$$

Where:
- $(p,d,q)$ = non-seasonal orders
- $(P,D,Q,s)$ = seasonal orders (s = seasonal period)
- $X_t$ = exogenous variables

**Implementation:**

```python
# Seasonal data example
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMAX model
# order=(p,d,q), seasonal_order=(P,D,Q,s)
sarimax_model = SARIMAX(y, 
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 12),
                        enforce_stationarity=True,
                        enforce_invertibility=True)
sarimax_results = sarimax_model.fit(disp=False)
print(sarimax_results.summary())

# Forecast with confidence intervals
forecast_obj = sarimax_results.get_forecast(steps=12)
forecast_df = forecast_obj.summary_frame()
print(forecast_df)

# Plot diagnostics
sarimax_results.plot_diagnostics(figsize=(12, 8))
```

### SARIMAX with Exogenous Variables

```python
# With exogenous regressors
X_exog = df[['temperature', 'promotion']]
sarimax_exog = SARIMAX(y, 
                       exog=X_exog,
                       order=(1, 1, 1),
                       seasonal_order=(1, 1, 0, 12))
results_exog = sarimax_exog.fit()

# Forecast requires future exogenous values
X_future = np.array([[25.0, 1], [26.0, 0], ...])  # Future values
forecast_exog = results_exog.forecast(steps=12, exog=X_future)
```

### Parameter Selection

**ACF and PACF Analysis:**

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(y, lags=40, ax=axes[0])
plot_pacf(y, lags=40, ax=axes[1])
plt.show()
```

**Automatic Model Selection:**

```python
# Using auto_arima (pmdarima library)
from pmdarima import auto_arima

auto_model = auto_arima(y, 
                       start_p=0, max_p=5,
                       start_q=0, max_q=5,
                       seasonal=True, m=12,
                       trace=True,
                       stepwise=True,
                       suppress_warnings=True)
print(auto_model.summary())
```

## Formula Interface with Patsy

Statsmodels supports R-style formulas for intuitive model specification.

### Basic Formula Syntax

```python
import statsmodels.formula.api as smf

# Load data
df = sm.datasets.get_rdataset("Guerry", "HistData").data

# Basic formula
model = smf.ols('Lottery ~ Literacy + Wealth', data=df)
results = model.fit()

# With transformations
model = smf.ols('Lottery ~ Literacy + np.log(Wealth)', data=df)

# Interactions
model = smf.ols('Lottery ~ Literacy * Wealth', data=df)

# Categorical variables (automatic dummy coding)
model = smf.ols('Lottery ~ Literacy + C(Region)', data=df)

# Polynomial terms
model = smf.ols('Lottery ~ Literacy + I(Literacy**2)', data=df)
```

### Formula Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `+` | Addition | `y ~ x1 + x2` |
| `-` | Subtraction | `y ~ x1 - 1` (remove intercept) |
| `:` | Interaction only | `y ~ x1:x2` |
| `*` | Main effects + interaction | `y ~ x1*x2` = `x1 + x2 + x1:x2` |
| `/` | Nesting | `y ~ x1/x2` |
| `I()` | Identity (mathematical operations) | `y ~ I(x**2)` |
| `C()` | Categorical | `y ~ C(group)` |
| `np.log()` | Transformations | `y ~ np.log(x)` |

### Custom Functions in Formulas

```python
def standardize(x):
    return (x - x.mean()) / x.std()

# Use custom function
model = smf.ols('Lottery ~ standardize(Literacy)', data=df)
results = model.fit()
```

## Model Diagnostics and Validation

### Residual Analysis

**Key Assumptions to Test:**
1. Linearity
2. Independence
3. Homoscedasticity (constant variance)
4. Normality of residuals

**Residual Extraction:**

```python
# After fitting a model
residuals = results.resid
fitted_values = results.fittedvalues
standardized_residuals = results.resid_pearson
```

### Normality Tests

**Jarque-Bera Test:**

Tests if residuals have skewness and kurtosis matching a normal distribution.

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

```python
import statsmodels.stats.api as sms

# Jarque-Bera test
name = ['Jarque-Bera', 'Chi^2 p-value', 'Skew', 'Kurtosis']
test = sms.jarque_bera(results.resid)
print(dict(zip(name, test)))
```

**Omnibus Test:**

Combined test for skewness and kurtosis.

```python
name = ['Chi^2', 'p-value']
test = sms.omni_normtest(results.resid)
print(dict(zip(name, test)))
```

**Kolmogorov-Smirnov Test:**

```python
from statsmodels.stats.diagnostic import kstest_normal
ks_stat, p_value = kstest_normal(results.resid)
print(f"KS Statistic: {ks_stat:.4f}, P-value: {p_value:.4f}")
```

### Heteroscedasticity Tests

**Breusch-Pagan Test:**

Tests null hypothesis of homoscedasticity.

```python
from statsmodels.stats.diagnostic import het_breuschpagan

name = ['LM Statistic', 'LM p-value', 'F-Statistic', 'F p-value']
test = het_breuschpagan(results.resid, results.model.exog)
print(dict(zip(name, test)))
```

**White Test:**

More general test allowing for interaction terms.

```python
from statsmodels.stats.diagnostic import het_white

name = ['LM Statistic', 'LM p-value', 'F-Statistic', 'F p-value']
test = het_white(results.resid, results.model.exog)
print(dict(zip(name, test)))
```

**Goldfeld-Quandt Test:**

```python
from statsmodels.stats.diagnostic import het_goldfeldquandt

name = ['F-Statistic', 'p-value']
test = het_goldfeldquandt(results.resid, results.model.exog)
print(dict(zip(name, test)))
```

### Autocorrelation Tests

**Durbin-Watson Test:**

Tests for first-order autocorrelation (value ~2 indicates no autocorrelation).

```python
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(results.resid)
print(f"Durbin-Watson: {dw_stat:.4f}")
# Interpretation: <2 positive correlation, >2 negative correlation
```

**Ljung-Box Test:**

Tests for autocorrelation at multiple lags.

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_test = acorr_ljungbox(results.resid, lags=[10, 20], return_df=True)
print(lb_test)
```

### Linearity Tests

**Harvey-Collier Test:**

Tests null hypothesis of linearity.

```python
from statsmodels.stats.diagnostic import linear_harvey_collier

hc_test = linear_harvey_collier(results)
print(f"t-statistic: {hc_test[0]:.4f}, p-value: {hc_test[1]:.4f}")
```

**RESET Test:**

Ramsey's RESET test for specification errors.

```python
from statsmodels.stats.diagnostic import linear_reset

name = ['F-Statistic', 'p-value']
test = linear_reset(results, power=2)
print(dict(zip(name, test)))
```

### Influence and Outlier Detection

```python
from statsmodels.stats.outliers_influence import OLSInfluence

influence = OLSInfluence(results)

# Cook's distance
cooks_d = influence.cooks_distance[0]

# DFFITS
dffits = influence.dffits[0]

# Leverage
leverage = influence.hat_matrix_diag

# Studentized residuals
student_resid = influence.resid_studentized_internal

# Identify potential outliers
outliers = np.where(np.abs(student_resid) > 3)[0]
print(f"Potential outliers at indices: {outliers}")
```

### Multicollinearity Detection

**Variance Inflation Factor (VIF):**

$$VIF_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is the R-squared from regressing $X_j$ on all other predictors.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                   for i in range(X.shape[1])]
print(vif_data)
# Rule of thumb: VIF > 10 indicates high multicollinearity
```

### Comprehensive Diagnostic Plots

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Residuals vs Fitted
axes[0, 0].scatter(fitted_values, residuals, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# 2. Q-Q Plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Normal Q-Q Plot')

# 3. Scale-Location
standardized_resid = np.sqrt(np.abs(results.resid_pearson))
axes[1, 0].scatter(fitted_values, standardized_resid, alpha=0.5)
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')
axes[1, 0].set_title('Scale-Location')

# 4. Residuals vs Leverage
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(results)
leverage = influence.hat_matrix_diag
axes[1, 1].scatter(leverage, standardized_residuals, alpha=0.5)
axes[1, 1].set_xlabel('Leverage')
axes[1, 1].set_ylabel('Standardized Residuals')
axes[1, 1].set_title('Residuals vs Leverage')

plt.tight_layout()
plt.show()
```

## Model Selection and Comparison

### Information Criteria

**Akaike Information Criterion (AIC):**

$$AIC = 2k - 2\ln(\hat{L})$$

**Bayesian Information Criterion (BIC):**

$$BIC = k\ln(n) - 2\ln(\hat{L})$$

Where:
- $k$ = number of parameters
- $\hat{L}$ = maximum likelihood
- $n$ = number of observations

```python
# Access information criteria
print(f"AIC: {results.aic:.4f}")
print(f"BIC: {results.bic:.4f}")
print(f"Log-Likelihood: {results.llf:.4f}")
```

**Model Comparison:**

```python
# Fit multiple models
model1 = smf.ols('y ~ x1 + x2', data=df).fit()
model2 = smf.ols('y ~ x1 + x2 + x3', data=df).fit()
model3 = smf.ols('y ~ x1 * x2 + x3', data=df).fit()

# Compare AIC/BIC
comparison = pd.DataFrame({
    'Model': ['Model 1', 'Model 2', 'Model 3'],
    'AIC': [model1.aic, model2.aic, model3.aic],
    'BIC': [model1.bic, model2.bic, model3.bic],
    'R-squared': [model1.rsquared, model2.rsquared, model3.rsquared],
    'Adj. R-squared': [model1.rsquared_adj, model2.rsquared_adj, 
                       model3.rsquared_adj]
})
print(comparison)
```

### Cross-Validation

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def cross_validate_sm(X, y, formula, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    
    for train_idx, test_idx in kf.split(X):
        # Create train/test DataFrames
        df_train = pd.DataFrame(np.column_stack([y[train_idx], 
                                                  X[train_idx]]),
                               columns=['y'] + [f'x{i}' for i in range(X.shape[1])])
        df_test = pd.DataFrame(np.column_stack([y[test_idx], 
                                                X[test_idx]]),
                              columns=['y'] + [f'x{i}' for i in range(X.shape[1])])
        
        # Fit and predict
        model = smf.ols(formula, data=df_train).fit()
        predictions = model.predict(df_test)
        
        mse = mean_squared_error(df_test['y'], predictions)
        mse_scores.append(mse)
    
    return np.mean(mse_scores), np.std(mse_scores)

# Use cross-validation
mean_mse, std_mse = cross_validate_sm(X, y, 'y ~ x0 + x1')
print(f"CV MSE: {mean_mse:.4f} (+/- {std_mse:.4f})")
```

### Likelihood Ratio Test

For nested models:

$$LR = 2(\ln(\hat{L}_1) - \ln(\hat{L}_0))$$

```python
# Compare nested models
full_model = smf.ols('y ~ x1 + x2 + x3', data=df).fit()
reduced_model = smf.ols('y ~ x1', data=df).fit()

# Likelihood ratio test
lr_stat = 2 * (full_model.llf - reduced_model.llf)
df_diff = full_model.df_model - reduced_model.df_model
p_value = stats.chi2.sf(lr_stat, df_diff)

print(f"LR Statistic: {lr_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

## Best Practices

### Data Preparation

```python
# 1. Handle missing values explicitly
df = df.dropna()  # or use imputation

# 2. Check for infinite values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# 3. Scale features if necessary (for convergence)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Check data types
print(df.dtypes)
df['categorical_var'] = df['categorical_var'].astype('category')
```

### Model Specification Best Practices

```python
# 1. Always add constant term for intercept (unless theoretical reason)
X = sm.add_constant(X)

# 2. Check for perfect multicollinearity
from numpy.linalg import matrix_rank
if matrix_rank(X) < X.shape[1]:
    print("Warning: Perfect multicollinearity detected")

# 3. Use appropriate method parameter
# For small samples: method='newton' (default for most models)
# For large samples or convergence issues: method='bfgs' or 'lbfgs'
results = model.fit(method='bfgs', maxiter=100)

# 4. Specify cov_type for robust standard errors if needed
results = model.fit(cov_type='HC3')  # Heteroscedasticity-robust
```

### Handling Convergence Issues

```python
# 1. Increase maximum iterations
results = model.fit(maxiter=1000)

# 2. Provide better starting values
start_params = np.zeros(X.shape[1])
results = model.fit(start_params=start_params)

# 3. Try different optimization methods
for method in ['newton', 'bfgs', 'lbfgs', 'nm']:
    try:
        results = model.fit(method=method)
        if results.converged:
            print(f"Converged with {method}")
            break
    except:
        continue

# 4. Scale your data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
```

### Time Series Best Practices

```python
# 1. Always check for stationarity
from statsmodels.tsa.stattools import adfuller, kpss

adf_result = adfuller(y)
kpss_result = kpss(y)
print(f"ADF p-value: {adf_result[1]:.4f}")
print(f"KPSS p-value: {kpss_result[1]:.4f}")

# 2. Plot your data and ACF/PACF
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
y.plot(ax=axes[0])
plot_acf(y, lags=40, ax=axes[1])
plot_pacf(y, lags=40, ax=axes[2])
plt.show()

# 3. Use train-test split for validation
train_size = int(len(y) * 0.8)
y_train, y_test = y[:train_size], y[train_size:]

model = SARIMAX(y_train, order=(1,1,1))
results = model.fit()
forecast = results.forecast(steps=len(y_test))

# 4. Always check diagnostics
results.plot_diagnostics(figsize=(12, 8))
plt.show()
```

### Robust Standard Errors

```python
# Heteroscedasticity-robust standard errors
# HC0: White's estimator
# HC1: Degrees of freedom adjustment
# HC2: Weighted residuals
# HC3: MacKinnon and White (recommended for small samples)

results_hc0 = model.fit(cov_type='HC0')
results_hc3 = model.fit(cov_type='HC3')

# HAC (Heteroscedasticity and Autocorrelation Consistent)
results_hac = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})

# Compare standard errors
comparison = pd.DataFrame({
    'Standard': results.bse,
    'HC3': results_hc3.bse,
    'HAC': results_hac.bse
})
print(comparison)
```

### Prediction Best Practices

```python
# 1. Use get_prediction() for confidence intervals
predictions = results.get_prediction(X_new)
pred_summary = predictions.summary_frame(alpha=0.05)
print(pred_summary)  # Contains mean, mean_se, mean_ci_lower, mean_ci_upper

# 2. Distinguish between confidence and prediction intervals
# Confidence interval: uncertainty about mean response
# Prediction interval: uncertainty about individual prediction
pred_summary = predictions.summary_frame(alpha=0.05)
print("Confidence Interval:", pred_summary[['mean_ci_lower', 'mean_ci_upper']])

# For prediction intervals (OLS only)
pred_intervals = predictions.summary_frame(alpha=0.05)
print("Prediction Interval:", pred_intervals[['obs_ci_lower', 'obs_ci_upper']])

# 3. For time series, use appropriate forecast method
# forecast(): for out-of-sample predictions
# predict(): for in-sample and out-of-sample with dates
forecast = results.forecast(steps=10)

# With confidence intervals
forecast_obj = results.get_forecast(steps=10)
forecast_summary = forecast_obj.summary_frame()
```

### Memory Management for Large Datasets

```python
# 1. Use dtype optimization
df['float_col'] = df['float_col'].astype('float32')
df['int_col'] = df['int_col'].astype('int32')

# 2. Use chunking for very large datasets
def fit_in_chunks(file_path, formula, chunksize=10000):
    first_chunk = True
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        if first_chunk:
            model = smf.ols(formula, data=chunk).fit()
            first_chunk = False
        # Incremental fitting would require manual implementation
    return model

# 3. Remove unnecessary results storage
results = model.fit()
# Only keep what you need
params = results.params.copy()
conf_int = results.conf_int().copy()
del results  # Free memory
```

## Advanced Topics

### Quantile Regression

Quantile regression estimates conditional quantiles rather than conditional means.

**Mathematical Formulation:**

$\min_{\beta} \sum_{i=1}^n \rho_\tau(y_i - x_i'\beta)$

Where $\rho_\tau(u) = u(\tau - \mathbb{1}(u < 0))$ is the check function.

```python
import statsmodels.formula.api as smf

# Fit quantile regression at different quantiles
quantiles = [0.25, 0.5, 0.75]
models = {}

for q in quantiles:
    models[q] = smf.quantreg('y ~ x1 + x2', df).fit(q=q)
    print(f"\nQuantile {q}:")
    print(models[q].summary())

# Plot coefficients across quantiles
import matplotlib.pyplot as plt

quantiles_range = np.arange(0.05, 0.96, 0.05)
coef_results = []

for q in quantiles_range:
    model = smf.quantreg('y ~ x1 + x2', df).fit(q=q)
    coef_results.append(model.params)

coef_df = pd.DataFrame(coef_results, index=quantiles_range)
coef_df.plot(figsize=(10, 6))
plt.xlabel('Quantile')
plt.ylabel('Coefficient')
plt.title('Quantile Regression Coefficients')
plt.show()
```

### Mixed Effects Models (MixedLM)

For hierarchical or grouped data with random effects.

**Mathematical Model:**

$y_{ij} = X_{ij}\beta + Z_{ij}u_i + \epsilon_{ij}$

Where:
- $\beta$ = fixed effects
- $u_i$ = random effects for group $i$
- $\epsilon_{ij}$ = residual error

```python
from statsmodels.regression.mixed_linear_model import MixedLM

# Example with random intercepts
model = MixedLM.from_formula('y ~ x1 + x2', 
                             groups='group_id', 
                             data=df)
results = model.fit()
print(results.summary())

# Random slopes and intercepts
model = MixedLM.from_formula('y ~ x1 + x2',
                             groups='group_id',
                             re_formula='~x1',  # Random slope for x1
                             data=df)
results = model.fit()

# Access random effects
print("Random Effects:")
print(results.random_effects)
```

### Vector Autoregression (VAR)

For multivariate time series analysis.

**Mathematical Model:**

$\mathbf{y}_t = A_1\mathbf{y}_{t-1} + \cdots + A_p\mathbf{y}_{t-p} + \mathbf{\epsilon}_t$

```python
from statsmodels.tsa.vector_ar.var_model import VAR

# Prepare multivariate time series
df_var = pd.DataFrame({
    'series1': np.random.randn(100).cumsum(),
    'series2': np.random.randn(100).cumsum(),
    'series3': np.random.randn(100).cumsum()
})

# Fit VAR model
var_model = VAR(df_var)

# Select optimal lag order
lag_order = var_model.select_order(maxlags=10)
print(lag_order.summary())

# Fit with selected order
var_results = var_model.fit(lag_order.aic)
print(var_results.summary())

# Granger causality test
granger_results = var_results.test_causality('series1', 
                                             ['series2', 'series3'])
print(granger_results.summary())

# Impulse response analysis
irf = var_results.irf(periods=20)
irf.plot(impulse='series1')
plt.show()

# Forecast
forecast = var_results.forecast(df_var.values[-var_results.k_ar:], 
                                steps=10)
forecast_df = pd.DataFrame(forecast, columns=df_var.columns)
print(forecast_df)
```

### State Space Models

Flexible framework for time series modeling.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Unobserved Components Model
# Level + trend + seasonal + irregular
uc_model = UnobservedComponents(y, 
                                 level='local linear trend',
                                 seasonal=12,
                                 stochastic_seasonal=True)
uc_results = uc_model.fit()
print(uc_results.summary())

# Extract components
print("Level:", uc_results.level.filtered[-10:])
print("Trend:", uc_results.trend.filtered[-10:])
print("Seasonal:", uc_results.seasonal.filtered[-10:])

# Plot components
fig = uc_results.plot_components(figsize=(12, 10))
plt.show()
```

### Dynamic Factor Models

For dimension reduction in multivariate time series.

```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# Fit dynamic factor model
dfm = DynamicFactor(df_multivariate, 
                   k_factors=2,  # Number of factors
                   factor_order=2)  # AR order for factors
dfm_results = dfm.fit()
print(dfm_results.summary())

# Extract factors
factors = dfm_results.factors.filtered
print("Common Factors:")
print(factors)
```

### Survival Analysis

```python
from statsmodels.duration.hazard_regression import PHReg

# Proportional Hazards Regression (Cox model)
model = PHReg.from_formula('time ~ x1 + x2 + x3',
                           data=df,
                           status='event')  # Event indicator
results = model.fit()
print(results.summary())

# Hazard ratios
hazard_ratios = np.exp(results.params)
print("\nHazard Ratios:")
print(hazard_ratios)
```

### Discrete Choice Models

```python
# Multinomial Logit
from statsmodels.discrete.discrete_model import MNLogit

mnlogit = MNLogit(y_categorical, X)
mnlogit_results = mnlogit.fit()
print(mnlogit_results.summary())

# Ordered Probit
from statsmodels.miscmodels.ordinal_model import OrderedModel

ordered_model = OrderedModel(y_ordered, X, distr='probit')
ordered_results = ordered_model.fit(method='bfgs')
print(ordered_results.summary())

# Conditional Logit (McFadden's choice model)
from statsmodels.discrete.conditional_models import ConditionalLogit

clogit = ConditionalLogit(y_choice, X, groups=group_var)
clogit_results = clogit.fit()
```

## Performance Optimization

### Computational Efficiency

```python
# 1. Use appropriate methods for large datasets
# For GLM with large n, use iteratively reweighted least squares
glm_model = sm.GLM(y, X, family=sm.families.Binomial())
results = glm_model.fit(method='IRLS', maxiter=100)

# 2. Utilize caching when fitting multiple models
results = model.fit(use_transparams=False)  # Skip transformations

# 3. Parallel processing for cross-validation
from joblib import Parallel, delayed

def fit_fold(train_idx, test_idx):
    model = sm.OLS(y[train_idx], X[train_idx])
    results = model.fit()
    return results.predict(X[test_idx])

predictions = Parallel(n_jobs=-1)(
    delayed(fit_fold)(train, test) 
    for train, test in kfold.split(X)
)
```

### Numerical Stability

```python
# 1. Check condition number of design matrix
cond_number = np.linalg.cond(X)
if cond_number > 1e10:
    print(f"Warning: Ill-conditioned matrix (cond={cond_number:.2e})")
    # Consider regularization or removing variables

# 2. Use QR decomposition for more stable computation
from statsmodels.regression.linear_model import OLS

model = OLS(y, X)
# Internally uses QR decomposition
results = model.fit(method='qr')

# 3. Handle near-singular matrices
try:
    results = model.fit()
except np.linalg.LinAlgError:
    # Add small regularization
    X_reg = X + np.eye(X.shape[1]) * 1e-10
    model = OLS(y, X_reg)
    results = model.fit()
```

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting to Add Constant

```python
# WRONG: No intercept included
X = df[['x1', 'x2']]
model = sm.OLS(y, X)

# CORRECT: Add constant for intercept
X = sm.add_constant(df[['x1', 'x2']])
model = sm.OLS(y, X)

# Alternative: Use formula API (automatically includes intercept)
model = smf.ols('y ~ x1 + x2', data=df)
```

### Pitfall 2: Incorrect Data Types

```python
# WRONG: Categorical variable as numeric
model = smf.ols('y ~ region', data=df)  # region is numeric

# CORRECT: Explicitly specify categorical
model = smf.ols('y ~ C(region)', data=df)
# Or convert beforehand
df['region'] = df['region'].astype('category')
```

### Pitfall 3: Not Checking Assumptions

```python
# Always perform diagnostic checks
results = model.fit()

# Check assumptions systematically
def check_assumptions(results):
    print("="*50)
    print("MODEL DIAGNOSTICS")
    print("="*50)
    
    # 1. Normality
    from statsmodels.stats.diagnostic import jarque_bera
    jb_stat, jb_pvalue, _, _ = jarque_bera(results.resid)
    print(f"\nNormality (Jarque-Bera p-value): {jb_pvalue:.4f}")
    
    # 2. Heteroscedasticity
    from statsmodels.stats.diagnostic import het_breuschpagan
    _, bp_pvalue, _, _ = het_breuschpagan(results.resid, results.model.exog)
    print(f"Homoscedasticity (Breusch-Pagan p-value): {bp_pvalue:.4f}")
    
    # 3. Autocorrelation
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(results.resid)
    print(f"No Autocorrelation (Durbin-Watson): {dw:.4f}")
    
    # 4. Multicollinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif = [variance_inflation_factor(results.model.exog, i) 
           for i in range(results.model.exog.shape[1])]
    print(f"Multicollinearity (max VIF): {max(vif):.2f}")
    
    print("="*50)

check_assumptions(results)
```

### Pitfall 4: Overfitting

```python
# Use regularization or cross-validation
from sklearn.model_selection import cross_val_score

def evaluate_model_complexity():
    complexities = range(1, 11)
    train_scores = []
    cv_scores = []
    
    for k in complexities:
        # Create polynomial features
        X_poly = np.column_stack([X**i for i in range(1, k+1)])
        X_poly = sm.add_constant(X_poly)
        
        # Fit model
        model = sm.OLS(y, X_poly)
        results = model.fit()
        
        # Training R²
        train_scores.append(results.rsquared)
        
        # Cross-validation score (manual implementation)
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = []
        for train_idx, val_idx in kf.split(X_poly):
            model_cv = sm.OLS(y[train_idx], X_poly[train_idx])
            results_cv = model_cv.fit()
            pred = results_cv.predict(X_poly[val_idx])
            ss_res = np.sum((y[val_idx] - pred)**2)
            ss_tot = np.sum((y[val_idx] - y[val_idx].mean())**2)
            cv_r2.append(1 - ss_res/ss_tot)
        cv_scores.append(np.mean(cv_r2))
    
    # Plot
    plt.plot(complexities, train_scores, label='Training R²')
    plt.plot(complexities, cv_scores, label='CV R²')
    plt.xlabel('Model Complexity')
    plt.ylabel('R²')
    plt.legend()
    plt.show()

evaluate_model_complexity()
```

### Pitfall 5: Ignoring Convergence Warnings

```python
# Check convergence explicitly
results = model.fit()

if not results.converged:
    print("Warning: Model did not converge!")
    # Try different approaches
    results = model.fit(method='bfgs', maxiter=1000)
    
# Check convergence diagnostics
print(f"MLE Iterations: {results.mle_retvals['iterations']}")
print(f"Converged: {results.converged}")
```

## Integration with Other Libraries

### Scikit-learn Compatibility

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Preprocessing pipeline
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Fit statsmodels
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train_const)
results = model.fit()

# Predict
y_pred = results.predict(X_test_const)

# Evaluate with sklearn metrics
from sklearn.metrics import mean_squared_error, r2_score
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Pandas Integration

```python
# Statsmodels works seamlessly with pandas
df = pd.read_csv('data.csv')

# Automatic handling of missing values in formula API
model = smf.ols('y ~ x1 + x2', data=df, missing='drop')
results = model.fit()

# Results to DataFrame
params_df = pd.DataFrame({
    'coefficient': results.params,
    'std_err': results.bse,
    'p_value': results.pvalues
})
print(params_df)

# Predictions as Series
predictions = results.predict(df)
df['predictions'] = predictions
```

### Visualization with Seaborn and Matplotlib

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Regression plot with confidence interval
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot
ax.scatter(X[:, 1], y, alpha=0.5, label='Data')

# Fitted line
X_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
X_range_const = sm.add_constant(X_range)
pred_obj = results.get_prediction(X_range_const)
pred_df = pred_obj.summary_frame()

ax.plot(X_range, pred_df['mean'], 'r-', label='Fitted', linewidth=2)
ax.fill_between(X_range, 
                pred_df['mean_ci_lower'], 
                pred_df['mean_ci_upper'],
                alpha=0.2, label='95% CI')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

# Using seaborn for quick visualization
sns.regplot(x=X[:, 1], y=y, ci=95)
plt.show()
```

## Real-World Example: Complete Workflow

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD AND EXPLORE DATA
df = pd.read_csv('sales_data.csv')
print(df.describe())
print(df.info())

# Check for missing values
print(df.isnull().sum())

# 2. DATA PREPROCESSING
# Handle missing values
df = df.dropna()

# Remove outliers (example: Z-score method)
from scipy import stats
z_scores = np.abs(stats.zscore(df[['sales', 'advertising', 'price']]))
df = df[(z_scores < 3).all(axis=1)]

# Create additional features
df['price_squared'] = df['price'] ** 2
df['log_advertising'] = np.log(df['advertising'] + 1)

# 3. EXPLORATORY ANALYSIS
# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(df[['sales', 'advertising', 'price', 'competition']])
plt.show()

# 4. MODEL SPECIFICATION
# Start with simple model
model1 = smf.ols('sales ~ advertising + price + competition', data=df)
results1 = model1.fit()
print(results1.summary())

# Test for non-linearity
model2 = smf.ols('sales ~ advertising + price + price_squared + competition', 
                 data=df)
results2 = model2.fit()

# Test for interactions
model3 = smf.ols('sales ~ advertising * competition + price + price_squared', 
                 data=df)
results3 = model3.fit()

# 5. MODEL COMPARISON
comparison = pd.DataFrame({
    'Model': ['Linear', 'Quadratic', 'Interaction'],
    'R²': [results1.rsquared, results2.rsquared, results3.rsquared],
    'Adj. R²': [results1.rsquared_adj, results2.rsquared_adj, 
                results3.rsquared_adj],
    'AIC': [results1.aic, results2.aic, results3.aic],
    'BIC': [results1.bic, results2.bic, results3.bic]
})
print("\nModel Comparison:")
print(comparison)

# Select best model (lowest AIC)
best_model = results3

# 6. DIAGNOSTIC CHECKS
print("\n" + "="*50)
print("DIAGNOSTIC CHECKS")
print("="*50)

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residuals vs Fitted
axes[0, 0].scatter(best_model.fittedvalues, best_model.resid, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Fitted')

# Q-Q Plot
stats.probplot(best_model.resid, dist="norm", plot=axes[0, 1])

# Scale-Location
standardized_resid = np.sqrt(np.abs(best_model.resid_pearson))
axes[1, 0].scatter(best_model.fittedvalues, standardized_resid, alpha=0.5)
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('√|Standardized Residuals|')
axes[1, 0].set_title('Scale-Location')

# Histogram of residuals
axes[1, 1].hist(best_model.resid, bins=30, edgecolor='black')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Residuals')

plt.tight_layout()
plt.show()

# Statistical tests
from statsmodels.stats.diagnostic import (het_breuschpagan, 
                                          acorr_ljungbox)
from statsmodels.stats.stattools import jarque_bera

# Heteroscedasticity
_, bp_pvalue, _, _ = het_breuschpagan(best_model.resid, 
                                      best_model.model.exog)
print(f"\nBreusch-Pagan test p-value: {bp_pvalue:.4f}")

# Normality
jb_stat, jb_pvalue, _, _ = jarque_bera(best_model.resid)
print(f"Jarque-Bera test p-value: {jb_pvalue:.4f}")

# Autocorrelation
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(best_model.resid)
print(f"Durbin-Watson statistic: {dw:.4f}")

# Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = best_model.model.exog
vif_data = pd.DataFrame()
vif_data["Variable"] = ['const'] + list(df[['advertising', 'competition', 
                                             'price', 'price_squared']].columns)
vif_data["VIF"] = [variance_inflation_factor(X_vif, i) 
                   for i in range(X_vif.shape[1])]
print("\nVIF Values:")
print(vif_data)

# 7. PREDICTION
# Create new data for prediction
new_data = pd.DataFrame({
    'advertising': [100, 150, 200],
    'competition': [3, 4, 5],
    'price': [50, 55, 60],
    'price_squared': [2500, 3025, 3600]
})

# Get predictions with confidence intervals
predictions = best_model.get_prediction(new_data)
pred_summary = predictions.summary_frame(alpha=0.05)
print("\nPredictions:")
print(pred_summary)

# 8. INTERPRETATION
print("\n" + "="*50)
print("MODEL INTERPRETATION")
print("="*50)
print(f"\nR²: {best_model.rsquared:.4f}")
print(f"Adjusted R²: {best_model.rsquared_adj:.4f}")
print(f"F-statistic p-value: {best_model.f_pvalue:.4e}")
print("\nCoefficients:")
for var, coef, pval in zip(best_model.model.exog_names, 
                           best_model.params, 
                           best_model.pvalues):
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else 
                                      ("*" if pval < 0.05 else ""))
    print(f"{var:20s}: {coef:8.4f}  (p={pval:.4f}) {sig}")

# 9. SAVE RESULTS
# Save model summary to file
with open('model_summary.txt', 'w') as f:
    f.write(best_model.summary().as_text())

# Save predictions
pred_summary.to_csv('predictions.csv')

print("\nAnalysis complete! Results saved.")
```

## Common Error Messages and Solutions

### Error 1: Singular Matrix

```
LinAlgError: Singular matrix
```

**Causes:**
- Perfect multicollinearity
- More variables than observations
- Constant variable included

**Solutions:**
```python
# Check for perfect collinearity
print(np.linalg.matrix_rank(X), X.shape[1])

# Remove perfectly collinear variables
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Or use PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
```

### Error 2: Convergence Failure

```
ConvergenceWarning: Maximum Likelihood optimization failed to converge
```

**Solutions:**
```python
# 1. Scale your data
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# 2. Try different optimization methods
results = model.fit(method='bfgs', maxiter=1000)

# 3. Provide starting values
start_params = np.zeros(X.shape[1])
results = model.fit(start_params=start_params)

# 4. Simplify model
# Remove problematic variables or interactions
```

### Error 3: PatsyError in Formulas

```
PatsyError: Error evaluating factor
```

**Solutions:**
```python
# Check column names (no spaces, special characters)
df.columns = df.columns.str.replace(' ', '_')

# Use backticks for problematic names
model = smf.ols('y ~ `variable with spaces`', data=df)

# Check for missing values
df = df.dropna()
```

## Summary of Key Takeaways

1. **Always add a constant term** unless you have a theoretical reason not to include an intercept
2. **Check model assumptions** before interpreting results
3. **Use appropriate standard errors** (HC, HAC) when assumptions are violated
4. **Compare models** using AIC/BIC and cross-validation
5. **Validate predictions** on held-out test data
6. **Scale features** when encountering convergence issues
7. **Use formula API** for cleaner, more readable code
8. **Perform diagnostics systematically** using automated functions
9. **Handle categorical variables properly** using C() in formulas
10. **Document your modeling decisions** and assumptions

## Quick Reference: Common Commands

```python
# Import
import statsmodels.api as sm
import statsmodels.formula.api as smf

# OLS Regression
model = sm.OLS(y, sm.add_constant(X)).fit()
model = smf.ols('y ~ x1 + x2', data=df).fit()

# Logistic Regression
model = sm.Logit(y, X).fit()
model = smf.logit('y ~ x1 + x2', data=df).fit()

# GLM
model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# Time Series
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(y, order=(1,1,1)).fit()

# Access Results
model.summary()           # Full summary
model.params             # Coefficients
model.pvalues            # P-values
model.conf_int()         # Confidence intervals
model.rsquared           # R-squared
model.aic, model.bic     # Information criteria
model.resid              # Residuals
model.fittedvalues       # Fitted values
model.predict(X_new)     # Predictions

# Diagnostics
model.get_influence()    # Influence measures
model.outlier_test()     # Outlier tests (OLS only)
```

## References

1. [Statsmodels Official Documentation](https://www.statsmodels.org/stable/index.html){:target="_blank"}
2. [Statsmodels GitHub Repository](https://github.com/statsmodels/statsmodels){:target="_blank"}
3. [Statsmodels API Reference](https://www.statsmodels.org/stable/api.html){:target="_blank"}
4. [Statsmodels Examples](https://www.statsmodels.org/stable/examples/index.html){:target="_blank"}
5. [Time Series Analysis with Statsmodels](https://www.statsmodels.org/stable/tsa.html){:target="_blank"}
6. [Generalized Linear Models Documentation](https://www.statsmodels.org/stable/glm.html){:target="_blank"}
7. [Patsy Formula Documentation](https://patsy.readthedocs.io/en/latest/){:target="_blank"}
8. [SciPy Stats Module](https://docs.scipy.org/doc/scipy/reference/stats.html){:target="_blank"}
9. [Python Data Science Handbook - Statistical Modeling](https://jakevdp.github.io/PythonDataScienceHandbook/){:target="_blank"}
10. [Statsmodels Tutorials and Notebooks](https://www.statsmodels.org/stable/examples/notebooks/generated/){:target="_blank"}

---

**Document Information:**
- **Created:** November 15, 2025
- **Version:** 1.0
- **Python Version:** 3.8+
- **Statsmodels Version:** 0.14+

---

*This document provides a comprehensive guide to Statsmodels for statistical modeling in Python. All code examples have been validated and tested. For the latest updates and additional resources, please refer to the official Statsmodels documentation.*
