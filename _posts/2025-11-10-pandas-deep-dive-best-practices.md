---
layout: post
title: "🌊 Pandas: Deep Dive & Best Practices"
author: technical_notes
date: 2025-11-10 7:30:00 +0530
categories: [Notes, Pandas]
tags: [Pandas, Machine Learning, Python, Data Analysis, Data Science, Data Cleaning, Optimization, Best Practices]
description: "Concise, clear, and validated revision notes on Pandas — structured for beginners and practitioners."
image: /assets/img/posts/pandas-logo.png
toc: true
math: true
mermaid: true
---
# Pandas: Deep Dive & Best Practices

## Introduction

Pandas is the cornerstone library for data manipulation and analysis in Python. Built on top of NumPy, it provides high-performance, easy-to-use data structures and data analysis tools that bridge the gap between raw data and machine learning models. This comprehensive guide explores the Pandas ecosystem from foundational concepts to advanced optimization techniques, empowering you to write efficient, production-ready data pipelines.

---

## Table 1: Pandas Lifecycle & Operation Terminology Mapping

Different contexts use varying terminology for similar Pandas operations. This table maps equivalent terms across documentation, tutorials, and industry practices:

| Standard Term | Alternative Names | Context/Usage |
|--------------|-------------------|---------------|
| **Data Loading** | Data Ingestion, Data Import, Data Reading, I/O Operations | Initial data acquisition stage |
| **Data Exploration** | Exploratory Data Analysis (EDA), Data Profiling, Initial Analysis | Understanding data structure and content |
| **Data Cleaning** | Data Wrangling, Data Munging, Data Preprocessing, Data Preparation | Fixing inconsistencies and errors |
| **Data Transformation** | Feature Engineering, Data Reshaping, Data Manipulation | Modifying data structure or values |
| **Missing Value Handling** | Null Handling, NaN Management, Imputation | Dealing with incomplete data |
| **Data Filtering** | Data Selection, Subsetting, Querying, Indexing | Selecting specific rows/columns |
| **Aggregation** | Summarization, GroupBy Operations, Roll-up | Computing summary statistics |
| **Merging** | Joining, Combining, Concatenating | Integrating multiple datasets |
| **Vectorization** | Broadcast Operations, Array Operations, Bulk Operations | Element-wise operations without loops |
| **Memory Optimization** | Downcasting, Type Optimization, Memory Management | Reducing RAM usage |
| **Pipeline Construction** | Chaining, Method Chaining, Workflow Building | Sequential operation composition |
| **Data Export** | Data Serialization, Data Writing, Data Output | Saving processed data |

---

## Table 2: Hierarchical Pandas Concept Structure

This table organizes Pandas concepts from high-level abstractions to specific implementations:

| Level | Category | Term | Parent Concept | Description |
|-------|----------|------|----------------|-------------|
| **L1** | Architecture | Pandas Library | - | Top-level data analysis toolkit |
| **L2** | Data Structures | Series | Pandas Library | One-dimensional labeled array |
| **L2** | Data Structures | DataFrame | Pandas Library | Two-dimensional labeled data structure |
| **L2** | Data Structures | Index | Pandas Library | Immutable sequence for axis labeling |
| **L3** | DataFrame Components | Columns | DataFrame | Vertical data slices (Series collection) |
| **L3** | DataFrame Components | Rows | DataFrame | Horizontal data slices |
| **L3** | DataFrame Components | Index | DataFrame | Row labels |
| **L3** | DataFrame Components | Values | DataFrame | Underlying NumPy array |
| **L4** | Data Types | Numeric Types | DataFrame | int64, float64, Int64, Float64 |
| **L4** | Data Types | String Types | DataFrame | object, string, StringDtype |
| **L4** | Data Types | Categorical | DataFrame | category |
| **L4** | Data Types | DateTime Types | DataFrame | datetime64, timedelta64, Period |
| **L4** | Data Types | Boolean Types | DataFrame | bool, boolean |
| **L4** | Data Types | Nullable Types | DataFrame | Int64, Float64, boolean, string |
| **L5** | Operations | Indexing | DataFrame | loc, iloc, at, iat, [] operator |
| **L5** | Operations | Selection | DataFrame | Filtering, querying, masking |
| **L5** | Operations | Transformation | DataFrame | apply, map, applymap, pipe |
| **L5** | Operations | Aggregation | DataFrame | sum, mean, groupby, agg |
| **L5** | Operations | Reshaping | DataFrame | pivot, melt, stack, unstack |
| **L5** | Operations | Merging | DataFrame | merge, join, concat, append |
| **L6** | Indexing Methods | Label-based | Indexing | loc (inclusive slicing) |
| **L6** | Indexing Methods | Position-based | Indexing | iloc (exclusive slicing) |
| **L6** | Indexing Methods | Scalar Access | Indexing | at, iat (single value) |
| **L6** | Indexing Methods | Boolean Indexing | Indexing | Conditional selection |
| **L6** | Missing Data | NaN (NumPy) | Missing Values | np.nan for floats |
| **L6** | Missing Data | None | Missing Values | Python None for objects |
| **L6** | Missing Data | NaT | Missing Values | Not-a-Time for datetime |
| **L6** | Missing Data | pd.NA | Missing Values | Pandas NA for nullable types |
| **L6** | GroupBy Operations | Split | Aggregation | Divide data into groups |
| **L6** | GroupBy Operations | Apply | Aggregation | Compute function on groups |
| **L6** | GroupBy Operations | Combine | Aggregation | Merge results back |
| **L6** | Merge Types | Inner Join | Merging | Intersection of keys |
| **L6** | Merge Types | Left Join | Merging | All left keys + matches |
| **L6** | Merge Types | Right Join | Merging | All right keys + matches |
| **L6** | Merge Types | Outer Join | Merging | Union of all keys |
| **L6** | Merge Types | Cross Join | Merging | Cartesian product |
| **L7** | Optimization | Vectorization | Performance | NumPy array operations |
| **L7** | Optimization | Chunking | Performance | Process data in batches |
| **L7** | Optimization | Cython/Numba | Performance | Compiled extensions |
| **L7** | Optimization | Parallel Processing | Performance | Multicore utilization |
| **L7** | Optimization | Memory Dtypes | Performance | Optimal type selection |
| **L7** | File Formats | CSV | I/O | Comma-separated values |
| **L7** | File Formats | Parquet | I/O | Columnar storage format |
| **L7** | File Formats | HDF5 | I/O | Hierarchical data format |
| **L7** | File Formats | JSON | I/O | JavaScript object notation |
| **L7** | File Formats | Excel | I/O | Spreadsheet format |
| **L7** | File Formats | SQL | I/O | Relational database |
| **L7** | File Formats | Pickle | I/O | Python object serialization |

---

## Pandas Data Processing Lifecycle

Understanding the typical workflow helps organize your code and think systematically about data manipulation.

### Complete Lifecycle Overview

```mermaid
graph TD
    A[Data Source] --> B[Data Loading]
    B --> C[Data Exploration]
    C --> D[Data Cleaning]
    D --> E[Data Transformation]
    E --> F[Data Analysis/Aggregation]
    F --> G[Data Visualization/Modeling]
    G --> H[Data Export/Persistence]
    
    D --> D1[Handle Missing Values]
    D --> D2[Remove Duplicates]
    D --> D3[Fix Data Types]
    D --> D4[Handle Outliers]
    
    E --> E1[Feature Engineering]
    E --> E2[Encoding Categories]
    E --> E3[Scaling/Normalization]
    E --> E4[Reshaping Data]
    
    F --> F1[GroupBy Operations]
    F --> F2[Pivot Tables]
    F --> F3[Statistical Analysis]
    
    style A fill:#e1f5ff
    style H fill:#e1f5ff
    style D fill:#ffe1e1
    style E fill:#fff4e1
    style F fill:#e1ffe1
```

---

## Phase 1: Data Loading and I/O Operations

### 1.1 Reading Data from Various Sources

**Reading CSV Files:**

```python
import pandas as pd

# Basic CSV reading
df = pd.read_csv('data.csv')

# Optimized CSV reading with specifications
df = pd.read_csv(
    'large_data.csv',
    usecols=['col1', 'col2', 'col3'],  # Load only needed columns
    dtype={'col1': 'int32', 'col2': 'float32'},  # Specify dtypes
    parse_dates=['date_column'],  # Parse dates during load
    na_values=['NA', 'missing', ''],  # Custom NA values
    nrows=10000,  # Limit rows for testing
    chunksize=5000  # Iterator for large files
)
```

**Reading with Chunking for Large Files:**

```python
# Process large CSV in chunks
chunk_iter = pd.read_csv('huge_file.csv', chunksize=10000)

results = []
for chunk in chunk_iter:
    # Process each chunk
    processed = chunk[chunk['value'] > 100]
    results.append(processed)

# Combine all results
df = pd.concat(results, ignore_index=True)
```

**Reading Other Formats:**

```python
# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1', engine='openpyxl')

# JSON
df = pd.read_json('data.json', orient='records')

# Parquet (recommended for large datasets)
df = pd.read_parquet('data.parquet')

# SQL Database
import sqlalchemy as sa
engine = sa.create_engine('postgresql://user:pass@localhost/db')
df = pd.read_sql('SELECT * FROM table', engine)

# HDF5 (hierarchical data format)
df = pd.read_hdf('data.h5', key='df')

# Pickle (Python serialization)
df = pd.read_pickle('data.pkl')
```

### 1.2 File Format Performance Comparison

| Format | Read Speed | Write Speed | Compression | Use Case |
|--------|-----------|-------------|-------------|----------|
| CSV | Slow | Slow | Low | Human-readable, universal |
| Parquet | Fast | Fast | Excellent | Large datasets, analytics |
| HDF5 | Very Fast | Fast | Good | Time series, append operations |
| Pickle | Very Fast | Very Fast | Medium | Python-only, quick save/load |
| JSON | Slow | Slow | Low | APIs, nested structures |
| Excel | Very Slow | Very Slow | Low | Business reporting |

**Best Practice: Use Parquet for Production**

```python
# Writing to Parquet with compression
df.to_parquet(
    'output.parquet',
    compression='snappy',  # Fast compression
    engine='pyarrow',
    index=False
)

# Reading Parquet is 10-100x faster than CSV
df = pd.read_parquet('output.parquet')
```

---

## Phase 2: Data Exploration

### 2.1 Understanding Your Data

```python
# Basic information
df.info()  # Columns, dtypes, non-null counts, memory usage
df.describe()  # Statistical summary for numeric columns
df.describe(include='all')  # Include object types

# Data structure
df.shape  # (rows, columns)
df.columns  # Column names
df.dtypes  # Data types
df.index  # Index information

# Preview data
df.head(10)  # First 10 rows
df.tail(10)  # Last 10 rows
df.sample(5)  # Random 5 rows

# Memory usage
df.memory_usage(deep=True)  # Actual memory per column
df.memory_usage(deep=True).sum() / 1024**2  # Total MB

# Unique values
df['column'].nunique()  # Count unique
df['column'].unique()  # Array of unique values
df['column'].value_counts()  # Frequency distribution
df['column'].value_counts(normalize=True)  # Proportions
```

### 2.2 Data Profiling

```python
# Missing value analysis
df.isnull().sum()  # Count nulls per column
df.isnull().sum() / len(df) * 100  # Percentage nulls

# Correlation analysis
df.corr()  # Numeric correlation matrix
df.corr()['target'].sort_values(ascending=False)  # Correlations with target

# Duplicate analysis
df.duplicated().sum()  # Count duplicates
df[df.duplicated(keep=False)]  # View all duplicates

# Data quality checks
def data_quality_report(df):
    """Generate comprehensive data quality report"""
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null_count': df.count(),
        'null_count': df.isnull().sum(),
        'null_percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_count': df.nunique(),
        'memory_mb': df.memory_usage(deep=True) / 1024**2
    })
    return report

print(data_quality_report(df))
```

---

## Phase 3: Data Cleaning

### 3.1 Handling Missing Values

**Understanding Pandas Missing Value Representations:**

```python
import numpy as np

# Different missing value types
df = pd.DataFrame({
    'float_col': [1.0, np.nan, 3.0],       # np.nan for floats
    'int_col': [1, pd.NA, 3],              # pd.NA for nullable integers
    'datetime_col': [pd.Timestamp('2024-01-01'), pd.NaT, pd.Timestamp('2024-01-03')],  # NaT
    'object_col': ['a', None, 'c']         # None for objects
})
```

**Detecting Missing Values:**

```python
# Check for missing values
df.isnull()  # Boolean DataFrame
df.isna()    # Alias for isnull()
df.notna()   # Opposite of isna()

# Missing value locations
df[df['column'].isnull()]  # Rows with nulls in specific column

# Any/all checks
df.isnull().any()  # Columns with any nulls
df.isnull().all()  # Columns with all nulls
```

**Dropping Missing Values:**

```python
# Drop rows with any nulls
df_clean = df.dropna()

# Drop rows where all values are null
df_clean = df.dropna(how='all')

# Drop rows with nulls in specific columns
df_clean = df.dropna(subset=['col1', 'col2'])

# Drop columns with any nulls
df_clean = df.dropna(axis=1)

# Drop columns with more than 50% nulls
threshold = len(df) * 0.5
df_clean = df.dropna(axis=1, thresh=threshold)

# Inplace modification
df.dropna(inplace=True)
```

**Filling Missing Values:**

```python
# Fill with constant
df['column'].fillna(0, inplace=True)

# Fill with mean/median/mode
df['numeric_col'].fillna(df['numeric_col'].mean(), inplace=True)
df['numeric_col'].fillna(df['numeric_col'].median(), inplace=True)
df['category_col'].fillna(df['category_col'].mode()[0], inplace=True)

# Forward fill (propagate last valid value)
df['column'].fillna(method='ffill', inplace=True)  # or 'pad'

# Backward fill (use next valid value)
df['column'].fillna(method='bfill', inplace=True)  # or 'backfill'

# Fill with interpolation
df['column'].interpolate(method='linear', inplace=True)
df['column'].interpolate(method='polynomial', order=2, inplace=True)
df['column'].interpolate(method='time', inplace=True)  # For time series

# Fill different columns with different strategies
fill_values = {
    'col1': 0,
    'col2': df['col2'].mean(),
    'col3': 'Unknown'
}
df.fillna(fill_values, inplace=True)

# Advanced: Fill with group mean
df['value'] = df.groupby('category')['value'].transform(
    lambda x: x.fillna(x.mean())
)
```

### 3.2 Handling Duplicates

```python
# Identify duplicates
df.duplicated()  # Boolean Series
df.duplicated(keep='first')  # Mark all except first occurrence
df.duplicated(keep='last')   # Mark all except last occurrence
df.duplicated(keep=False)    # Mark all duplicates

# Check specific columns for duplicates
df.duplicated(subset=['col1', 'col2'])

# View duplicate rows
duplicates = df[df.duplicated(keep=False)]

# Remove duplicates
df_clean = df.drop_duplicates()  # Keep first occurrence

# Remove based on specific columns
df_clean = df.drop_duplicates(subset=['col1', 'col2'], keep='last')

# Inplace removal
df.drop_duplicates(inplace=True)

# Count duplicates before removal
print(f"Duplicates found: {df.duplicated().sum()}")
```

### 3.3 Data Type Conversion

**Converting Types:**

```python
# Basic type conversion
df['int_col'] = df['int_col'].astype('int64')
df['float_col'] = df['float_col'].astype('float32')
df['str_col'] = df['str_col'].astype('string')

# Handle errors during conversion
df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')  # Invalid → NaN
df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='ignore')  # Keep original

# Downcast for memory optimization
df['int_col'] = pd.to_numeric(df['int_col'], downcast='integer')  # Smallest int type
df['float_col'] = pd.to_numeric(df['float_col'], downcast='float')  # float32 if possible

# DateTime conversion
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Invalid → NaT
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)  # Auto-detect

# Custom datetime format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S')

# Boolean conversion
df['bool_col'] = df['bool_col'].astype('boolean')  # Nullable boolean

# Category conversion (major memory savings)
df['category_col'] = df['category_col'].astype('category')

# Automatic type inference
df_converted = df.convert_dtypes()  # Uses nullable dtypes

# Example: Comprehensive type optimization
def optimize_dtypes(df):
    """Automatically optimize DataFrame dtypes"""
    # Numeric optimization
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Object to category (if cardinality < 50%)
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype('category')
    
    return df

df_optimized = optimize_dtypes(df.copy())
```

**Memory Comparison:**

```python
# Before and after memory usage
def memory_usage_comparison(df_before, df_after):
    """Compare memory usage before and after optimization"""
    before_mb = df_before.memory_usage(deep=True).sum() / 1024**2
    after_mb = df_after.memory_usage(deep=True).sum() / 1024**2
    savings = (1 - after_mb / before_mb) * 100
    
    print(f"Before: {before_mb:.2f} MB")
    print(f"After: {after_mb:.2f} MB")
    print(f"Savings: {savings:.2f}%")

memory_usage_comparison(df, df_optimized)
```

### 3.4 Handling Outliers

```python
# Z-score method
from scipy import stats

z_scores = np.abs(stats.zscore(df['numeric_col']))
df_no_outliers = df[z_scores < 3]

# IQR method
Q1 = df['numeric_col'].quantile(0.25)
Q3 = df['numeric_col'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_no_outliers = df[
    (df['numeric_col'] >= lower_bound) & 
    (df['numeric_col'] <= upper_bound)
]

# Clip values to bounds
df['numeric_col'] = df['numeric_col'].clip(lower=lower_bound, upper=upper_bound)

# Winsorization (cap at percentiles)
lower_percentile = df['numeric_col'].quantile(0.05)
upper_percentile = df['numeric_col'].quantile(0.95)
df['numeric_col'] = df['numeric_col'].clip(
    lower=lower_percentile,
    upper=upper_percentile
)
```

---

## Phase 4: Data Selection and Indexing

### 4.1 Selection Methods

**Basic Selection:**

```python
# Single column (returns Series)
df['column_name']
df.column_name  # Not recommended for names with spaces or special chars

# Multiple columns (returns DataFrame)
df[['col1', 'col2', 'col3']]

# Rows by position
df[0:5]  # First 5 rows (slicing)

# Boolean indexing
df[df['age'] > 25]
df[df['age'].between(20, 30)]
df[df['name'].str.contains('John')]

# Multiple conditions
df[(df['age'] > 25) & (df['city'] == 'NYC')]  # AND
df[(df['age'] < 20) | (df['age'] > 60)]       # OR
df[~(df['age'] > 25)]                         # NOT

# isin for membership
df[df['city'].isin(['NYC', 'LA', 'Chicago'])]
df[~df['city'].isin(['NYC', 'LA'])]  # NOT in list
```

### 4.2 loc and iloc

**loc (Label-based):**

```python
# Single value
df.loc[0, 'column']  # Row label 0, column 'column'

# Slicing (INCLUSIVE on both ends)
df.loc[0:5, 'column1':'column3']

# Boolean indexing with loc
df.loc[df['age'] > 25, ['name', 'age']]

# Set values with loc
df.loc[df['age'] > 25, 'category'] = 'adult'

# All rows, specific columns
df.loc[:, ['col1', 'col2']]

# Callable for complex logic
df.loc[lambda df: df['value'] > df['value'].mean()]
```

**iloc (Position-based):**

```python
# Single value
df.iloc[0, 1]  # First row, second column

# Slicing (EXCLUSIVE on end)
df.iloc[0:5, 0:3]  # Rows 0-4, columns 0-2

# Negative indexing
df.iloc[-5:]  # Last 5 rows
df.iloc[:, -2:]  # Last 2 columns

# Mixed indexing
df.iloc[[0, 2, 4], [1, 3]]  # Specific rows and columns

# Boolean array
mask = df['age'] > 25
df.iloc[mask.values, :]
```

### 4.3 Fast Scalar Access

```python
# at for label-based (faster than loc for scalar)
value = df.at[0, 'column']
df.at[0, 'column'] = new_value

# iat for position-based (faster than iloc for scalar)
value = df.iat[0, 1]
df.iat[0, 1] = new_value

# Performance comparison (accessing single value 1M times)
# iat: ~0.5s
# iloc: ~2.0s
# at: ~0.5s
# loc: ~2.0s
```

### 4.4 Query Method

```python
# SQL-like querying
df.query('age > 25 and city == "NYC"')

# Variable substitution
min_age = 25
df.query('age > @min_age')

# Column names with spaces (use backticks)
df.query('`column name` > 100')

# String operations
df.query('name.str.contains("John")', engine='python')

# Multiple conditions
df.query('(age > 20 & age < 30) | (income > 100000)')

# In operator
cities = ['NYC', 'LA']
df.query('city in @cities')
```

---

## Phase 5: Data Transformation

### 5.1 Apply, Map, and Lambda Functions

**apply() - Apply function along axis:**

```python
# Apply to Series (column)
df['squared'] = df['value'].apply(lambda x: x ** 2)
df['category'] = df['score'].apply(lambda x: 'high' if x > 70 else 'low')

# Apply to DataFrame (row-wise)
df['total'] = df.apply(lambda row: row['col1'] + row['col2'], axis=1)

# Apply to DataFrame (column-wise)
df_normalized = df.apply(lambda col: (col - col.mean()) / col.std())

# Apply custom function
def categorize_age(age):
    if age < 18:
        return 'minor'
    elif age < 65:
        return 'adult'
    else:
        return 'senior'

df['age_group'] = df['age'].apply(categorize_age)

# Apply with additional arguments
df['adjusted'] = df['value'].apply(lambda x: x * factor, factor=1.1)
```

**map() - Element-wise mapping (Series only):**

```python
# Map with dictionary
gender_map = {'M': 'Male', 'F': 'Female'}
df['gender_full'] = df['gender'].map(gender_map)

# Map with Series
avg_scores = df.groupby('student_id')['score'].mean()
df['avg_score'] = df['student_id'].map(avg_scores)

# Map with function
df['upper_name'] = df['name'].map(str.upper)
df['doubled'] = df['value'].map(lambda x: x * 2)
```

**applymap() - Element-wise for entire DataFrame:**

```python
# Apply function to every cell
df_rounded = df.applymap(lambda x: round(x, 2))

# Format all values
df_formatted = df.select_dtypes(include=[np.number]).applymap('${:,.2f}'.format)
```

**Performance Comparison:**

```python
import time

# Method 1: Loop (SLOWEST - NEVER USE)
start = time.time()
for idx in df.index:
    df.loc[idx, 'new_col'] = df.loc[idx, 'col1'] * 2
print(f"Loop: {time.time() - start:.4f}s")

# Method 2: apply()
start = time.time()
df['new_col'] = df['col1'].apply(lambda x: x * 2)
print(f"Apply: {time.time() - start:.4f}s")

# Method 3: Vectorization (FASTEST)
start = time.time()
df['new_col'] = df['col1'] * 2
print(f"Vectorized: {time.time() - start:.4f}s")

# Typical results:
# Loop: 5.2341s (100x slower)
# Apply: 0.0891s (10x slower)
# Vectorized: 0.0089s (baseline)
```

### 5.2 Vectorization Best Practices

**Always Prefer Vectorized Operations:**

```python
# BAD: Using loops or apply unnecessarily
df['result'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# GOOD: Vectorized operation
df['result'] = df['a'] + df['b']

# BAD: Loop through rows
total = 0
for idx, row in df.iterrows():
    total += row['value']

# GOOD: Use built-in methods
total = df['value'].sum()

# BAD: Apply with simple arithmetic
df['normalized'] = df['value'].apply(lambda x: (x - df['value'].mean()) / df['value'].std())

# GOOD: Vectorized normalization
df['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
```

**Complex Conditions with np.where and np.select:**

```python
import numpy as np

# Simple if-else
df['category'] = np.where(df['value'] > 100, 'high', 'low')

# Nested conditions
df['category'] = np.where(
    df['value'] > 100, 
    'high', 
    np.where(df['value'] > 50, 'medium', 'low')
)

# Multiple conditions (better approach)
conditions = [
    df['value'] > 100,
    df['value'] > 50,
    df['value'] > 0
]
choices = ['high', 'medium', 'low']
df['category'] = np.select(conditions, choices, default='invalid')

# Complex logic
df['bonus'] = np.select(
    [
        (df['sales'] > 100000) & (df['years'] > 5),
        (df['sales'] > 100000) & (df['years'] <= 5),
        (df['sales'] > 50000)
    ],
    [
        df['salary'] * 0.20,
        df['salary'] * 0.15,
        df['salary'] * 0.10
    ],
    default=0
)
```

### 5.3 String Operations

```python
# String methods (vectorized)
df['upper'] = df['name'].str.upper()
df['lower'] = df['name'].str.lower()
df['title'] = df['name'].str.title()
df['length'] = df['name'].str.len()

# String search
df[df['name'].str.contains('John')]
df[df['name'].str.contains('john', case=False)]
df[df['name'].str.contains(r'\d+', regex=True)]  # Contains digits

# String replace
df['cleaned'] = df['text'].str.replace('[^a-zA-Z]', '', regex=True)
df['formatted'] = df['phone'].str.replace(r'(\d{3})(\d{3})(\d{4})', r'(\1) \2-\3', regex=True)

# String splitting
df[['first_name', 'last_name']] = df['full_name'].str.split(' ', expand=True)
df['parts'] = df['text'].str.split(',')  # Returns list

# Extract with regex
df['area_code'] = df['phone'].str.extract(r'(\d{3})')
df[['year', 'month', 'day']] = df['date_str'].str.extract(r'(\d{4})-(\d{2})-(\d{2})')

# Strip whitespace
df['cleaned'] = df['text'].str.strip()
df['cleaned'] = df['text'].str.lstrip()
df['cleaned'] = df['text'].str.rstrip()

# Padding
df['padded'] = df['id'].str.zfill(5)  # Pad with zeros
df['padded'] = df['name'].str.pad(20, side='right', fillchar=' ')

# String slicing
df['first_char'] = df['name'].str[0]
df['first_three'] = df['name'].str[:3]

# Join strings
df['combined'] = df['first_name'].str.cat(df['last_name'], sep=' ')
```

### 5.4 DateTime Operations

```python
# Creating datetime
df['date'] = pd.to_datetime(df['date_str'])
df['date'] = pd.to_datetime(df['date_str'], format='%Y-%m-%d')

# Extracting components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df['dayofweek'] = df['date'].dt.dayofweek  # Monday=0
df['day_name'] = df['date'].dt.day_name()
df['month_name'] = df['date'].dt.month_name()
df['quarter'] = df['date'].dt.quarter
df['is_month_end'] = df['date'].dt.is_month_end
df['is_month_start'] = df['date'].dt.is_month_start

# Date arithmetic
df['tomorrow'] = df['date'] + pd.Timedelta(days=1)
df['next_week'] = df['date'] + pd.Timedelta(weeks=1)
df['next_month'] = df['date'] + pd.DateOffset(months=1)

# Time differences
df['age_days'] = (pd.Timestamp.now() - df['birth_date']).dt.days
df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()

# Formatting
df['formatted'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Time zones
df['date_utc'] = df['date'].dt.tz_localize('UTC')
df['date_est'] = df['date_utc'].dt.tz_convert('US/Eastern')

# Date ranges
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
date_range = pd.date_range(start='2024-01-01', periods=365, freq='D')

# Business days
business_days = pd.bdate_range(start='2024-01-01', end='2024-12-31')

# Resampling time series
df.set_index('date', inplace=True)
df_daily = df.resample('D').mean()  # Daily averages
df_monthly = df.resample('M').sum()  # Monthly sums
df_weekly = df.resample('W').agg({'col1': 'sum', 'col2': 'mean'})
```

---

## Phase 6: Grouping and Aggregation

### 6.1 GroupBy Operations

**Understanding Split-Apply-Combine:**

```python
# Basic groupby
grouped = df.groupby('category')

# Single aggregation
result = df.groupby('category')['value'].sum()
result = df.groupby('category')['value'].mean()
result = df.groupby('category')['value'].count()

# Multiple columns groupby
result = df.groupby(['category', 'subcategory'])['value'].sum()

# Multiple aggregations
result = df.groupby('category').agg({
    'value': 'sum',
    'quantity': 'mean',
    'price': ['min', 'max', 'median']
})

# Flatten multi-level columns
result.columns = ['_'.join(col).strip() for col in result.columns.values]

# Named aggregations (Pandas 0.25+)
result = df.groupby('category').agg(
    total_value=('value', 'sum'),
    avg_quantity=('quantity', 'mean'),
    max_price=('price', 'max'),
    count=('value', 'count')
)

# Custom aggregation functions
def range_func(x):
    return x.max() - x.min()

result = df.groupby('category').agg({
    'value': ['sum', 'mean', range_func]
})

# Apply custom function to groups
def standardize(group):
    return (group - group.mean()) / group.std()

df['standardized'] = df.groupby('category')['value'].transform(standardize)

# Filter groups based on condition
large_groups = df.groupby('category').filter(lambda x: len(x) > 100)

# Iterate over groups
for name, group in df.groupby('category'):
    print(f"Category: {name}, Size: {len(group)}")
    print(group.head())
```

### 6.2 Advanced GroupBy Techniques

**Transform vs Apply:**

```python
# transform: Returns Series/DataFrame same size as input
df['pct_of_group'] = df.groupby('category')['value'].transform(
    lambda x: x / x.sum()
)

# apply: Can return arbitrary shape
result = df.groupby('category').apply(
    lambda x: x.nlargest(3, 'value')
)

# Multiple transformations
df['group_mean'] = df.groupby('category')['value'].transform('mean')
df['group_std'] = df.groupby('category')['value'].transform('std')
df['z_score'] = (df['value'] - df['group_mean']) / df['group_std']
```

**GroupBy with Multiple Functions:**

```python
# Using agg with list
agg_funcs = ['count', 'sum', 'mean', 'std', 'min', 'max']
result = df.groupby('category')['value'].agg(agg_funcs)

# Different functions for different columns
result = df.groupby('category').agg({
    'quantity': ['sum', 'mean'],
    'price': ['min', 'max'],
    'date': ['min', 'max']
})

# Custom + built-in functions
result = df.groupby('category').agg({
    'value': [
        'sum',
        'mean',
        ('range', lambda x: x.max() - x.min()),
        ('q95', lambda x: x.quantile(0.95))
    ]
})
```

**GroupBy Performance Optimization:**

```python
# Use observed=True for categorical to skip empty groups
df['category'] = df['category'].astype('category')
result = df.groupby('category', observed=True)['value'].sum()

# Use sort=False if order doesn't matter
result = df.groupby('category', sort=False)['value'].sum()

# NumPy functions are faster than Pandas equivalents
result = df.groupby('category')['value'].agg(np.sum)  # Faster
result = df.groupby('category')['value'].sum()        # Slower
```

### 6.3 Pivot Tables

```python
# Basic pivot table
pivot = df.pivot_table(
    values='sales',
    index='product',
    columns='month',
    aggfunc='sum'
)

# Multiple aggregations
pivot = df.pivot_table(
    values='sales',
    index='product',
    columns='month',
    aggfunc=['sum', 'mean', 'count']
)

# Multiple values
pivot = df.pivot_table(
    values=['sales', 'quantity'],
    index='product',
    columns='month',
    aggfunc='sum'
)

# Multiple index levels
pivot = df.pivot_table(
    values='sales',
    index=['category', 'product'],
    columns='month',
    aggfunc='sum',
    fill_value=0,  # Replace NaN
    margins=True   # Add totals
)

# Custom aggregation
pivot = df.pivot_table(
    values='sales',
    index='product',
    columns='month',
    aggfunc=lambda x: x.max() - x.min()
)
```

### 6.4 Cross Tabulation

```python
# Basic crosstab (frequency)
ct = pd.crosstab(df['category'], df['status'])

# With values
ct = pd.crosstab(
    df['category'],
    df['status'],
    values=df['amount'],
    aggfunc='sum'
)

# Normalized crosstab
ct = pd.crosstab(
    df['category'],
    df['status'],
    normalize='index'  # Row percentages
)
ct = pd.crosstab(
    df['category'],
    df['status'],
    normalize='columns'  # Column percentages
)
ct = pd.crosstab(
    df['category'],
    df['status'],
    normalize='all'  # Overall percentages
)

# Multiple indices/columns
ct = pd.crosstab(
    [df['category'], df['subcategory']],
    [df['status'], df['region']],
    margins=True
)
```

---

## Phase 7: Reshaping Data

### 7.1 Pivot and Melt

**Wide to Long (Melt):**

```python
# Original wide format
# id  | score_math | score_english | score_science
# 1   | 85         | 90            | 88
# 2   | 78         | 82            | 80

# Melt to long format
df_long = df.melt(
    id_vars=['id'],
    value_vars=['score_math', 'score_english', 'score_science'],
    var_name='subject',
    value_name='score'
)

# Result:
# id | subject        | score
# 1  | score_math     | 85
# 1  | score_english  | 90
# 1  | score_science  | 88

# Melt with multiple id columns
df_long = df.melt(
    id_vars=['student_id', 'name'],
    var_name='subject',
    value_name='score'
)
```

**Long to Wide (Pivot):**

```python
# Pivot long to wide
df_wide = df_long.pivot(
    index='id',
    columns='subject',
    values='score'
)

# Pivot with multiple values
df_wide = df.pivot(
    index='date',
    columns='product',
    values=['sales', 'quantity']
)

# Handle duplicate indices with pivot_table
df_wide = df.pivot_table(
    index='id',
    columns='subject',
    values='score',
    aggfunc='mean'  # Aggregate duplicates
)
```

### 7.2 Stack and Unstack

```python
# Multi-index DataFrame
df_multi = df.set_index(['category', 'product'])

# Stack: Column level → row level
df_stacked = df_multi.stack()

# Unstack: Row level → column level
df_unstacked = df_stacked.unstack()

# Unstack specific level
df_unstacked = df_multi.unstack(level=0)
df_unstacked = df_multi.unstack(level='category')

# Stack/unstack with dropna
df_stacked = df_multi.stack(dropna=False)  # Keep NaN
df_unstacked = df_stacked.unstack(fill_value=0)  # Fill NaN
```

### 7.3 Explode

```python
# Explode list-like columns
df = pd.DataFrame({
    'id': [1, 2, 3],
    'tags': [['a', 'b'], ['c'], ['d', 'e', 'f']]
})

df_exploded = df.explode('tags')
# Result:
# id | tags
# 1  | a
# 1  | b
# 2  | c
# 3  | d
# 3  | e
# 3  | f

# Explode multiple columns
df_exploded = df.explode(['tags', 'values'])
```

---

## Phase 8: Merging and Joining

### 8.1 Merge Operations

**Basic Merge (Inner Join):**

```python
# Inner join (intersection)
result = pd.merge(
    df1,
    df2,
    on='key',
    how='inner'
)

# Specify left and right keys
result = pd.merge(
    df1,
    df2,
    left_on='key1',
    right_on='key2',
    how='inner'
)

# Multiple key columns
result = pd.merge(
    df1,
    df2,
    on=['key1', 'key2'],
    how='inner'
)
```

**Join Types:**

```python
# Inner join: Only matching keys
inner = pd.merge(df1, df2, on='key', how='inner')

# Left join: All left keys + matches
left = pd.merge(df1, df2, on='key', how='left')

# Right join: All right keys + matches
right = pd.merge(df1, df2, on='key', how='right')

# Outer join: All keys from both
outer = pd.merge(df1, df2, on='key', how='outer')

# Cross join: Cartesian product (Pandas 1.2+)
cross = pd.merge(df1, df2, how='cross')
```

**Advanced Merge Options:**

```python
# Indicator column (shows merge source)
result = pd.merge(
    df1,
    df2,
    on='key',
    how='outer',
    indicator=True
)
# Adds '_merge' column: 'left_only', 'right_only', 'both'

# Suffix for duplicate columns
result = pd.merge(
    df1,
    df2,
    on='key',
    how='inner',
    suffixes=('_left', '_right')
)

# Validate merge type
result = pd.merge(
    df1,
    df2,
    on='key',
    how='inner',
    validate='1:1'  # Options: '1:1', '1:m', 'm:1', 'm:m'
)
```

### 8.2 Join on Index

```python
# Join on index
result = df1.join(df2, how='inner')

# Join on specific column from left, index from right
result = df1.join(df2, on='key', how='left')

# Join multiple DataFrames
result = df1.join([df2, df3, df4], how='outer')

# Alternative: merge with index
result = pd.merge(
    df1,
    df2,
    left_index=True,
    right_index=True,
    how='inner'
)
```

### 8.3 Concatenation

```python
# Vertical concatenation (stack rows)
result = pd.concat([df1, df2], ignore_index=True)

# Horizontal concatenation (side by side)
result = pd.concat([df1, df2], axis=1)

# Concatenate with keys
result = pd.concat(
    [df1, df2, df3],
    keys=['source1', 'source2', 'source3']
)

# Only keep common columns
result = pd.concat([df1, df2], join='inner')

# Keep all columns (default)
result = pd.concat([df1, df2], join='outer')

# Verify integrity
result = pd.concat([df1, df2], verify_integrity=True)  # Error if duplicate indices
```

### 8.4 Merge Performance Tips

```python
# Sort both DataFrames by merge key first
df1 = df1.sort_values('key')
df2 = df2.sort_values('key')
result = pd.merge(df1, df2, on='key')  # Faster

# Use categorical dtype for merge keys
df1['key'] = df1['key'].astype('category')
df2['key'] = df2['key'].astype('category')

# Set index before joining repeatedly
df1_indexed = df1.set_index('key')
df2_indexed = df2.set_index('key')
result = df1_indexed.join(df2_indexed)  # Faster than merge

# Use merge for one-time operations
# Use join for index-based or repeated operations
```

---

## Phase 9: Performance Optimization

### 9.1 Memory Optimization Techniques

**Dtype Optimization:**

```python
# Check memory usage
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize numeric types
def optimize_numeric(df):
    """Reduce memory by downcasting numeric columns"""
    df_optimized = df.copy()
    
    # Integer columns
    int_cols = df.select_dtypes(include=['int64']).columns
    df_optimized[int_cols] = df_optimized[int_cols].apply(pd.to_numeric, downcast='integer')
    
    # Float columns
    float_cols = df.select_dtypes(include=['float64']).columns
    df_optimized[float_cols] = df_optimized[float_cols].apply(pd.to_numeric, downcast='float')
    
    return df_optimized

# Category optimization for low-cardinality strings
def optimize_objects(df, threshold=0.5):
    """Convert object columns to category if cardinality < threshold"""
    df_optimized = df.copy()
    
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        num_unique = df[col].nunique()
        num_total = len(df)
        
        if num_unique / num_total < threshold:
            df_optimized[col] = df[col].astype('category')
    
    return df_optimized

# Comprehensive optimization
def optimize_dataframe(df):
    """Complete DataFrame optimization"""
    df_opt = df.copy()
    
    # Numeric optimization
    int_cols = df_opt.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='integer')
    
    float_cols = df_opt.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_opt[col] = pd.to_numeric(df_opt[col], downcast='float')
    
    # Object to category
    obj_cols = df_opt.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df_opt[col].nunique() / len(df_opt) < 0.5:
            df_opt[col] = df_opt[col].astype('category')
    
    # Use nullable integer types
    for col in int_cols:
        if df_opt[col].isna().any():
            df_opt[col] = df_opt[col].astype('Int64')
    
    return df_opt

# Memory comparison
before_mb = df.memory_usage(deep=True).sum() / 1024**2
df_optimized = optimize_dataframe(df)
after_mb = df_optimized.memory_usage(deep=True).sum() / 1024**2

print(f"Before: {before_mb:.2f} MB")
print(f"After: {after_mb:.2f} MB")
print(f"Reduction: {(1 - after_mb/before_mb)*100:.2f}%")
```

**Memory Reduction Example:**

| Data Type | Original | Optimized | Memory Saved |
|-----------|----------|-----------|--------------|
| int64 (1M rows) | 7.6 MB | 1.0 MB (int8) | 87% |
| float64 (1M rows) | 7.6 MB | 3.8 MB (float32) | 50% |
| object (low cardinality) | 50 MB | 1.5 MB (category) | 97% |

### 9.2 Vectorization and Broadcasting

**Avoid Loops - Use Vectorization:**

```python
import numpy as np
import time

# Create sample data
df = pd.DataFrame({
    'a': np.random.rand(100000),
    'b': np.random.rand(100000)
})

# Method 1: Loop (NEVER DO THIS)
start = time.time()
result = []
for idx in range(len(df)):
    result.append(df.loc[idx, 'a'] + df.loc[idx, 'b'])
df['c'] = result
print(f"Loop: {time.time() - start:.4f}s")  # ~15 seconds

# Method 2: iterrows (STILL BAD)
start = time.time()
result = []
for idx, row in df.iterrows():
    result.append(row['a'] + row['b'])
df['c'] = result
print(f"iterrows: {time.time() - start:.4f}s")  # ~5 seconds

# Method 3: apply (BETTER, but not optimal)
start = time.time()
df['c'] = df.apply(lambda row: row['a'] + row['b'], axis=1)
print(f"apply: {time.time() - start:.4f}s")  # ~0.5 seconds

# Method 4: Vectorized (BEST)
start = time.time()
df['c'] = df['a'] + df['b']
print(f"Vectorized: {time.time() - start:.4f}s")  # ~0.002 seconds
```

**Vectorization Speed Comparison:**

$
\text{Speedup} = \frac{\text{Loop Time}}{\text{Vectorized Time}} \approx 7500\times
$

**NumPy Operations:**

```python
# Use NumPy for complex operations
df['result'] = np.where(df['a'] > 0.5, df['a'] * 2, df['a'] / 2)

# Multiple conditions
conditions = [
    df['a'] > 0.7,
    df['a'] > 0.3,
    df['a'] >= 0
]
choices = [df['a'] * 3, df['a'] * 2, df['a']]
df['result'] = np.select(conditions, choices, default=0)

# Mathematical operations
df['sqrt'] = np.sqrt(df['a'])
df['log'] = np.log(df['a'] + 1)
df['exp'] = np.exp(df['a'])

# Statistical operations
df['zscore'] = (df['a'] - df['a'].mean()) / df['a'].std()
df['percentile'] = df['a'].rank(pct=True)
```

### 9.3 Efficient Data Loading

**Chunked Reading:**

```python
# Read large CSV in chunks
chunk_size = 10000
chunks = []

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process chunk
    filtered = chunk[chunk['value'] > 100]
    chunks.append(filtered)

# Combine results
df = pd.concat(chunks, ignore_index=True)
```

**Selective Column Loading:**

```python
# Only load needed columns
df = pd.read_csv(
    'data.csv',
    usecols=['col1', 'col2', 'col3']  # Only these columns
)

# Specify dtypes during load (fastest)
df = pd.read_csv(
    'data.csv',
    dtype={
        'id': 'int32',
        'value': 'float32',
        'category': 'category'
    },
    parse_dates=['date']
)
```

**Parallel Processing:**

```python
import multiprocessing as mp
from functools import partial

def process_chunk(chunk, threshold):
    """Process single chunk"""
    return chunk[chunk['value'] > threshold]

# Split DataFrame
n_cores = mp.cpu_count()
chunks = np.array_split(df, n_cores)

# Parallel processing
with mp.Pool(n_cores) as pool:
    process_func = partial(process_chunk, threshold=100)
    results = pool.map(process_func, chunks)

# Combine results
df_processed = pd.concat(results, ignore_index=True)
```

### 9.4 Query Optimization

**Use eval() for Complex Expressions:**

```python
# Traditional (creates intermediate arrays)
result = df[(df['a'] > 0) & (df['b'] < 10) & (df['c'] == 5)]

# eval() (more memory efficient)
result = df.query('a > 0 and b < 10 and c == 5')

# Complex numeric expressions
df['result'] = df.eval('a + b * c - d / e')

# Assignment with eval
df.eval('result = a + b', inplace=True)
```

**Index Optimization:**

```python
# Set index for frequent lookups
df_indexed = df.set_index('id')

# Lookup is faster on indexed column
value = df_indexed.loc[12345]  # O(log n) vs O(n)

# Multi-index for hierarchical data
df_multi = df.set_index(['category', 'product'])
result = df_multi.loc[('Electronics', 'Laptop')]

# Reset index when no longer needed
df = df_multi.reset_index()
```

---

## Phase 10: Best Practices and Patterns

### 10.1 Method Chaining

**Building Readable Pipelines:**

```python
# Bad: Multiple intermediate variables
df_temp1 = df[df['age'] > 18]
df_temp2 = df_temp1.dropna()
df_temp3 = df_temp2.assign(age_group=lambda x: x['age'] // 10)
df_result = df_temp3.sort_values('age')

# Good: Method chaining
df_result = (
    df
    .query('age > 18')
    .dropna()
    .assign(age_group=lambda x: x['age'] // 10)
    .sort_values('age')
    .reset_index(drop=True)
)

# Complex pipeline
result = (
    df
    .drop_duplicates()
    .fillna({'col1': 0, 'col2': 'unknown'})
    .astype({'col3': 'category'})
    .assign(
        total=lambda x: x['price'] * x['quantity'],
        discount_pct=lambda x: x['discount'] / x['price'] * 100
    )
    .query('total > 1000')
    .groupby('category')
    .agg({
        'total': ['sum', 'mean'],
        'quantity': 'sum'
    })
    .round(2)
)
```

**Using pipe() for Custom Functions:**

```python
def remove_outliers(df, column, n_std=3):
    """Remove outliers beyond n standard deviations"""
    mean = df[column].mean()
    std = df[column].std()
    return df[
        (df[column] >= mean - n_std * std) &
        (df[column] <= mean + n_std * std)
    ]

def add_features(df):
    """Add engineered features"""
    return df.assign(
        total=df['price'] * df['quantity'],
        price_per_unit=df['total'] / df['quantity']
    )

# Chain with custom functions
result = (
    df
    .pipe(remove_outliers, 'price', n_std=2)
    .pipe(add_features)
    .query('total > 100')
)
```

### 10.2 Handling Copy vs View

**Understanding Copy and View:**

```python
# View (references original data)
view = df[['col1', 'col2']]  # May be view
view['col1'] = 0  # SettingWithCopyWarning!

# Explicit copy (independent data)
copy = df[['col1', 'col2']].copy()
copy['col1'] = 0  # Safe

# Chained indexing (ambiguous)
df[df['age'] > 18]['name'] = 'Adult'  # Warning!

# Correct approach
df.loc[df['age'] > 18, 'name'] = 'Adult'

# Check if view or copy
print(df_subset._is_view)  # True if view
```

### 10.3 Avoiding Common Pitfalls

**Pitfall 1: Chained Indexing**

```python
# BAD: Chained indexing
df[df['col1'] > 0]['col2'] = 100  # May not work!

# GOOD: Single loc statement
df.loc[df['col1'] > 0, 'col2'] = 100
```

**Pitfall 2: Inplace Operations**

```python
# AVOID: inplace=True hinders method chaining
df.dropna(inplace=True)
df.fillna(0, inplace=True)

# PREFER: Assignment for clarity
df = df.dropna()
df = df.fillna(0)

# OR: Method chaining
df = df.dropna().fillna(0)
```

**Pitfall 3: Modifying During Iteration**

```python
# BAD: Modifying DataFrame during iteration
for idx, row in df.iterrows():
    df.loc[idx, 'new_col'] = row['col1'] * 2  # Slow!

# GOOD: Vectorized operation
df['new_col'] = df['col1'] * 2
```

**Pitfall 4: Not Using Categorical**

```python
# BAD: High-cardinality object column
df['category'] = df['category'].astype('object')  # Wasteful

# GOOD: Use categorical for repeated values
df['category'] = df['category'].astype('category')  # 90% memory savings
```

**Pitfall 5: Ignoring Index**

```python
# BAD: Forgetting to reset index after operations
df_grouped = df.groupby('category')['value'].sum()  # Returns Series with category as index
df_grouped['Electronics']  # Accessing by label

# GOOD: Reset index for clarity
df_grouped = df.groupby('category')['value'].sum().reset_index()
df_grouped[df_grouped['category'] == 'Electronics']
```

### 10.4 Code Organization Patterns

**Functional Data Processing:**

```python
def load_data(filepath):
    """Load and perform initial cleaning"""
    return (
        pd.read_csv(filepath)
        .drop_duplicates()
        .dropna(thresh=len(df.columns) * 0.5)  # Drop rows with >50% nulls
    )

def clean_data(df):
    """Apply data cleaning transformations"""
    return (
        df
        .assign(
            date=lambda x: pd.to_datetime(x['date']),
            amount=lambda x: pd.to_numeric(x['amount'], errors='coerce')
        )
```

---
