---
layout: post
title: "🌊 NumPy: Deep Dive & Best Practices"
description: "Concise, clear, and validated revision notes on NumPy — structured for beginners and practitioners."
author: technical_notes
date: 2025-11-10 5:30:00 +0530
categories: [Notes, NumPy]
tags: [NumPy, Python, Machine Learning, Data Science, Optimization, Best Practices]
image: /assets/img/posts/numpy-logo.png
toc: true
math: true
mermaid: false
---

## Introduction

**NumPy** (Numerical Python) is the foundational library for scientific computing in Python. It provides powerful support for large, multi-dimensional arrays and matrices, along with an extensive collection of high-level mathematical functions to operate on these data structures. NumPy serves as the backbone for nearly all scientific computing packages in Python, including Pandas, SciPy, Scikit-learn, TensorFlow, and PyTorch.

### Why NumPy?

Python lists are versatile but inefficient for numerical computations:
- **Heterogeneous storage**: Lists can contain mixed data types, requiring type checking at every operation
- **Memory overhead**: Each list element is a Python object with significant overhead (28+ bytes per integer)
- **Interpreted loops**: Pure Python loops are slow due to dynamic typing and interpretation overhead

NumPy solves these problems by:
- **Homogeneous arrays**: All elements share the same data type, enabling optimizations
- **Contiguous memory**: Data stored in C-style arrays for cache-efficient access
- **Vectorized operations**: Operations delegated to optimized C/Fortran code
- **Broadcasting**: Intelligent handling of arrays with different shapes

**Performance difference example**:
```python
import numpy as np
import time

# Python list approach
python_list = list(range(1000000))
start = time.time()
result_list = [x ** 2 for x in python_list]
print(f"Python list: {time.time() - start:.4f}s")

# NumPy approach
numpy_array = np.arange(1000000)
start = time.time()
result_array = numpy_array ** 2
print(f"NumPy array: {time.time() - start:.4f}s")

# NumPy is typically 10-100x faster
```

---

## NumPy Terminology & Jargon

Understanding NumPy terminology is crucial for effective usage and reading documentation.

### Table 1: Core NumPy Terminology

| Term | Alternative Names | Definition | Example |
|------|------------------|------------|---------|
| **ndarray** | array, NumPy array | N-dimensional array object | `np.array([1, 2, 3])` |
| **axis** | dimension | Direction along array (0=rows, 1=cols, etc.) | `arr.sum(axis=0)` |
| **shape** | dimensions | Tuple of array dimensions | `(3, 4)` for 3×4 array |
| **dtype** | data type | Type of elements in array | `np.float64`, `np.int32` |
| **broadcast** | - | Implicit element-wise operations on different shapes | `arr + 5` |
| **vectorization** | vectorized operations | Operations on entire arrays without explicit loops | `arr * 2` |
| **view** | shallow copy | Reference to same data, no copy made | `arr[:]` |
| **copy** | deep copy | Independent duplicate of data | `arr.copy()` |
| **strides** | step size | Bytes to step in each dimension | `arr.strides` |
| **ufunc** | universal function | Function that operates element-wise | `np.sin`, `np.add` |
| **fancy indexing** | advanced indexing | Indexing with arrays | `arr[[0, 2, 4]]` |
| **boolean indexing** | mask indexing | Indexing with boolean arrays | `arr[arr > 5]` |
| **structured array** | record array | Array with named fields | Heterogeneous data types |
| **memory layout** | order | C-order (row-major) or F-order (column-major) | `order='C'` |

### Table 2: Hierarchical Organization of NumPy Concepts

| Level | Category | Subcategories | Components | Technical Detail |
|-------|----------|---------------|------------|------------------|
| **Foundation** | ndarray object | - | Shape, dtype, strides, data buffer | Memory-contiguous homogeneous array |
| **Creation** | Array generation | Explicit, Generators, I/O | `array()`, `zeros()`, `arange()`, `loadtxt()` | Various initialization methods |
| **Manipulation** | Array operations | Shape, Content, Joining/Splitting | `reshape()`, `append()`, `concatenate()` | Transform and combine arrays |
| **Indexing** | Element access | Basic, Advanced, Boolean | Slicing, fancy indexing, masks | Multiple access patterns |
| **Operations** | Computations | Element-wise, Aggregation, Linear algebra | ufuncs, `sum()`, `dot()` | Mathematical operations |
| **Broadcasting** | Shape alignment | Rules, Applications | Implicit dimension matching | Automatic shape compatibility |
| **Performance** | Optimization | Vectorization, Memory, Compilation | Remove loops, views, Numba | Speed and memory efficiency |
| **Advanced** | Specialized features | Structured arrays, Memory views, C-API | Complex data structures | Low-level control |

### Table 3: Operation Terminology Equivalences

| Generic Term | NumPy Term | Mathematical Notation | Example Code |
|--------------|------------|----------------------|--------------|
| **Add** | Element-wise addition | $$c_{ij} = a_{ij} + b_{ij}$$ | `np.add(a, b)` or `a + b` |
| **Multiply** | Element-wise multiplication | $$c_{ij} = a_{ij} \cdot b_{ij}$$ | `np.multiply(a, b)` or `a * b` |
| **Dot product** | Matrix multiplication | $$c_{ij} = \sum_k a_{ik} b_{kj}$$ | `np.dot(a, b)` or `a @ b` |
| **Transpose** | Axis permutation | $$A^T$$ | `arr.T` or `np.transpose(arr)` |
| **Reduce** | Aggregate operation | $$\sum, \prod, \max, \min$$ | `np.sum()`, `np.mean()` |
| **Map** | Element-wise function | $$f(x_i)$$ | ufuncs: `np.sin()`, `np.exp()` |
| **Filter** | Boolean indexing | Conditional selection | `arr[arr > 0]` |
| **Reshape** | Change dimensions | Reinterpret shape | `arr.reshape((2, 3))` |

---

## Core NumPy Concepts

### 1. The ndarray Object

The `ndarray` (N-dimensional array) is NumPy's fundamental data structure.

#### 1.1 Array Attributes

```python
import numpy as np

# Create a sample array
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# Fundamental attributes
print(f"Shape: {arr.shape}")        # (3, 4) - dimensions
print(f"Dimensions: {arr.ndim}")    # 2 - number of axes
print(f"Size: {arr.size}")          # 12 - total elements
print(f"Data type: {arr.dtype}")    # dtype('int64') or similar
print(f"Item size: {arr.itemsize}") # 8 bytes (for int64)
print(f"Total bytes: {arr.nbytes}") # 96 bytes
print(f"Strides: {arr.strides}")    # (32, 8) - bytes per step
print(f"Flags: {arr.flags}")        # Memory layout info
```

**Shape**: Tuple describing dimensions. Shape `(3, 4)` means 3 rows, 4 columns.

**Strides**: Number of bytes to skip to reach next element in each dimension. For a `(3, 4)` array of `int64`:
- Stride for axis 0 (rows): 32 bytes (skip 4 elements × 8 bytes)
- Stride for axis 1 (columns): 8 bytes (skip 1 element × 8 bytes)

#### 1.2 Data Types (dtype)

NumPy provides precise control over data types:

```python
# Integer types
np.int8, np.int16, np.int32, np.int64      # Signed integers
np.uint8, np.uint16, np.uint32, np.uint64   # Unsigned integers

# Float types
np.float16, np.float32, np.float64         # Floating point
np.complex64, np.complex128                 # Complex numbers

# Boolean and string
np.bool_, np.str_, np.unicode_

# Creating arrays with specific dtype
arr_float = np.array([1, 2, 3], dtype=np.float32)
arr_int8 = np.array([1, 2, 3], dtype=np.int8)

# Type conversion
arr_converted = arr_float.astype(np.int32)

# Memory savings example
large_arr_int64 = np.arange(1000000, dtype=np.int64)  # 8 MB
large_arr_int8 = np.arange(1000000, dtype=np.int8)    # 1 MB (8x smaller!)

print(f"int64 size: {large_arr_int64.nbytes / 1024:.2f} KB")
print(f"int8 size: {large_arr_int8.nbytes / 1024:.2f} KB")
```

**dtype Selection Guidelines**:
- Use smallest sufficient type to save memory
- `int8` (-128 to 127), `int16` (-32768 to 32767), etc.
- `float32` for neural networks (balance precision/memory)
- `float64` (default) for scientific computing requiring precision
- `bool` for masks (1 byte per element)

### 2. Array Creation

#### 2.1 From Python Sequences

```python
# 1D array from list
arr_1d = np.array([1, 2, 3, 4, 5])

# 2D array from nested lists
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Specify data type
arr_float = np.array([1, 2, 3], dtype=float)

# From tuple
arr_tuple = np.array((1, 2, 3))
```

#### 2.2 Array Generation Functions

```python
# Zeros: all elements 0
zeros_arr = np.zeros((3, 4))              # 3×4 array of zeros
zeros_int = np.zeros(10, dtype=int)       # 10 integer zeros

# Ones: all elements 1
ones_arr = np.ones((2, 3, 4))            # 3D array of ones

# Empty: uninitialized (faster than zeros, use with caution)
empty_arr = np.empty((2, 2))

# Full: constant value
full_arr = np.full((3, 3), 7)            # 3×3 array of 7s

# Identity matrix
identity = np.eye(4)                      # 4×4 identity matrix
identity_offset = np.eye(5, k=1)          # Diagonal offset by 1

# Like: same shape as existing array
ones_like_arr = np.ones_like(arr_2d)
zeros_like_arr = np.zeros_like(arr_2d)
```

#### 2.3 Range and Sequence Generation

```python
# arange: similar to Python range()
arr_range = np.arange(10)                 # [0, 1, 2, ..., 9]
arr_range_step = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
arr_range_float = np.arange(0, 1, 0.1)    # Floats: [0, 0.1, 0.2, ..., 0.9]

# linspace: linearly spaced values (includes endpoint by default)
lin_arr = np.linspace(0, 1, 11)           # 11 values from 0 to 1
# [0.0, 0.1, 0.2, ..., 1.0]

# logspace: logarithmically spaced
log_arr = np.logspace(0, 2, 5)            # 5 values from 10^0 to 10^2
# [1, 10, 100]

# geomspace: geometrically spaced (better than logspace for negative numbers)
geom_arr = np.geomspace(1, 1000, 4)       # [1, 10, 100, 1000]
```

**arange vs linspace**:
- `arange(start, stop, step)`: Excludes stop, can have floating-point errors
- `linspace(start, stop, num)`: Includes stop, guaranteed exact number of points

#### 2.4 Random Number Generation

```python
# NumPy 1.17+ recommended API (Generator)
rng = np.random.default_rng(seed=42)  # Reproducible with seed

# Random floats [0, 1)
random_floats = rng.random((3, 4))

# Random integers
random_ints = rng.integers(low=0, high=10, size=(3, 3))

# Standard normal distribution (mean=0, std=1)
normal_dist = rng.standard_normal((1000,))

# Normal distribution (custom mean, std)
custom_normal = rng.normal(loc=5, scale=2, size=(1000,))

# Uniform distribution [low, high)
uniform = rng.uniform(low=-1, high=1, size=(100,))

# Choice: random selection
choices = rng.choice([1, 2, 3, 4, 5], size=10, replace=True)

# Shuffle: in-place shuffle
arr_to_shuffle = np.arange(10)
rng.shuffle(arr_to_shuffle)

# Permutation: returns shuffled copy
permuted = rng.permutation(10)

# Legacy API (still works but not recommended for new code)
np.random.seed(42)  # Old way to set seed
np.random.rand(3, 3)  # Random [0, 1)
np.random.randn(3, 3)  # Standard normal
np.random.randint(0, 10, (3, 3))  # Random integers
```

### 3. Array Indexing & Slicing

#### 3.1 Basic Indexing

```python
arr = np.array([10, 20, 30, 40, 50])

# Single element access (0-indexed)
print(arr[0])      # 10 (first element)
print(arr[-1])     # 50 (last element)
print(arr[-2])     # 40 (second to last)

# 2D indexing
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(arr_2d[0, 0])    # 1 (row 0, col 0)
print(arr_2d[1, 2])    # 6 (row 1, col 2)
print(arr_2d[-1, -1])  # 9 (last row, last col)

# Alternative indexing (less efficient)
print(arr_2d[0][0])    # Works but slower (two indexing operations)
```

#### 3.2 Slicing

Syntax: `start:stop:step` (stop is exclusive)

```python
arr = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Basic slicing
print(arr[2:5])      # [2, 3, 4] (indices 2, 3, 4)
print(arr[:5])       # [0, 1, 2, 3, 4] (first 5 elements)
print(arr[5:])       # [5, 6, 7, 8, 9] (from index 5 to end)
print(arr[::2])      # [0, 2, 4, 6, 8] (every 2nd element)
print(arr[::-1])     # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0] (reversed)

# 2D slicing
arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(arr_2d[1:3, :])     # Rows 1-2, all columns
# [[5, 6, 7, 8],
#  [9, 10, 11, 12]]

print(arr_2d[:, 1:3])     # All rows, columns 1-2
# [[2, 3],
#  [6, 7],
#  [10, 11]]

print(arr_2d[::2, ::2])   # Every 2nd row, every 2nd column
# [[1, 3],
#  [9, 11]]
```

**Critical**: Slices create **views**, not copies!

```python
arr = np.arange(10)
slice_view = arr[2:5]  # View into arr

slice_view[0] = 999    # Modifies original!
print(arr)             # [0, 1, 999, 3, 4, 5, 6, 7, 8, 9]

# To create independent copy
slice_copy = arr[2:5].copy()
slice_copy[0] = 777    # Does NOT modify arr
```

#### 3.3 Boolean Indexing (Masking)

Powerful technique for filtering arrays:

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Create boolean mask
mask = arr > 5
print(mask)  # [False, False, False, False, False, True, True, True, True, True]

# Apply mask
filtered = arr[mask]
print(filtered)  # [6, 7, 8, 9, 10]

# Compound conditions
mask_complex = (arr > 3) & (arr < 8)  # Use & for AND, | for OR, ~ for NOT
print(arr[mask_complex])  # [4, 5, 6, 7]

# Direct filtering (more concise)
print(arr[arr % 2 == 0])  # Even numbers: [2, 4, 6, 8, 10]

# Modify using boolean indexing
arr[arr > 5] = 0  # Set all elements > 5 to 0
print(arr)  # [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]

# Multi-dimensional masking
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

mask_2d = arr_2d > 5
print(arr_2d[mask_2d])  # [6, 7, 8, 9] (flattened)

# Replace specific values
arr_2d[arr_2d > 5] = -1
print(arr_2d)
# [[1, 2, 3],
#  [4, 5, -1],
#  [-1, -1, -1]]
```

#### 3.4 Fancy Indexing (Advanced Indexing)

Index with integer arrays:

```python
arr = np.array([10, 20, 30, 40, 50])

# Index with list/array of integers
indices = [0, 2, 4]
print(arr[indices])  # [10, 30, 50]

# Can repeat indices
print(arr[[0, 0, 1, 1]])  # [10, 10, 20, 20]

# 2D fancy indexing
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Select specific rows
rows = [0, 2]
print(arr_2d[rows])
# [[1, 2, 3],
#  [7, 8, 9]]

# Select specific elements
rows = [0, 1, 2]
cols = [0, 1, 2]
print(arr_2d[rows, cols])  # [1, 5, 9] (diagonal elements)

# Grid-based selection
rows = np.array([[0, 0], [1, 1]])
cols = np.array([[0, 1], [0, 1]])
print(arr_2d[rows, cols])
# [[1, 2],
#  [4, 5]]
```

**Warning**: Fancy indexing creates **copies**, not views!

```python
arr = np.arange(5)
fancy_indexed = arr[[0, 2, 4]]
fancy_indexed[0] = 999
print(arr)  # [0, 1, 2, 3, 4] - unchanged!
```

---

## Array Operations

### 4. Element-wise Operations (Universal Functions - ufuncs)

NumPy ufuncs operate element-wise on arrays at C speed.

#### 4.1 Arithmetic Operations

```python
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

# Basic arithmetic (element-wise)
print(arr1 + arr2)   # [11, 22, 33, 44]
print(arr1 - arr2)   # [-9, -18, -27, -36]
print(arr1 * arr2)   # [10, 40, 90, 160]
print(arr1 / arr2)   # [0.1, 0.1, 0.1, 0.1]
print(arr1 ** 2)     # [1, 4, 9, 16]
print(arr2 // arr1)  # [10, 10, 10, 10] (integer division)
print(arr2 % arr1)   # [0, 0, 0, 0] (modulo)

# With scalars (broadcasting)
print(arr1 + 10)     # [11, 12, 13, 14]
print(arr1 * 2)      # [2, 4, 6, 8]

# Unary operations
print(-arr1)         # [-1, -2, -3, -4]
print(np.abs(np.array([-1, -2, 3])))  # [1, 2, 3]
```

#### 4.2 Mathematical Functions

```python
arr = np.array([0, np.pi/2, np.pi])

# Trigonometric
print(np.sin(arr))      # [0, 1, 0]
print(np.cos(arr))      # [1, 0, -1]
print(np.tan(arr))      # [0, large, 0]

# Inverse trig
print(np.arcsin([0, 1]))  # [0, π/2]
print(np.arccos([1, 0]))  # [0, π/2]

# Exponential and logarithmic
arr_pos = np.array([1, 2, 3])
print(np.exp(arr_pos))      # [e^1, e^2, e^3]
print(np.log(arr_pos))      # [0, ln(2), ln(3)] (natural log)
print(np.log10(arr_pos))    # [0, log10(2), log10(3)]
print(np.log2(arr_pos))     # [0, 1, log2(3)]

# Power and roots
print(np.power(arr_pos, 3))  # [1, 8, 27]
print(np.sqrt(arr_pos))      # [1, √2, √3]
print(np.cbrt(arr_pos))      # [1, ∛2, ∛3] (cube root)

# Rounding
arr_float = np.array([1.2, 2.5, 3.7, 4.5])
print(np.round(arr_float))    # [1, 2, 4, 4] (round half to even)
print(np.floor(arr_float))    # [1, 2, 3, 4]
print(np.ceil(arr_float))     # [2, 3, 4, 5]
print(np.trunc(arr_float))    # [1, 2, 3, 4] (truncate towards zero)

# Statistical functions
arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))           # 3.0
print(np.median(arr))         # 3.0
print(np.std(arr))            # 1.414... (standard deviation)
print(np.var(arr))            # 2.0 (variance)

# Cumulative operations
print(np.cumsum(arr))         # [1, 3, 6, 10, 15]
print(np.cumprod(arr))        # [1, 2, 6, 24, 120]
```

#### 4.3 Comparison Operations

```python
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([5, 4, 3, 2, 1])

# Element-wise comparison (returns boolean array)
print(arr1 == arr2)  # [False, False, True, False, False]
print(arr1 != arr2)  # [True, True, False, True, True]
print(arr1 < arr2)   # [True, True, False, False, False]
print(arr1 <= arr2)  # [True, True, True, False, False]
print(arr1 > arr2)   # [False, False, False, True, True]
print(arr1 >= arr2)  # [False, False, True, True, True]

# Array-wise comparison
print(np.array_equal(arr1, arr2))  # False
print(np.allclose(arr1, arr2))     # False (within tolerance)

# Logical operations on boolean arrays
mask1 = arr1 > 2
mask2 = arr1 < 5
print(np.logical_and(mask1, mask2))  # Element-wise AND
print(np.logical_or(mask1, mask2))   # Element-wise OR
print(np.logical_not(mask1))         # Element-wise NOT
print(np.logical_xor(mask1, mask2))  # Element-wise XOR
```

### 5. Aggregation Functions

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Sum
print(np.sum(arr))         # 21 (all elements)
print(np.sum(arr, axis=0)) # [5, 7, 9] (column sums)
print(np.sum(arr, axis=1)) # [6, 15] (row sums)

# Product
print(np.prod(arr))         # 720 (all elements)
print(np.prod(arr, axis=0)) # [4, 10, 18]

# Min/Max
print(np.min(arr))          # 1
print(np.max(arr))          # 6
print(np.argmin(arr))       # 0 (index of min in flattened array)
print(np.argmax(arr))       # 5 (index of max in flattened array)

# Along axis
print(np.min(arr, axis=1))  # [1, 4] (min of each row)
print(np.argmin(arr, axis=1))  # [0, 0] (column index of min in each row)

# Mean, Median, Percentile
print(np.mean(arr))         # 3.5
print(np.median(arr))       # 3.5
print(np.percentile(arr, 50))  # 3.5 (50th percentile = median)
print(np.percentile(arr, [25, 50, 75]))  # [2.25, 3.5, 4.75]

# Standard deviation and variance
print(np.std(arr))          # 1.707...
print(np.var(arr))          # 2.916...

# Any/All (boolean testing)
mask = arr > 3
print(np.any(mask))         # True (at least one True)
print(np.all(mask))         # False (not all True)
print(np.any(mask, axis=0)) # [False, True, True]
print(np.all(arr > 0))      # True (all positive)
```

**Axis Understanding**:
- `axis=0`: Operation along rows (column-wise result)
- `axis=1`: Operation along columns (row-wise result)
- `axis=None` (default): Operation on flattened array

### 6. Array Manipulation

#### 6.1 Reshaping

```python
arr = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape to 2D
arr_2d = arr.reshape((3, 4))
print(arr_2d)
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11]]

# Reshape to 3D
arr_3d = arr.reshape((2, 3, 2))
print(arr_3d.shape)  # (2, 3, 2)

# Automatic dimension calculation
arr_auto = arr.reshape((3, -1))  # -1 means "calculate this dimension"
# Result: (3, 4)

# Flatten to 1D
flattened = arr_2d.flatten()  # Creates copy
raveled = arr_2d.ravel()      # Returns view if possible

# Transpose
transposed = arr_2d.T
print(transposed.shape)  # (4, 3)

# Swap axes
arr_3d = np.arange(24).reshape((2, 3, 4))
swapped = np.swapaxes(arr_3d, 0, 2)  # Swap axis 0 and 2
print(swapped.shape)  # (4, 3, 2)

# Add new axis
arr_1d = np.array([1, 2, 3])
arr_col = arr_1d[:, np.newaxis]  # Column vector (3, 1)
arr_row = arr_1d[np.newaxis, :]  # Row vector (1, 3)

# Alternative: expand_dims
arr_col = np.expand_dims(arr_1d, axis=1)
```

#### 6.2 Concatenation and Splitting

```python
arr1 = np.array([[1, 2],
                 [3, 4]])
arr2 = np.array([[5, 6],
                 [7, 8]])

# Concatenate (specify axis)
concat_vertical = np.concatenate((arr1, arr2), axis=0)
print(concat_vertical)
# [[1, 2],
#  [3, 4],
#  [5, 6],
#  [7, 8]]

concat_horizontal = np.concatenate((arr1, arr2), axis=1)
print(concat_horizontal)
# [[1, 2, 5, 6],
#  [3, 4, 7, 8]]

# Convenience functions
vstack = np.vstack((arr1, arr2))  # Vertical stack (axis=0)
hstack = np.hstack((arr1, arr2))  # Horizontal stack (axis=1)

# For 1D arrays
arr_a = np.array([1, 2, 3])
arr_b = np.array([4, 5, 6])
column_stack = np.column_stack((arr_a, arr_b))
print(column_stack)
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# dstack: depth stack (along 3rd axis)
dstacked = np.dstack((arr1, arr2))
print(dstacked.shape)  # (2, 2, 2)

# Splitting
arr = np.arange(12).reshape((3, 4))

# Split into equal parts
split_rows = np.split(arr, 3, axis=0)  # 3 sub-arrays along axis 0
print(len(split_rows))  # 3

# Split at specific indices
split_at_indices = np.split(arr, [1, 3], axis=1)
# Splits at columns 1 and 3: [:1], [1:3], [3:]

# Convenience functions
vsplit = np.vsplit(arr, 3)   # Vertical split (rows)
hsplit = np.hsplit(arr, [2])  # Horizontal split at column 2
```

#### 6.3 Array Copying

```python
arr = np.array([1, 2, 3, 4, 5])

# Assignment creates reference (NOT a copy)
ref = arr
ref[0] = 999
print(arr[0])  # 999 (modified!)

# View: shares data but different metadata
view = arr.view()
view[1] = 888
print(arr[1])  # 888 (data is shared!)

# However, reshaping view doesn't affect original shape
view_reshaped = view.reshape((5, 1))
print(arr.shape)  # (5,) - unchanged

# Deep copy: independent
deep_copy = arr.copy()
deep_copy[2] = 777
print(arr[2])  # 3 (unchanged!)

# Check if arrays share data
print(np.shares_memory(arr, view))      # True
print(np.shares_memory(arr, deep_copy)) # False
```

---

## Broadcasting

Broadcasting is NumPy's powerful mechanism for performing operations on arrays of different shapes.

### Broadcasting Rules

For two arrays to be compatible for broadcasting:
1. Dimensions are compared from **right to left** (trailing dimensions)
2. Dimensions are compatible if:
   - They are equal, OR
   - One of them is 1

**Examples**:

```python
# Rule visualization
# Array A shape: (3, 4, 5)
# Array B shape:    (4, 5)  → broadcasts to (1, 4, 5) → compatible
# Array C shape:       (5)  → broadcasts to (1, 1, 5) → compatible
# Array D shape:    (3, 1)  → broadcasts to (3, 1, 1) → NOT compatible
```

### Broadcasting Examples

```python
# Example 1: Scalar broadcasting
arr = np.array([1, 2, 3, 4])
result = arr + 10  # Scalar 10 broadcasts to [10, 10, 10, 10]
print(result)  # [11, 12, 13, 14]

# Example 2: 1D + 2D
arr_1d = np.array([1, 2, 3])       # Shape: (3,)
arr_2d = np.array([[10],           # Shape: (2, 1)
                   [20]])

result = arr_1d + arr_2d
# Broadcasting:
# arr_1d: (3,)   → (1, 3) → (2, 3)
# arr_2d: (2, 1) → (2, 1) → (2, 3)
print(result)
# [[11, 12, 13],   # 10 + [1, 2, 3]
#  [21, 22, 23]]   # 20 + [1, 2, 3]

# Example 3: Standardization (subtract mean, divide by std)
data = np.random.randn(100, 5)  # 100 samples, 5 features
mean = data.mean(axis=0)         # Shape: (5,)
std = data.std(axis=0)           # Shape: (5,)

standardized = (data - mean) / std  # Broadcasting: (100, 5) - (5,) / (5,)

# Example 4: Outer product using broadcasting
a = np.array([1, 2, 3])[:, np.newaxis]  # Shape: (3, 1)
b = np.array([4, 5, 6])                 # Shape: (3,)

outer = a * b  # Shape: (3, 3)
print(outer)
# [[4, 5, 6],
#  [8, 10, 12],
#  [12, 15, 18]]

# Example 5: Distance matrix
points = np.array([[0, 0],
                   [1, 0],
                   [0, 1]])  # Shape: (3, 2)

# Compute pairwise distances
diff = points[:, np.newaxis, :] - points  # Shape: (3, 3, 2)
distances = np.sqrt((diff ** 2).sum(axis=2))
print(distances)
# [[0, 1, 1],
#  [1, 0, √2],
#  [1, √2, 0]]
```

### Broadcasting Best Practices

```python
# ✅ GOOD: Use broadcasting instead of loops
arr = np.random.randn(1000, 100)
mean = arr.mean(axis=0)  # Shape: (100,)
centered = arr - mean    # Broadcasting

# ❌ BAD: Explicit loop
centered_bad = np.zeros_like(arr)
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        centered_bad[i, j] = arr[i, j] - mean[j]  # 100x slower!

# ✅ GOOD: Use newaxis for clarity
row_vector = np.array([1, 2, 3])
col_vector = row_vector[:, np.newaxis]  # Explicit intent

# ✅ GOOD: Check shapes when debugging
print(f"Array shape: {arr.shape}")
print(f"Mean shape: {mean.shape}")
print(f"Result shape: {(arr - mean).shape}")
```

---

## Vectorization

Vectorization is the practice of replacing explicit Python loops with array operations. This leverages NumPy's C-optimized code.

### Why Vectorize?

```python
import time

# Example: Element-wise square
n = 1000000

# Method 1: Python loop (SLOW)
data = list(range(n))
start = time.time()
result_loop = [x ** 2 for x in data]
time_loop = time.time() - start

# Method 2: NumPy vectorized (FAST)
data_np = np.arange(n)
start = time.time()
result_np = data_np ** 2
time_np = time.time() - start

print(f"Loop: {time_loop:.4f}s")
print(f"Vectorized: {time_np:.4f}s")
print(f"Speedup: {time_loop / time_np:.1f}x")
# Typical speedup: 10-100x
```

### Vectorization Techniques

#### Replace Loops with Array Operations

```python
# ❌ BAD: Explicit loop
arr = np.random.randn(1000)
result = np.zeros_like(arr)
for i in range(len(arr)):
    result[i] = np.sin(arr[i]) ** 2 + np.cos(arr[i]) ** 2

# ✅ GOOD: Vectorized
result = np.sin(arr) ** 2 + np.cos(arr) ** 2

# ❌ BAD: Nested loops for matrix multiplication
A = np.random.randn(100, 50)
B = np.random.randn(50, 200)
C = np.zeros((100, 200))
for i in range(100):
    for j in range(200):
        for k in range(50):
            C[i, j] += A[i, k] * B[k, j]

# ✅ GOOD: Use dot product
C = A @ B  # or np.dot(A, B)
```

#### Use Universal Functions (ufuncs)

```python
# Vectorized conditional logic
arr = np.random.randn(1000)

# ❌ BAD: Loop with if-else
result = np.zeros_like(arr)
for i in range(len(arr)):
    if arr[i] > 0:
        result[i] = arr[i] ** 2
    else:
        result[i] = arr[i] ** 3

# ✅ GOOD: np.where (vectorized ternary)
result = np.where(arr > 0, arr ** 2, arr ** 3)

# Multiple conditions with np.select
conditions = [arr < -1, (arr >= -1) & (arr < 0), arr >= 0]
choices = [arr ** 3, arr, arr ** 2]
result = np.select(conditions, choices)
```

#### Vectorize Custom Functions

```python
# Custom function (scalar)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Already works with NumPy arrays!
arr = np.array([-2, -1, 0, 1, 2])
result = sigmoid(arr)  # Vectorized automatically

# For functions that don't naturally vectorize
def custom_function(x, y):
    if x > y:
        return x - y
    else:
        return x + y

# Vectorize it
vectorized_func = np.vectorize(custom_function)
result = vectorized_func(np.array([1, 2, 3]), np.array([2, 1, 3]))
# Note: np.vectorize is a convenience, not always faster than loops

# Better approach: rewrite to be vectorizable
def custom_function_vec(x, y):
    return np.where(x > y, x - y, x + y)
```

---

## Linear Algebra

NumPy provides comprehensive linear algebra operations.

### Matrix Operations

```python
# Create matrices
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Matrix multiplication (dot product)
C = A @ B          # Preferred in NumPy 1.10+
C = np.dot(A, B)   # Traditional
C = A.dot(B)       # Method form

print(C)
# [[19, 22],
#  [43, 50]]

# Element-wise multiplication (Hadamard product)
elementwise = A * B
print(elementwise)
# [[5, 12],
#  [21, 32]]

# Matrix power
A_squared = np.linalg.matrix_power(A, 2)  # A @ A

# Transpose
A_T = A.T
A_T = np.transpose(A)

# Trace (sum of diagonal)
trace = np.trace(A)  # 1 + 4 = 5

# Determinant
det = np.linalg.det(A)  # -2.0

# Inverse
A_inv = np.linalg.inv(A)
print(A @ A_inv)  # Identity matrix (with floating point errors)

# Verify inverse
identity = np.eye(2)
print(np.allclose(A @ A_inv, identity))  # True

# Solve linear system: Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(x)  # [1, 2]
# Verify: A @ x == b
```

### Eigenvalues and Eigenvectors

```python
A = np.array([[4, 2],
              [1, 3]])

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")      # [5, 2]
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: A @ v = λ @ v
for i in range(len(eigenvalues)):
    lam = eigenvalues[i]
    v = eigenvectors[:, i]
    print(f"A @ v = {A @ v}")
    print(f"λ * v = {lam * v}")
    print(np.allclose(A @ v, lam * v))  # True
```

### Matrix Decompositions

```python
A = np.random.randn(5, 3)

# Singular Value Decomposition (SVD)
U, s, Vt = np.linalg.svd(A, full_matrices=False)
# A ≈ U @ diag(s) @ Vt

# Reconstruct
S = np.diag(s)
A_reconstructed = U @ S @ Vt
print(np.allclose(A, A_reconstructed))  # True

# QR Decomposition
A = np.random.randn(5, 3)
Q, R = np.linalg.qr(A)
# Q: orthogonal matrix
# R: upper triangular
print(np.allclose(A, Q @ R))  # True

# Cholesky Decomposition (for positive definite matrices)
A_sym = np.array([[4, 2],
                  [2, 3]])
L = np.linalg.cholesky(A_sym)
# A = L @ L.T
print(np.allclose(A_sym, L @ L.T))  # True
```

### Norms

```python
v = np.array([3, 4])

# L2 norm (Euclidean)
l2_norm = np.linalg.norm(v)  # 5.0
l2_norm = np.sqrt(np.sum(v ** 2))  # Manual calculation

# L1 norm (Manhattan)
l1_norm = np.linalg.norm(v, ord=1)  # 7.0

# L∞ norm (maximum absolute value)
linf_norm = np.linalg.norm(v, ord=np.inf)  # 4.0

# Matrix norms
A = np.array([[1, 2],
              [3, 4]])
frobenius = np.linalg.norm(A, ord='fro')  # √(1² + 2² + 3² + 4²)
```

---

## Advanced Topics

### Structured Arrays

For heterogeneous data (like database records):

```python
# Define structured dtype
dt = np.dtype([('name', 'U20'),      # Unicode string, 20 chars
               ('age', 'i4'),         # 32-bit integer
               ('salary', 'f8')])     # 64-bit float

# Create structured array
employees = np.array([('Alice', 30, 75000.0),
                      ('Bob', 35, 82000.0),
                      ('Charlie', 28, 68000.0)],
                     dtype=dt)

# Access by field name
print(employees['name'])     # ['Alice', 'Bob', 'Charlie']
print(employees['age'])      # [30, 35, 28]

# Access individual record
print(employees[0])          # ('Alice', 30, 75000.0)
print(employees[0]['name'])  # 'Alice'

# Boolean indexing with fields
high_earners = employees[employees['salary'] > 70000]
print(high_earners['name'])  # ['Alice', 'Bob']

# Sort by field
sorted_employees = np.sort(employees, order='age')
```

### Memory Views and Strides

Understanding memory layout:

```python
arr = np.arange(12).reshape((3, 4))

# Memory layout
print(arr.flags)
# C_CONTIGUOUS: True (row-major, C-style)
# F_CONTIGUOUS: False (column-major, Fortran-style)

# Strides: bytes to step in each dimension
print(arr.strides)  # (32, 8) for int64
# Moving one row: skip 4 elements × 8 bytes = 32 bytes
# Moving one column: skip 1 element × 8 bytes = 8 bytes

# Transpose changes strides, not data
arr_T = arr.T
print(arr_T.strides)  # (8, 32) - reversed!
print(np.shares_memory(arr, arr_T))  # True

# Force Fortran order (column-major)
arr_F = np.array([[1, 2, 3],
                  [4, 5, 6]], order='F')
print(arr_F.flags['F_CONTIGUOUS'])  # True

# Why it matters: cache efficiency
# C-contiguous: fast row operations
# F-contiguous: fast column operations
```

### NumPy and Memory Management

```python
# Check memory usage
arr = np.random.randn(1000, 1000)
print(f"Memory: {arr.nbytes / 1024**2:.2f} MB")  # ~7.6 MB for float64

# Memory-efficient operations
# ❌ BAD: Creates temporary array
result = (arr + 1) * 2 - 3

# ✅ BETTER: In-place operations
result = arr.copy()
result += 1
result *= 2
result -= 3

# ✅ BEST: Single operation
result = arr * 2 + 2 - 3

# Pre-allocate arrays
output = np.empty((1000, 1000))
np.add(arr, 1, out=output)  # Write directly to output

# Memory-mapped files for large data
# Create memory-mapped array
mmap_arr = np.memmap('large_data.dat', dtype='float64', 
                     mode='w+', shape=(10000, 10000))
mmap_arr[:] = np.random.randn(10000, 10000)
mmap_arr.flush()  # Write to disk

# Access memory-mapped array (doesn't load all into RAM)
mmap_arr = np.memmap('large_data.dat', dtype='float64',
                     mode='r', shape=(10000, 10000))
subset = mmap_arr[0:100, 0:100]  # Only this loaded into RAM
```

### Masked Arrays

Handle missing or invalid data:

```python
import numpy.ma as ma

# Create masked array
data = np.array([1, 2, -999, 4, 5, -999, 7])
masked_data = ma.masked_values(data, -999)  # Mask -999 values

print(masked_data)
# [1 2 -- 4 5 -- 7]

# Operations ignore masked values
print(masked_data.mean())  # 3.8 (only valid values)
print(masked_data.sum())   # 19

# Create mask manually
mask = np.array([False, False, True, False, False, True, False])
masked_manual = ma.array(data, mask=mask)

# Filled values
filled = masked_data.filled(0)  # Replace masked with 0
print(filled)  # [1, 2, 0, 4, 5, 0, 7]

# Mask based on condition
arr = np.array([1, 2, 3, 4, 5])
masked_conditional = ma.masked_where(arr > 3, arr)
print(masked_conditional)  # [1 2 3 -- --]
```

---

## Performance Optimization

### Best Practices for Speed

#### 1. Use Vectorized Operations

```python
import numpy as np

# ❌ SLOW: Python loops
def slow_function(arr):
    result = []
    for x in arr:
        result.append(x ** 2 + 2 * x + 1)
    return np.array(result)

# ✅ FAST: Vectorized
def fast_function(arr):
    return arr ** 2 + 2 * arr + 1

arr = np.random.randn(1000000)

import time
start = time.time()
slow_function(arr)
print(f"Slow: {time.time() - start:.4f}s")

start = time.time()
fast_function(arr)
print(f"Fast: {time.time() - start:.4f}s")
```

#### 2. Choose Appropriate Data Types

```python
# Memory and speed comparison
arr_float64 = np.random.randn(1000000).astype(np.float64)
arr_float32 = arr_float64.astype(np.float32)

print(f"float64: {arr_float64.nbytes / 1024**2:.2f} MB")  # ~7.6 MB
print(f"float32: {arr_float32.nbytes / 1024**2:.2f} MB")  # ~3.8 MB

# float32 operations can be faster on some systems
# Trade-off: precision vs. memory/speed
```

#### 3. Avoid Unnecessary Copies

```python
arr = np.random.randn(1000, 1000)

# ✅ GOOD: View (no copy)
view = arr[:500, :]

# ❌ BAD: Unnecessary copy
copy = arr[:500, :].copy()

# ✅ GOOD: In-place operations
arr += 1

# ❌ BAD: Creates new array
arr = arr + 1

# Check if operation returns view or copy
arr = np.arange(10)
view = arr[::2]
print(view.base is arr)  # True for view, False for copy
```

#### 4. Use Appropriate Functions

```python
arr = np.random.randn(1000, 1000)

# ✅ GOOD: Specialized functions
result = np.sum(arr)

# ❌ BAD: Generic reduce
result = np.add.reduce(arr.ravel())

# ✅ GOOD: Boolean operations
count = np.sum(arr > 0)

# ❌ BAD: Loop
count = sum(1 for x in arr.ravel() if x > 0)

# ✅ GOOD: Use @ for matrix multiplication
C = A @ B

# ❌ BAD: Element-wise then sum
C = np.sum(A[:, :, np.newaxis] * B[np.newaxis, :, :], axis=1)
```

#### 5. Leverage NumPy's C-Backend

```python
# ✅ GOOD: NumPy universal functions
result = np.sin(arr) * np.cos(arr)

# ❌ BAD: Python math library
import math
result = np.array([math.sin(x) * math.cos(x) for x in arr.ravel()])
```

### Profiling NumPy Code

```python
import numpy as np

# Simple timing
import time

def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    elapsed = time.time() - start
    return result, elapsed

arr = np.random.randn(1000000)
result, elapsed = measure_time(np.sum, arr)
print(f"Time: {elapsed:.6f}s")

# More accurate: timeit
import timeit

def sum_array():
    arr = np.random.randn(1000000)
    return np.sum(arr)

time_taken = timeit.timeit(sum_array, number=100)
print(f"Average time: {time_taken / 100:.6f}s")

# Line-by-line profiling with line_profiler
# Install: pip install line_profiler
# Usage: kernprof -l -v script.py

# Memory profiling with memory_profiler
# Install: pip install memory_profiler
# Usage: python -m memory_profiler script.py
```

### Numba for Further Acceleration

Numba JIT-compiles Python/NumPy code to machine code:

```python
import numpy as np
from numba import jit

# Regular Python function (slow)
def python_sum(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

# JIT-compiled version (fast)
@jit(nopython=True)
def numba_sum(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total

arr = np.random.randn(1000000)

# First call compiles (slow)
result = numba_sum(arr)

# Subsequent calls are fast
import time
start = time.time()
result = python_sum(arr)
print(f"Python: {time.time() - start:.4f}s")

start = time.time()
result = numba_sum(arr)
print(f"Numba: {time.time() - start:.4f}s")

# NumPy native is still fastest for this simple case
start = time.time()
result = np.sum(arr)
print(f"NumPy: {time.time() - start:.4f}s")

# Numba shines for complex logic that can't be vectorized easily
@jit(nopython=True)
def complex_calculation(arr):
    result = np.empty_like(arr)
    for i in range(len(arr)):
        if arr[i] > 0:
            result[i] = np.sqrt(arr[i])
        else:
            result[i] = arr[i] ** 2
    return result
```

---

## Common Pitfalls and How to Avoid Them

### 1. Unintentional Broadcasting

```python
# ❌ PROBLEM: Unexpected result
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
col_means = arr.mean(axis=0)  # Shape: (3,)

# Intend to subtract column means from each row
centered = arr - col_means  # Works, but might not be obvious

# ✅ SOLUTION: Be explicit
col_means = col_means.reshape(1, -1)  # Shape: (1, 3)
centered = arr - col_means  # Clearer intent

# Verify shapes
print(f"Array shape: {arr.shape}")
print(f"Means shape: {col_means.shape}")
print(f"Result shape: {centered.shape}")
```

### 2. Copy vs View Confusion

```python
# ❌ PROBLEM: Unexpected modification
arr = np.arange(10)
slice_arr = arr[2:5]
slice_arr[0] = 999
print(arr)  # [0, 1, 999, 3, 4, 5, 6, 7, 8, 9] - modified!

# ✅ SOLUTION: Explicit copy when needed
arr = np.arange(10)
slice_arr = arr[2:5].copy()
slice_arr[0] = 999
print(arr)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] - unchanged

# Check if view or copy
print(slice_arr.base is arr)  # True for view, False for copy
```

### 3. Floating Point Precision

```python
# ❌ PROBLEM: Floating point errors
a = 0.1 + 0.2
print(a == 0.3)  # False! (0.30000000000000004)

# ✅ SOLUTION: Use np.isclose or np.allclose
print(np.isclose(a, 0.3))  # True

# For arrays
arr1 = np.array([0.1, 0.2, 0.3])
arr2 = arr1 + 1e-10
print(np.allclose(arr1, arr2))  # True (within tolerance)

# Custom tolerance
print(np.allclose(arr1, arr2, atol=1e-12))  # False
```

### 4. Integer Division Truncation

```python
# ❌ PROBLEM: Unexpected integer division
arr = np.array([1, 2, 3, 4, 5])
result = arr / 2
print(result)  # [0.5, 1.0, 1.5, 2.0, 2.5] - float64

arr_int = np.array([1, 2, 3, 4, 5], dtype=int)
result_int = arr_int / 2
print(result_int)  # Still float! [0.5, 1.0, 1.5, 2.0, 2.5]

# Integer division operator
result_floor = arr_int // 2
print(result_floor)  # [0, 1, 1, 2, 2] - integer floor division

# ✅ SOLUTION: Be aware of types
arr_int = arr_int.astype(float) / 2  # Explicit conversion
```

### 5. Array Comparison Pitfalls

```python
# ❌ PROBLEM: Comparing arrays directly
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])

if arr1 == arr2:  # ValueError: ambiguous!
    print("Equal")

# ✅ SOLUTION: Use appropriate functions
if np.array_equal(arr1, arr2):  # Exact equality
    print("Equal")

if np.allclose(arr1, arr2):  # Tolerance-based
    print("Close")

# Element-wise comparison
comparison = (arr1 == arr2)  # Boolean array
if comparison.all():  # All True?
    print("All elements equal")
```

### 6. Axis Confusion

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# Confusion: which axis?
# axis=0: along rows (result has shape of columns)
# axis=1: along columns (result has shape of rows)

print(np.sum(arr, axis=0))  # [5, 7, 9] - sum each column
print(np.sum(arr, axis=1))  # [6, 15] - sum each row

# ✅ Remember: axis parameter specifies which axis to "collapse"
```

---

## NumPy with Other Libraries

### NumPy and Pandas

```python
import pandas as pd
import numpy as np

# Convert between NumPy and Pandas
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

# NumPy → Pandas DataFrame
df = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# Pandas → NumPy
arr_from_df = df.values  # Returns NumPy array
arr_from_df = df.to_numpy()  # Preferred method

# Pandas Series ↔ NumPy array
series = pd.Series([1, 2, 3, 4, 5])
arr_from_series = series.to_numpy()

# Apply NumPy functions to Pandas
df_normalized = (df - df.mean()) / df.std()  # Works directly!
```

### NumPy and Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data with NumPy
x = np.linspace(0, 2 * np.pi, 1000)
y1 = np.sin(x)
y2 = np.cos(x)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Trigonometric Functions')
plt.show()

# 2D plotting (heatmap)
data = np.random.randn(50, 50)
plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Random Data Heatmap')
plt.show()

# Histogram
data = np.random.randn(10000)
plt.hist(data, bins=50, density=True, alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Normal Distribution')
plt.show()
```

### NumPy and Scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate synthetic data
X = np.random.randn(1000, 5)  # 1000 samples, 5 features
y = X @ np.array([1, 2, 3, 4, 5]) + np.random.randn(1000) * 0.1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict (returns NumPy array)
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

---

## Real-World Use Cases

### 1. Image Processing

```python
import numpy as np
from PIL import Image

# Load image as NumPy array
img = np.array(Image.open('photo.jpg'))
print(f"Image shape: {img.shape}")  # (height, width, channels)

# Grayscale conversion
gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

# Flip image vertically
flipped = np.flipud(img)

# Flip horizontally
flipped_h = np.fliplr(img)

# Rotate 90 degrees
rotated = np.rot90(img)

# Crop image
cropped = img[100:300, 100:300]

# Brightness adjustment
brightened = np.clip(img * 1.5, 0, 255).astype(np.uint8)

# Add noise
noise = np.random.randn(*img.shape) * 25
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)

# Blur (simple averaging)
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
# For actual convolution, use scipy.ndimage.convolve

# Convert back to image
result_img = Image.fromarray(brightened)
result_img.save('output.jpg')
```

### 2. Time Series Analysis

```python
import numpy as np

# Generate synthetic time series
t = np.arange(0, 10, 0.01)
signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
noise = np.random.randn(len(t)) * 0.1
noisy_signal = signal + noise

# Moving average (smoothing)
window_size = 50
smoothed = np.convolve(noisy_signal, np.ones(window_size)/window_size, mode='valid')

# Compute differences (derivatives)
diff = np.diff(signal)

# Cumulative sum (integration)
cumsum = np.cumsum(signal)

# Autocorrelation
def autocorr(x, lag):
    c = np.correlate(x, x, mode='full')
    return c[len(c)//2 + lag] / c[len(c)//2]

lags = range(0, 100)
autocorr_values = [autocorr(signal, lag) for lag in lags]

# FFT (Frequency analysis)
fft = np.fft.fft(signal)
freq = np.fft.fftfreq(len(signal), t[1] - t[0])
power = np.abs(fft) ** 2

# Find dominant frequencies
dominant_freq = freq[np.argsort(power)[-5:]]
print(f"Dominant frequencies: {dominant_freq}")
```

### 3. Financial Data Analysis

```python
import numpy as np

# Stock price simulation (Geometric Brownian Motion)
S0 = 100  # Initial price
mu = 0.1  # Expected return (10%)
sigma = 0.2  # Volatility (20%)
T = 1  # Time period (1 year)
dt = 1/252  # Daily time step (252 trading days)
N = int(T / dt)

# Generate random returns
np.random.seed(42)
returns = np.random.randn(N)

# Cumulative price path
price_path = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * returns))

# Calculate returns
daily_returns = np.diff(np.log(price_path))

# Risk metrics
volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
sharpe_ratio = (np.mean(daily_returns) * 252) / volatility

# Value at Risk (VaR) - 95% confidence
var_95 = np.percentile(daily_returns, 5)

# Maximum drawdown
cummax = np.maximum.accumulate(price_path)
drawdown = (price_path - cummax) / cummax
max_drawdown = np.min(drawdown)

print(f"Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"VaR (95%): {var_95:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Portfolio optimization (simplified)
# Returns matrix: assets × time periods
returns_matrix = np.random.randn(3, 252) * 0.01 + 0.0004  # 3 assets

# Covariance matrix
cov_matrix = np.cov(returns_matrix)

# Random portfolio weights
weights = np.random.random(3)
weights /= weights.sum()  # Normalize to sum to 1

# Portfolio metrics
portfolio_return = np.sum(returns_matrix.mean(axis=1) * weights) * 252
portfolio_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)

print(f"\nPortfolio Return: {portfolio_return:.2%}")
print(f"Portfolio Volatility: {portfolio_vol:.2%}")
```

### 4. Machine Learning - Neural Network from Scratch

```python
import numpy as np

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ReLU activation
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Simple Neural Network class
class NeuralNetwork:
    def __init__(self, layers):
        """
        layers: list of layer sizes, e.g., [2, 4, 1]
        """
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases (He initialization)
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        
        for i in range(len(self.weights)):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            a = sigmoid(z) if i == len(self.weights) - 1 else relu(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        """Backpropagation"""
        m = X.shape[0]
        
        # Output layer error
        delta = (self.activations[-1] - y) * sigmoid_derivative(self.activations[-1])
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients
            dW = self.activations[i].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(self.activations[i])
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """Training loop"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create and train network
nn = NeuralNetwork([2, 4, 1])
nn.train(X, y, epochs=5000, learning_rate=0.1)

# Test
output = nn.forward(X)
print("\nPredictions:")
print(output)
```

### 5. Signal Processing

```python
import numpy as np

# Generate composite signal
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)
freq1, freq2 = 50, 120  # Two frequency components
signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# Add noise
noise = np.random.randn(len(t)) * 0.2
noisy_signal = signal + noise

# Fourier Transform
fft = np.fft.fft(noisy_signal)
freq = np.fft.fftfreq(len(t), 1/fs)

# Power spectrum
power = np.abs(fft) ** 2

# Find peaks (dominant frequencies)
positive_freq_mask = freq > 0
peak_indices = np.argsort(power[positive_freq_mask])[-10:]
dominant_frequencies = freq[positive_freq_mask][peak_indices]

print(f"Dominant frequencies: {dominant_frequencies}")

# Low-pass filter (simple)
cutoff_freq = 80  # Hz
fft_filtered = fft.copy()
fft_filtered[np.abs(freq) > cutoff_freq] = 0

# Inverse FFT
filtered_signal = np.fft.ifft(fft_filtered).real

# Convolution for filtering
kernel = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Moving average
filtered_conv = np.convolve(noisy_signal, kernel, mode='same')

# Cross-correlation (similarity between signals)
cross_corr = np.correlate(signal, filtered_signal, mode='full')
max_corr_idx = np.argmax(cross_corr)
time_lag = (max_corr_idx - len(signal) + 1) / fs

print(f"Time lag at max correlation: {time_lag:.4f}s")
```

---

## NumPy Best Practices Summary

### Code Style and Organization

```python
# ✅ GOOD: Descriptive variable names
temperature_celsius = np.array([20, 25, 30, 35])
temperature_fahrenheit = temperature_celsius * 9/5 + 32

# ❌ BAD: Unclear names
t = np.array([20, 25, 30, 35])
t2 = t * 9/5 + 32

# ✅ GOOD: Use constants
GRAVITY = 9.81  # m/s²
time = np.linspace(0, 10, 100)
height = 0.5 * GRAVITY * time ** 2

# ✅ GOOD: Add docstrings
def normalize_features(X):
    """
    Normalize features to zero mean and unit variance.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Input features
    
    Returns:
    --------
    X_normalized : ndarray of shape (n_samples, n_features)
        Normalized features
    mean : ndarray of shape (n_features,)
        Feature means
    std : ndarray of shape (n_features,)
        Feature standard deviations
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std
```

### Performance Guidelines

1. **Prefer vectorized operations over loops**
2. **Use appropriate data types** (smallest that fits your data)
3. **Avoid unnecessary copies** (use views when possible)
4. **Pre-allocate arrays** when size is known
5. **Use in-place operations** where appropriate
6. **Leverage broadcasting** instead of explicit loops
7. **Use built-in NumPy functions** (they're optimized)
8. **Profile before optimizing** (measure, don't guess)

### Memory Management

```python
# ✅ GOOD: Pre-allocate
n = 1000000
result = np.zeros(n)
for i in range(n):
    result[i] = compute_value(i)

# ❌ BAD: Dynamic appending
result = np.array([])
for i in range(n):
    result = np.append(result, compute_value(i))  # Very slow!

# ✅ GOOD: Use lists for dynamic size, convert once
result_list = []
for i in range(n):
    result_list.append(compute_value(i))
result = np.array(result_list)

# ✅ GOOD: In-place modification
arr = np.random.randn(1000, 1000)
arr += 1  # Modifies arr in place

# ❌ BAD: Creates new array
arr = arr + 1  # Creates new array, more memory
```

### Debugging Tips

```python
# Check array properties
arr = np.random.randn(10, 5)
print(f"Shape: {arr.shape}")
print(f"Dtype: {arr.dtype}")
print(f"Size: {arr.size}")
print(f"Dimensions: {arr.ndim}")
print(f"Strides: {arr.strides}")

# Check for NaN or Inf
print(f"Contains NaN: {np.isnan(arr).any()}")
print(f"Contains Inf: {np.isinf(arr).any()}")

# Find NaN locations
nan_locations = np.argwhere(np.isnan(arr))

# Check if arrays share memory
arr2 = arr[:]
print(f"Share memory: {np.shares_memory(arr, arr2)}")

# Validate shapes before operations
A = np.random.randn(10, 5)
B = np.random.randn(5, 3)
assert A.shape[1] == B.shape[0], f"Incompatible shapes: {A.shape} and {B.shape}"
C = A @ B

# Use assertions for invariants
assert arr.shape[0] > 0, "Array is empty"
assert (arr >= 0).all(), "Array contains negative values"
```

### Testing NumPy Code

```python
import numpy as np
import numpy.testing as npt

def test_normalize():
    """Test normalization function"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    X_norm, mean, std = normalize_features(X)
    
    # Test mean is close to 0
    npt.assert_allclose(X_norm.mean(axis=0), 0, atol=1e-10)
    
    # Test std is close to 1
    npt.assert_allclose(X_norm.std(axis=0), 1, atol=1e-10)
    
    # Test shape preserved
    assert X_norm.shape == X.shape
    
    # Test reconstruction
    X_reconstructed = X_norm * std + mean
    npt.assert_allclose(X_reconstructed, X)

# Run test
test_normalize()
print("All tests passed!")

# Useful testing functions
# npt.assert_array_equal(x, y)        # Exact equality
# npt.assert_array_almost_equal(x, y) # Almost equal (decimals=7)
# npt.assert_allclose(x, y)           # Close (rtol=1e-7, atol=0)
# npt.assert_raises(Exception, func)  # Expect exception
```

---

## Advanced NumPy Patterns

### Design Patterns

#### 1. Factory Pattern for Array Creation

```python
class ArrayFactory:
    """Factory for creating commonly used arrays"""
    
    @staticmethod
    def zeros(shape, dtype=float):
        return np.zeros(shape, dtype=dtype)
    
    @staticmethod
    def random_normal(shape, mean=0, std=1):
        return np.random.randn(*shape) * std + mean
    
    @staticmethod
    def random_uniform(shape, low=0, high=1):
        return np.random.uniform(low, high, shape)
    
    @staticmethod
    def grid_2d(x_range, y_range, num_points):
        """Create 2D grid of points"""
        x = np.linspace(*x_range, num_points)
        y = np.linspace(*y_range, num_points)
        return np.meshgrid(x, y)

# Usage
factory = ArrayFactory()
data = factory.random_normal((100, 50), mean=10, std=2)
```

#### 2. Strategy Pattern for Different Operations

```python
class NormalizationStrategy:
    """Base class for normalization strategies"""
    def normalize(self, X):
        raise NotImplementedError

class ZScoreNormalization(NormalizationStrategy):
    def normalize(self, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        return (X - mean) / std

class MinMaxNormalization(NormalizationStrategy):
    def normalize(self, X):
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)
        return (X - min_val) / (max_val - min_val)

class RobustNormalization(NormalizationStrategy):
    def normalize(self, X):
        median = np.median(X, axis=0)
        iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        return (X - median) / iqr

# Usage
data = np.random.randn(1000, 10)
strategy = ZScoreNormalization()
normalized = strategy.normalize(data)
```

#### 3. Pipeline Pattern for Data Processing

```python
class DataPipeline:
    """Pipeline for sequential data transformations"""
    
    def __init__(self):
        self.steps = []
    
    def add_step(self, name, func):
        self.steps.append((name, func))
        return self
    
    def fit_transform(self, X):
        """Apply all steps sequentially"""
        for name, func in self.steps:
            print(f"Applying: {name}")
            X = func(X)
        return X

# Usage
def remove_outliers(X):
    """Remove outliers beyond 3 standard deviations"""
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    return X[(z_scores < 3).all(axis=1)]

def normalize(X):
    """Z-score normalization"""
    return (X - X.mean(axis=0)) / X.std(axis=0)

def add_polynomial_features(X):
    """Add squared features"""
    return np.hstack([X, X ** 2])

# Create and run pipeline
pipeline = (DataPipeline()
           .add_step("Remove outliers", remove_outliers)
           .add_step("Normalize", normalize)
           .add_step("Add polynomial", add_polynomial_features))

data = np.random.randn(1000, 5)
processed = pipeline.fit_transform(data)
```

### Custom Universal Functions

```python
# Create custom ufunc
def python_sinc(x):
    """Sinc function: sin(x) / x"""
    return np.sin(x) / x if x != 0 else 1.0

# Vectorize (simple but not fastest)
sinc_vectorized = np.vectorize(python_sinc)

# Use on array
x = np.linspace(-10, 10, 1000)
y = sinc_vectorized(x)

# Better: implement directly with NumPy
def numpy_sinc(x):
    """Vectorized sinc function"""
    result = np.sin(x) / x
    result[x == 0] = 1.0  # Handle division by zero
    return result

# Or use np.where
def numpy_sinc_v2(x):
    return np.where(x == 0, 1.0, np.sin(x) / x)

# NumPy actually has sinc built-in
y = np.sinc(x / np.pi)  # Note: np.sinc(x) = sin(πx)/(πx)
```

---

## Ecosystem and Resources

### Essential NumPy Packages

1. **SciPy**: Scientific computing (optimization, integration, signal processing)
2. **Pandas**: Data manipulation and analysis
3. **Matplotlib**: Plotting and visualization
4. **Scikit-learn**: Machine learning
5. **TensorFlow/PyTorch**: Deep learning (use NumPy-like APIs)
6. **Numba**: JIT compilation for NumPy code
7. **CuPy**: GPU-accelerated NumPy
8. **Dask**: Parallel computing with NumPy-like API

### NumPy Configuration

```python
# Check NumPy configuration
import numpy as np
print(np.show_config())

# Set print options
np.set_printoptions(
    precision=4,      # Number of decimal places
    suppress=True,    # Suppress scientific notation
    linewidth=100,    # Characters per line
    threshold=1000,   # Max array elements before summarizing
    edgeitems=3       # Items at edge in summary
)

# Example
arr = np.random.randn(10, 10)
print(arr)

# Temporarily change options
with np.printoptions(precision=2, suppress=False):
    print(arr)

# Reset to defaults
np.set_printoptions(precision=8, suppress=False, threshold=1000)
```

### NumPy Version and Compatibility

```python
import numpy as np

# Check version
print(f"NumPy version: {np.__version__}")

# Check if version supports certain features
from packaging import version
if version.parse(np.__version__) >= version.parse("1.20.0"):
    print("Supports type annotations")

# Deprecation warnings
import warnings
warnings.filterwarnings('default', category=DeprecationWarning)
```

---

## Conclusion

NumPy is the cornerstone of scientific computing in Python, providing:

1. **Efficient arrays**: Memory-efficient, homogeneous, multi-dimensional arrays
2. **Vectorization**: Fast operations without explicit loops
3. **Broadcasting**: Intuitive operations on different-shaped arrays
4. **Rich functionality**: Comprehensive mathematical, statistical, and linear algebra operations
5. **Interoperability**: Seamless integration with the Python scientific ecosystem

### Key Takeaways

- **Think in arrays**, not loops
- **Leverage broadcasting** for elegant, efficient code
- **Understand views vs copies** to avoid bugs
- **Choose appropriate dtypes** for memory efficiency
- **Profile before optimizing** - premature optimization is the root of all evil
- **Use built-in functions** - they're heavily optimized
- **Read the documentation** - NumPy docs are excellent

### Learning Path

1. **Beginner**: Array creation, indexing, basic operations
2. **Intermediate**: Broadcasting, vectorization, linear algebra
3. **Advanced**: Strides, memory layouts, custom ufuncs, performance optimization
4. **Expert**: Contributing to NumPy, understanding C API, advanced memory management

---

## References

1. [NumPy Official Documentation](https://numpy.org/doc/stable/){:target="_blank"}

2. [NumPy User Guide](https://numpy.org/doc/stable/user/index.html){:target="_blank"}

3. [NumPy Reference](https://numpy.org/doc/stable/reference/index.html){:target="_blank"}

4. [NumPy for Absolute Beginners](https://numpy.org/doc/stable/user/absolute_beginners.html){:target="_blank"}

5. [NumPy Tutorials](https://numpy.org/numpy-tutorials/){:target="_blank"}

6. [From Python to NumPy - Nicolas P. Rougier](https://www.labri.fr/perso/nrougier/from-python-to-numpy/){:target="_blank"}

7. [Guide to NumPy - Travis Oliphant](https://web.mit.edu/dvp/Public/numpybook.pdf){:target="_blank"}

8. [NumPy Enhancement Proposals (NEPs)](https://numpy.org/neps/){:target="_blank"}

9. [SciPy Lectures - NumPy](https://scipy-lectures.org/intro/numpy/index.html){:target="_blank"}

10. [Python Data Science Handbook - NumPy](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html){:target="_blank"}

11. [NumPy GitHub Repository](https://github.com/numpy/numpy){:target="_blank"}

12. [NumPy Array Programming Tutorial](https://numpy.org/doc/stable/user/quickstart.html){:target="_blank"}

---

*Document Version: 1.0*  
*Last Updated: November 10, 2025*  
*NumPy Version Covered: 1.26+*
