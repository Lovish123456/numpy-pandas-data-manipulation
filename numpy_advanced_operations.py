"""
Advanced NumPy Operations
========================

This script demonstrates advanced NumPy operations for scientific computing
and data manipulation, including linear algebra, statistical operations,
and array manipulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import time

print("🔢 Advanced NumPy Operations Demonstration")
print("=" * 50)

# ============================================================================
# 1. Array Creation and Manipulation
# ============================================================================
print("\n1. Array Creation and Manipulation")
print("-" * 35)

# Different ways to create arrays
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 3, 4))
identity_matrix = np.eye(4)
random_array = np.random.rand(3, 3)
arange_array = np.arange(0, 20, 2)
linspace_array = np.linspace(0, 10, 50)

print(f"Zeros array shape: {zeros_array.shape}")
print(f"Ones array shape: {ones_array.shape}")
print(f"Identity matrix:\n{identity_matrix}")
print(f"Random array:\n{random_array.round(3)}")
print(f"Arange array: {arange_array}")
print(f"Linspace array (first 10): {linspace_array[:10].round(2)}")

# Array reshaping and manipulation
original = np.arange(24)
reshaped_2d = original.reshape(4, 6)
reshaped_3d = original.reshape(2, 3, 4)

print(f"\nOriginal array: {original}")
print(f"Reshaped 2D:\n{reshaped_2d}")
print(f"Reshaped 3D shape: {reshaped_3d.shape}")

# Array indexing and slicing
print(f"\nAdvanced indexing:")
print(f"2D array diagonal: {np.diag(reshaped_2d[:4, :4])}")
print(f"Boolean indexing (even numbers): {original[original % 2 == 0]}")

# ============================================================================
# 2. Mathematical Operations
# ============================================================================
print("\n\n2. Mathematical Operations")
print("-" * 30)

# Basic mathematical operations
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8, 9], [10, 11, 12]])

print(f"Array A:\n{a}")
print(f"Array B:\n{b}")
print(f"Element-wise addition:\n{a + b}")
print(f"Element-wise multiplication:\n{a * b}")
print(f"Matrix multiplication:\n{np.dot(a, b.T)}")

# Universal functions (ufuncs)
x = np.linspace(0, 2*np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_exp = np.exp(x/10)

print(f"\nTrigonometric and exponential functions computed for {len(x)} points")
print(f"Sin values range: [{np.min(y_sin):.3f}, {np.max(y_sin):.3f}]")
print(f"Cos values range: [{np.min(y_cos):.3f}, {np.max(y_cos):.3f}]")

# ============================================================================
# 3. Statistical Operations
# ============================================================================
print("\n\n3. Statistical Operations")
print("-" * 28)

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 10000)
data_2d = np.random.normal(0, 1, (100, 50))

print(f"1D Statistical Analysis:")
print(f"  Mean: {np.mean(data):.2f}")
print(f"  Median: {np.median(data):.2f}")
print(f"  Standard Deviation: {np.std(data):.2f}")
print(f"  Variance: {np.var(data):.2f}")
print(f"  Min: {np.min(data):.2f}, Max: {np.max(data):.2f}")
print(f"  25th Percentile: {np.percentile(data, 25):.2f}")
print(f"  75th Percentile: {np.percentile(data, 75):.2f}")

print(f"\n2D Statistical Analysis:")
print(f"  Overall mean: {np.mean(data_2d):.3f}")
print(f"  Mean along axis 0 (columns): shape {np.mean(data_2d, axis=0).shape}")
print(f"  Mean along axis 1 (rows): shape {np.mean(data_2d, axis=1).shape}")
print(f"  Correlation matrix shape: {np.corrcoef(data_2d.T).shape}")

# ============================================================================
# 4. Linear Algebra Operations
# ============================================================================
print("\n\n4. Linear Algebra Operations")
print("-" * 32)

# Matrix operations
A = np.random.randn(4, 4)
b = np.random.randn(4)

# Make A symmetric and positive definite
A = np.dot(A, A.T) + np.eye(4)

print(f"Matrix A shape: {A.shape}")
print(f"Vector b shape: {b.shape}")

# Linear algebra operations
det_A = np.linalg.det(A)
inv_A = np.linalg.inv(A)
eigenvals, eigenvecs = np.linalg.eig(A)
solution = np.linalg.solve(A, b)

print(f"Determinant of A: {det_A:.3f}")
print(f"Condition number: {np.linalg.cond(A):.3f}")
print(f"Eigenvalues: {eigenvals.round(3)}")
print(f"Solution to Ax = b: {solution.round(3)}")

# Verify solution
verification = np.dot(A, solution)
print(f"Verification ||Ax - b||: {np.linalg.norm(verification - b):.2e}")

# SVD decomposition
U, s, Vt = np.linalg.svd(A)
print(f"SVD shapes: U{U.shape}, s{s.shape}, Vt{Vt.shape}")

# ============================================================================
# 5. Advanced Array Operations
# ============================================================================
print("\n\n5. Advanced Array Operations")
print("-" * 32)

# Broadcasting
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

broadcast_result = arr_2d + arr_1d
print(f"Broadcasting result:\n{broadcast_result}")

# Fancy indexing
large_array = np.random.randint(0, 100, (10, 10))
indices = np.array([1, 3, 5, 7])
selected_rows = large_array[indices]

print(f"Original array shape: {large_array.shape}")
print(f"Selected rows shape: {selected_rows.shape}")

# Conditional operations
mask = large_array > 50
high_values = large_array[mask]
print(f"Values > 50: {len(high_values)} out of {large_array.size}")

# Where function
result = np.where(large_array > 50, large_array, 0)
print(f"Conditional replacement completed")

# ============================================================================
# 6. Performance Comparisons
# ============================================================================
print("\n\n6. Performance Comparisons")
print("-" * 29)

# NumPy vs Pure Python performance
size = 1000000
np_array = np.random.rand(size)
python_list = np_array.tolist()

# NumPy sum
start_time = time.time()
np_sum = np.sum(np_array)
np_time = time.time() - start_time

# Python sum
start_time = time.time()
py_sum = sum(python_list)
py_time = time.time() - start_time

print(f"Array size: {size:,}")
print(f"NumPy sum time: {np_time:.6f} seconds")
print(f"Python sum time: {py_time:.6f} seconds")
print(f"NumPy is {py_time/np_time:.1f}x faster")

# Vectorized operations
def python_operation(arr):
    return [x**2 + 2*x + 1 for x in arr]

def numpy_operation(arr):
    return arr**2 + 2*arr + 1

test_array = np.random.rand(100000)
test_list = test_array.tolist()

# Time Python operation
start_time = time.time()
py_result = python_operation(test_list)
py_vec_time = time.time() - start_time

# Time NumPy operation
start_time = time.time()
np_result = numpy_operation(test_array)
np_vec_time = time.time() - start_time

print(f"\nVectorized operations comparison:")
print(f"Python list comprehension: {py_vec_time:.6f} seconds")
print(f"NumPy vectorized: {np_vec_time:.6f} seconds")
print(f"NumPy is {py_vec_time/np_vec_time:.1f}x faster")

# ============================================================================
# 7. Specialized Arrays and Functions
# ============================================================================
print("\n\n7. Specialized Arrays and Functions")
print("-" * 36)

# Structured arrays
dtype = [('name', 'U10'), ('age', 'i4'), ('salary', 'f8')]
employees = np.array([
    ('Alice', 25, 50000.0),
    ('Bob', 30, 60000.0),
    ('Charlie', 35, 70000.0)
], dtype=dtype)

print(f"Structured array:")
print(f"Names: {employees['name']}")
print(f"Average age: {np.mean(employees['age']):.1f}")
print(f"Total salary: ${np.sum(employees['salary']):,.2f}")

# Masked arrays
data_with_missing = np.array([1, 2, -999, 4, 5, -999, 7])
masked_array = np.ma.masked_where(data_with_missing == -999, data_with_missing)
print(f"\nMasked array: {masked_array}")
print(f"Mean (ignoring masked values): {np.ma.mean(masked_array):.2f}")

# Polynomial operations
coefficients = np.array([1, -2, 1])  # x^2 - 2x + 1
x_values = np.linspace(-2, 4, 100)
y_values = np.polyval(coefficients, x_values)
print(f"\nPolynomial evaluation completed for {len(x_values)} points")

# Polynomial fitting
x_data = np.array([0, 1, 2, 3, 4])
y_data = np.array([1, 2.1, 3.9, 9.1, 16.2])
fit_coeffs = np.polyfit(x_data, y_data, 2)
print(f"Polynomial fit coefficients: {fit_coeffs.round(3)}")

# ============================================================================
# 8. Array Manipulation Techniques
# ============================================================================
print("\n\n8. Array Manipulation Techniques")
print("-" * 35)

# Stacking and splitting
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

vstacked = np.vstack((arr1, arr2))
hstacked = np.hstack((arr1, arr2))
dstacked = np.dstack((arr1, arr2))

print(f"Original arrays shape: {arr1.shape}")
print(f"Vertical stack shape: {vstacked.shape}")
print(f"Horizontal stack shape: {hstacked.shape}")
print(f"Depth stack shape: {dstacked.shape}")

# Splitting
large_arr = np.arange(20).reshape(4, 5)
split_arrays = np.hsplit(large_arr, 5)
print(f"Split into {len(split_arrays)} arrays")

# Sorting
unsorted = np.random.randint(0, 100, 20)
sorted_arr = np.sort(unsorted)
sort_indices = np.argsort(unsorted)

print(f"\nSorting demonstration:")
print(f"Original: {unsorted}")
print(f"Sorted: {sorted_arr}")
print(f"Sort indices: {sort_indices}")

# ============================================================================
# 9. Random Number Generation
# ============================================================================
print("\n\n9. Random Number Generation")
print("-" * 30)

# Set random seed for reproducibility
np.random.seed(12345)

# Different distributions
normal_samples = np.random.normal(0, 1, 1000)
uniform_samples = np.random.uniform(-1, 1, 1000)
exponential_samples = np.random.exponential(2, 1000)
binomial_samples = np.random.binomial(10, 0.3, 1000)

print(f"Random sampling results:")
print(f"Normal distribution mean: {np.mean(normal_samples):.3f}")
print(f"Uniform distribution range: [{np.min(uniform_samples):.3f}, {np.max(uniform_samples):.3f}]")
print(f"Exponential distribution mean: {np.mean(exponential_samples):.3f}")
print(f"Binomial distribution mean: {np.mean(binomial_samples):.3f}")

# Random choice and permutation
choices = np.random.choice(['A', 'B', 'C', 'D'], size=20, p=[0.4, 0.3, 0.2, 0.1])
permutation = np.random.permutation(np.arange(10))

print(f"\nRandom choices: {choices}")
print(f"Random permutation: {permutation}")

print("\n" + "=" * 50)
print("Advanced NumPy Operations Complete!")
print("=" * 50)

print(f"\n📋 NumPy Operations Covered:")
print(f"✅ Array creation and manipulation")
print(f"✅ Mathematical and statistical operations")
print(f"✅ Linear algebra computations")
print(f"✅ Broadcasting and vectorization")
print(f"✅ Performance optimization")
print(f"✅ Specialized arrays and functions")
print(f"✅ Random number generation")
print(f"✅ Advanced indexing and slicing")
