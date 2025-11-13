import time
import sys
import numpy as np

# -----------------------------
# Pure Python implementation
# -----------------------------
start_time = time.perf_counter()

stars_python = []
for i in range(100000):
    distance = (i**2 + (i*2)**2)**0.5
    stars_python.append(distance)

python_time = time.perf_counter() - start_time
python_memory = sys.getsizeof(stars_python)

print("🐍 Pure Python version:")
print(f"  → Time: {python_time:.6f} seconds")
print(f"  → Memory: {python_memory:,} bytes")
print(f"  → Sample distances: {stars_python[:5]}\n")


# -----------------------------
# NumPy vectorized implementation
# -----------------------------
start_time = time.perf_counter()

positions = np.arange(100000, dtype=np.float64)
distances = np.sqrt(positions**2 + (positions*2)**2)

numpy_time = time.perf_counter() - start_time
numpy_memory = distances.nbytes

print("⚡ NumPy vectorized version:")
print(f"  → Time: {numpy_time:.6f} seconds")
print(f"  → Memory: {numpy_memory:,} bytes")
print(f"  → Sample distances: {distances[:5]}\n")


# -----------------------------
# Summary comparison
# -----------------------------
speedup = python_time / numpy_time if numpy_time > 0 else float('inf')
memory_ratio = python_memory / numpy_memory if numpy_memory > 0 else float('inf')

print("📊 Comparison Summary:")
print(f"  NumPy is about {speedup:.1f}× faster")
print(f"  NumPy uses about {memory_ratio:.1f}× less memory")
