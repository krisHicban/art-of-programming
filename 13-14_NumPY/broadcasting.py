import time
import numpy as np

# ============================================================
# 🎨 AI Image Normalization Demo — NumPy vs Pure Python
# ============================================================
# In deep learning, before sending images into a neural network,
# we "normalize" them channel-wise (R, G, B) so that:
#   - Each channel has zero mean and unit variance
#   - The network trains faster and more stably
#
# The values below are the standard ImageNet normalization stats:
#   mean = average pixel intensity for each color channel
#   std  = standard deviation (spread) for each color channel
# ============================================================

# Simulate a batch of 32 RGB images, 64×64 pixels each
batch = np.random.rand(32, 64, 64, 3)

# ImageNet dataset channel statistics (R, G, B)
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])


# ============================================================
# ⚡ NumPy vectorized normalization
# Broadcasting automatically applies (3,) to every pixel's RGB
# ============================================================
start = time.perf_counter()

normalized_numpy = (batch - mean) / std

numpy_time = time.perf_counter() - start

print("⚡ NumPy vectorized normalization:")
print(f"  → Time: {numpy_time:.6f} seconds\n")


# ============================================================
# 🐍 Pure Python nested loops normalization
# Equivalent to doing everything manually with four nested loops
# (this shows how many operations NumPy replaces)
# ============================================================
start = time.perf_counter()

batch_list = batch.tolist()  # convert NumPy array to native Python lists
normalized_py = [[[[0 for _ in range(3)] for _ in range(64)] for _ in range(64)] for _ in range(32)]

for i in range(32):      # each image
    for y in range(64):  # each row
        for x in range(64):  # each pixel
            for c in range(3):  # each RGB channel
                normalized_py[i][y][x][c] = (batch_list[i][y][x][c] - mean[c]) / std[c]

python_time = time.perf_counter() - start

print("🐍 Pure Python nested loops:")
print(f"  → Time: {python_time:.6f} seconds\n")


# ============================================================
# 📊 Comparison Summary
# ============================================================
speedup = python_time / numpy_time if numpy_time > 0 else float('inf')
print("📊 Comparison Summary:")
print(f"  NumPy is about {speedup:.1f}× faster")

# Example expected output:
# ⚡ NumPy vectorized normalization:
#   → Time: 0.0008 seconds
#
# 🐍 Pure Python nested loops:
#   → Time: 4.9000 seconds
#
# 📊 Comparison Summary:
#   NumPy is about 6,000× faster
