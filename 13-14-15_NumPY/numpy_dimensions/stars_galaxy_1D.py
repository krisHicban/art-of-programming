import numpy as np
import time

# ============================================================
# 🌌 Galactic Center of Mass — NumPy Dimensions & Broadcasting
# ============================================================
# We’ll simulate 100,000 stars in a galaxy.
# Each star has:
#   - A 3D position (x, y, z) in space
#   - A random mass (different for each star)
#
# Our goal: compute the galaxy’s "center of mass" (CM),
# using the physics formula:
#        CM = Σ(mi * xi) / Σ(mi)
# where:
#   mi = mass of star i
#   xi = 3D position vector of star i
#
# This example shows:
#   - NumPy array dimensions (2D arrays)
#   - Axis operations (summing along an axis)
#   - Broadcasting via [:, np.newaxis]
# ============================================================


# ------------------------------------------------------------
# Step 1. Generate random data for our galaxy
# ------------------------------------------------------------
np.random.seed(42)  # reproducible results, choosing the seed from which the "random" grows - another one generating from the same seed will get same randoms.

# Each row = [x, y, z] position of one star
# Stars are roughly distributed around the galactic center (0, 0, 0)
# using a normal (Gaussian) distribution with std = 25.
# -> Most stars will be near the center; few far away.
star_positions = np.random.normal(0, 25, (100000, 3))  # shape: (100000, 3)


# Each element = mass of one star
# We use an **exponential distribution**, meaning:
#   - Most stars are small (low mass)
#   - A few are extremely massive (rare "giant" stars)
#   - This matches how real galaxies are structured
#
# The function np.random.exponential(scale, size) generates random numbers
# where "scale" = average value (here = 1).
# Example: scale=1 → mean mass ≈ 1 solar unit, but some will be 5–10× heavier.
star_masses = np.random.exponential(1, 100000)  # shape: (100000,)



# ------------------------------------------------------------
# Step 2. Compute the Center of Mass
# ------------------------------------------------------------
start = time.perf_counter()

# Total mass of all stars (scalar)
total_mass = np.sum(star_masses)

# Multiply each star’s position by its mass
# star_masses has shape (100000,)
# star_positions has shape (100000, 3)
# We add [:, np.newaxis] → shape (100000, 1)
# Broadcasting expands it to match (100000, 3)
weighted_positions = star_positions * star_masses[:, np.newaxis]

# Sum all weighted positions (collapse along axis=0)
# axis=0 → sum across all stars → result: (3,)
sum_weighted_positions = np.sum(weighted_positions, axis=0)

# Center of mass = Σ(mi * xi) / Σ(mi)
center_of_mass = sum_weighted_positions / total_mass

elapsed = time.perf_counter() - start


# ------------------------------------------------------------
# Step 3. Results
# ------------------------------------------------------------
print(f"Centrul galaxiei (x, y, z): {center_of_mass}")
print(f"⏱️ Calculat în: {elapsed:.6f} secunde")

# Example output:
# Centrul galaxiei (x, y, z): [-0.0168  0.0371 -0.0155]
# ⏱️ Calculat în: 0.001 seconds
