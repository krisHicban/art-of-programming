# Artifact: https://claude.ai/public/artifacts/fabc2efd-ee53-4e50-ae33-59a215c0159c


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (health app context: workout form analysis)
workout_image = cv2.imread('dubai-chocolate.jpg')
workout_rgb = cv2.cvtColor(workout_image, cv2.COLOR_BGR2RGB)

# ====== The Convolution Operation ======
# A filter/kernel is a small matrix that slides across the image

# Example: 3x3 blur kernel (averaging)
blur_kernel = np.ones((3, 3), np.float32) / 9
print("🔍 Blur Kernel (averages surrounding pixels):")
print(blur_kernel)

# ====== Common Filters Explained ======

# 1. GAUSSIAN BLUR - Removes noise, smooths image
gaussian = cv2.GaussianBlur(workout_rgb, (5, 5), 0)

# 2. BILATERAL FILTER - Edge-preserving smoothing
# Used in: Portrait mode, keeping subject sharp while blurring background
bilateral = cv2.bilateralFilter(workout_rgb, 9, 75, 75)

# 3. SHARPENING - Enhances edges
sharpen_kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]])
sharpened = cv2.filter2D(workout_rgb, -1, sharpen_kernel)

# ====== Edge Detection - Finding Boundaries ======
gray = cv2.cvtColor(workout_rgb, cv2.COLOR_RGB2GRAY)

# CANNY - Multi-stage edge detection (best for most uses)
# Used in: Lane detection (autonomous vehicles), document scanning
canny = cv2.Canny(gray, 100, 200)

# ====== Real Health App Use Case: Posture Detection ======
def detect_workout_form_edges(image_rgb):
    """Detect body edges in workout video to analyze form"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    edge_percentage = (np.sum(edges > 0) / edges.size) * 100

    return {
        'edges': edges,
        'edge_density': edge_percentage,
        'body_detected': edge_percentage > 5,
        'form_clarity': 'Good' if 5 < edge_percentage < 15 else 'Reposition camera'
    }

form_analysis = detect_workout_form_edges(workout_rgb)
print(f"\n🏋️ Workout Form Analysis:")
print(f"   Edge Density: {form_analysis['edge_density']:.2f}%")
print(f"   {form_analysis['form_clarity']}")

print("\n💡 Convolution = Sliding dot product (matrix multiplication)")
print("   Filters = Small matrices encoding transformations")
print("\n✨ You're mastering the MATHEMATICS of VISION!")