import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====== OPENCV VISUAL LEARNING LAB ======
# This script breaks down what OpenCV actually DOES to your images

def show_processing_pipeline(image_path):
    """
    Demonstrates OpenCV operations step-by-step with visual output.
    Use a photo with objects/people for best results!
    """
    
    # Load your image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load image: {image_path}")
        print("💡 Try: 'example.jpg', 'photo.png', or full path like '/Users/you/Pictures/photo.jpg'")
        return
    
    # Convert BGR (OpenCV default) to RGB (for matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"\n📸 Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
    print(f"   Data type: {img.dtype} (values 0-255)")
    print(f"   Memory: {img.nbytes / 1024:.1f} KB\n")
    
    # ====== PIPELINE STAGE 1: Color Spaces ======
    print("🎨 STAGE 1: Understanding Color Spaces")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print(f"   Original: 3 channels (B,G,R) = {img.shape}")
    print(f"   Grayscale: 1 channel = {gray.shape}")
    print(f"   HSV: Hue, Saturation, Value (better for color detection)\n")
    
    # ====== PIPELINE STAGE 2: Edge Detection (Calculus!) ======
    print("📐 STAGE 2: Edge Detection = Finding Gradients")
    
    # Blur first to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection (finds where brightness changes rapidly)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    print(f"   Gaussian Blur: Smooths noise using 2D Gaussian function")
    print(f"   Canny Algorithm: Finds edges using derivative (gradient)")
    print(f"   Math: Edge = where ∂I/∂x or ∂I/∂y is large\n")
    
    # ====== PIPELINE STAGE 3: Morphological Operations ======
    print("🔧 STAGE 3: Morphological Operations (Shape Processing)")
    
    # Create binary image (threshold)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphology kernels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    
    morph_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    morph_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large)
    
    print(f"   Erosion: Shrinks bright regions (removes small noise)")
    print(f"   Dilation: Expands bright regions (fills gaps)")
    print(f"   Opening: Erosion → Dilation (removes noise)")
    print(f"   Closing: Dilation → Erosion (fills holes)\n")
    
    # ====== PIPELINE STAGE 4: Contour Detection ======
    print("🎯 STAGE 4: Contour Detection (Graph Theory)")
    
    contours, hierarchy = cv2.findContours(
        morph_close, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Draw all contours on original image
    contour_img = img_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Analyze significant contours
    significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
    
    print(f"   Found {len(contours)} total contours")
    print(f"   Significant contours (>500px²): {len(significant_contours)}")
    print(f"   Algorithm: Traces connected components (like flood fill)\n")
    
    # ====== PIPELINE STAGE 5: Object Analysis ======
    print("📊 STAGE 5: Object Analysis & Feature Extraction")
    
    analysis_img = img_rgb.copy()
    
    for i, contour in enumerate(significant_contours[:10]):  # Analyze top 10
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Moments (for centroid)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
        
        # Circularity (how round is it?)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Draw analysis
        color = tuple(np.random.randint(100, 255, 3).tolist())
        cv2.rectangle(analysis_img, (x, y), (x+w, y+h), color, 2)
        cv2.circle(analysis_img, (cx, cy), 5, (255, 0, 0), -1)
        
        cv2.putText(analysis_img, f"#{i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        print(f"   Object #{i+1}:")
        print(f"      Area: {area:.0f} px² | Perimeter: {perimeter:.0f} px")
        print(f"      Circularity: {circularity:.2f} (1.0 = perfect circle)")
        print(f"      Aspect Ratio: {w/h:.2f} | Center: ({cx}, {cy})")
    
    # ====== CREATE VISUAL COMPARISON ======
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('OpenCV Processing Pipeline - What Happens to Your Image', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Color spaces
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image\n(RGB Color Space)', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale\n(1 channel: luminance)', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(hsv)
    axes[0, 2].set_title('HSV Color Space\n(Hue-Saturation-Value)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Edge detection
    axes[1, 0].imshow(blurred, cmap='gray')
    axes[1, 0].set_title('Gaussian Blur\n(Noise Reduction)', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(edges, cmap='gray')
    axes[1, 1].set_title('Canny Edges\n(Gradient Detection)', fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(binary, cmap='gray')
    axes[1, 2].set_title('Binary Threshold\n(Otsu\'s Method)', fontweight='bold')
    axes[1, 2].axis('off')
    
    # Row 3: Morphology and contours
    axes[2, 0].imshow(morph_open, cmap='gray')
    axes[2, 0].set_title('Morphological Opening\n(Noise Removal)', fontweight='bold')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(contour_img)
    axes[2, 1].set_title(f'Contour Detection\n({len(contours)} objects found)', fontweight='bold')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(analysis_img)
    axes[2, 2].set_title('Object Analysis\n(Features Extracted)', fontweight='bold')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('opencv_pipeline_output.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Pipeline complete! Saved to: opencv_pipeline_output.png")
    plt.show()
    
    return img_rgb, analysis_img


# ====== BONUS: Real-time Color Detection ======
def color_object_detector(image_path, target_color='red'):
    """
    Shows how OpenCV detects specific colors using HSV color space.
    Try: 'red', 'green', 'blue', 'yellow'
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'green': ([40, 40, 40], [80, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'yellow': ([20, 100, 100], [30, 255, 255])
    }
    
    if target_color not in color_ranges:
        print(f"Color '{target_color}' not available. Try: {list(color_ranges.keys())}")
        return
    
    lower, upper = color_ranges[target_color]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # Apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Find contours of colored objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detection_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(detection_img, (x, y), (x+w, y+h), (255, 0, 0), 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'{target_color.title()} Color Mask')
    axes[1].axis('off')
    
    axes[2].imshow(detection_img)
    axes[2].set_title(f'Detected {target_color.title()} Objects: {len(contours)}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


# ====== RUN THE LAB ======
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 OPENCV VISUAL LEARNING LAB")
    print("="*60)
    print("\n📌 Instructions:")
    print("   1. Use a photo with clear objects/people")
    print("   2. Run: show_processing_pipeline('your_photo.jpg')")
    print("   3. Bonus: color_object_detector('photo.jpg', 'red')")
    print("\n💡 Good test images:")
    print("   - Room photo with furniture")
    print("   - Outdoor scene with people")
    print("   - Objects on a table")
    print("   - Download a sample: https://unsplash.com/photos/")
    print("\n" + "="*60)
    
    # Example usage (uncomment and replace with your image):
    show_processing_pipeline('photo.jpg')
    # color_object_detector('example.jpg', 'blue')