import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ====== INTERACTIVE IMAGE DIFFERENCE ANALYZER ======
# See EXACTLY what changed between two images!

def visualize_difference_math(img1, img2):
    """
    Show the actual mathematics behind image differencing.
    This is what OpenCV does under the hood!
    """
    print("\n" + "="*70)
    print("🔬 THE MATH BEHIND IMAGE DIFFERENCING")
    print("="*70)
    
    # Get small regions to show pixel values
    h, w = img1.shape[:2]
    sample_y, sample_x = h//2, w//2
    region_size = 5
    
    # Extract small regions
    region1 = img1[sample_y:sample_y+region_size, sample_x:sample_x+region_size, 0]
    region2 = img2[sample_y:sample_y+region_size, sample_x:sample_x+region_size, 0]
    
    print(f"\n📍 Sample Region at pixel ({sample_x}, {sample_y}):")
    print(f"\nImage 1 (Before) - Pixel Values:")
    print(region1)
    print(f"\nImage 2 (After) - Pixel Values:")
    print(region2)
    print(f"\nAbsolute Difference = |Image2 - Image1|:")
    diff_region = np.abs(region2.astype(int) - region1.astype(int))
    print(diff_region)
    
    print(f"\n💡 Algorithm:")
    print(f"   For each pixel (x,y):")
    print(f"   difference[x,y] = |pixel2[x,y] - pixel1[x,y]|")
    print(f"   if difference[x,y] > threshold:")
    print(f"       mark as CHANGED")
    print(f"   else:")
    print(f"       mark as UNCHANGED")
    print("\n" + "="*70)


def compare_two_images(image1_path, image2_path, threshold=30, min_area=100):
    """
    Compare two images and show what changed.
    Perfect for: before/after photos, spot-the-difference, change detection
    
    Args:
        image1_path: Path to first image (before)
        image2_path: Path to second image (after)
        threshold: Pixel difference threshold (0-255, default=30)
        min_area: Minimum changed area in pixels (default=100)
    """
    
    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        print(f"❌ Could not load images!")
        print(f"   Image 1: {image1_path}")
        print(f"   Image 2: {image2_path}")
        return
    
    # Ensure same size
    if img1.shape != img2.shape:
        print(f"⚠️  Resizing images to match...")
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
    
    print(f"\n📸 Images loaded: {img1.shape[1]}x{img1.shape[0]} pixels")
    
    # Show the actual math
    visualize_difference_math(img1, img2)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Clean up noise with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh_clean = cv2.morphologyEx(thresh_clean, cv2.MORPH_CLOSE, kernel)
    
    # Find contours (changed regions)
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze changes
    significant_changes = []
    result_img = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate change intensity
        roi_diff = diff[y:y+h, x:x+w]
        avg_intensity = np.mean(roi_diff)
        
        significant_changes.append({
            'id': i+1,
            'bbox': (x, y, w, h),
            'area': area,
            'intensity': avg_intensity
        })
        
        # Draw on result
        color = (255, 0, 0) if avg_intensity > 50 else (255, 165, 0)
        cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(result_img, f"#{i+1}", (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Calculate statistics
    total_changed_pixels = np.sum(thresh > 0)
    total_pixels = thresh.size
    change_percentage = (total_changed_pixels / total_pixels) * 100
    
    print(f"\n📊 CHANGE ANALYSIS:")
    print(f"   Threshold used: {threshold} (pixel difference)")
    print(f"   Total changed pixels: {total_changed_pixels:,} / {total_pixels:,}")
    print(f"   Overall change: {change_percentage:.2f}%")
    print(f"   Significant regions found: {len(significant_changes)}")
    
    if significant_changes:
        print(f"\n   Top changes:")
        for change in sorted(significant_changes, key=lambda x: x['area'], reverse=True)[:5]:
            x, y, w, h = change['bbox']
            print(f"      #{change['id']}: {int(change['area'])} px² at ({x},{y})")
            print(f"               Intensity: {change['intensity']:.1f}/255")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Original images
    ax1 = plt.subplot(3, 3, 1)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax1.set_title('Image 1: BEFORE', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    ax2.set_title('Image 2: AFTER', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.imshow(result_img)
    ax3.set_title(f'Changes Detected: {len(significant_changes)}', fontsize=14, fontweight='bold', color='red')
    ax3.axis('off')
    
    # Row 2: Processing steps
    ax4 = plt.subplot(3, 3, 4)
    ax4.imshow(gray1, cmap='gray')
    ax4.set_title('Grayscale Before', fontsize=12, fontweight='bold')
    ax4.axis('off')
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.imshow(gray2, cmap='gray')
    ax5.set_title('Grayscale After', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    ax6 = plt.subplot(3, 3, 6)
    diff_colored = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    ax6.imshow(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB))
    ax6.set_title('Difference Heatmap\n(Hot=Changed, Cool=Same)', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # Row 3: Analysis
    ax7 = plt.subplot(3, 3, 7)
    ax7.imshow(diff, cmap='hot')
    ax7.set_title('Raw Difference\n(Brighter=More Change)', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.imshow(thresh, cmap='gray')
    ax8.set_title(f'Binary Threshold\n(Threshold={threshold})', fontsize=12, fontweight='bold')
    ax8.axis('off')
    
    ax9 = plt.subplot(3, 3, 9)
    ax9.imshow(thresh_clean, cmap='gray')
    ax9.set_title('Cleaned Regions\n(Noise Removed)', fontsize=12, fontweight='bold')
    ax9.axis('off')
    
    plt.suptitle('IMAGE DIFFERENCE DETECTION - Visual Pipeline', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('difference_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Analysis saved to: difference_analysis.png")
    plt.show()
    
    return {
        'changes': significant_changes,
        'change_percentage': change_percentage,
        'difference_map': diff,
        'threshold_map': thresh
    }


def create_test_images_from_photo(image_path):
    """
    Create a before/after pair from ONE image by adding changes.
    Great for learning when you don't have two separate photos!
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not load: {image_path}")
        return
    
    print("\n🎨 Creating test image pair...")
    
    # Create "before" image
    before = img.copy()
    
    # Create "after" image with modifications
    after = img.copy()
    h, w = after.shape[:2]
    
    # Add some changes
    # Change 1: Draw rectangle
    cv2.rectangle(after, (w//4, h//4), (w//2, h//2), (0, 255, 0), -1)
    
    # Change 2: Add circle
    cv2.circle(after, (3*w//4, h//4), 50, (255, 0, 0), -1)
    
    # Change 3: Add text
    cv2.putText(after, "CHANGED!", (w//2-100, 3*h//4), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    
    # Change 4: Blur a region
    roi = after[h//2:3*h//4, w//4:w//2]
    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
    after[h//2:3*h//4, w//4:w//2] = blurred_roi
    
    # Save test images
    cv2.imwrite('test_before.jpg', before)
    cv2.imwrite('test_after.jpg', after)
    
    print(f"✅ Created test images:")
    print(f"   test_before.jpg")
    print(f"   test_after.jpg")
    print(f"\nNow run: compare_two_images('test_before.jpg', 'test_after.jpg')")
    
    return 'test_before.jpg', 'test_after.jpg'


def interactive_threshold_demo(image1_path, image2_path):
    """
    Shows how different thresholds affect change detection.
    Helps you understand the threshold parameter!
    """
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    
    if img1 is None or img2 is None:
        print("❌ Could not load images")
        return
    
    if img1.shape != img2.shape:
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    
    # Test different thresholds
    thresholds = [10, 20, 30, 50, 75, 100]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('How Threshold Affects Change Detection', fontsize=16, fontweight='bold')
    
    for idx, thresh_val in enumerate(thresholds):
        _, thresh = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
        changed_pixels = np.sum(thresh > 0)
        change_pct = (changed_pixels / thresh.size) * 100
        
        ax = axes[idx // 3, idx % 3]
        ax.imshow(thresh, cmap='gray')
        ax.set_title(f'Threshold = {thresh_val}\n{change_pct:.2f}% changed', 
                    fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Threshold comparison saved to: threshold_comparison.png")
    plt.show()


# ====== MAIN USAGE ======
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔍 IMAGE DIFFERENCE DETECTION LAB")
    print("="*70)
    print("\n📚 What you'll learn:")
    print("   • How pixel subtraction works mathematically")
    print("   • Why thresholding matters")
    print("   • Real applications: security, quality control, change tracking")
    
    print("\n" + "="*70)
    print("🎯 QUICK START OPTIONS:")
    print("="*70)
    
    print("\n1️⃣  Compare two existing images:")
    print("    compare_two_images('before.jpg', 'after.jpg')")
    
    print("\n2️⃣  Create test images from one photo:")
    print("    create_test_images_from_photo('your_photo.jpg')")
    
    print("\n3️⃣  Understand threshold parameter:")
    print("    interactive_threshold_demo('before.jpg', 'after.jpg')")
    
    print("\n" + "="*70)
    print("💡 BEST PRACTICES:")
    print("="*70)
    print("   • Use images taken from same angle/lighting")
    print("   • Start with threshold=30 (adjust 10-50 for subtle changes)")
    print("   • min_area removes tiny noise (try 50-500)")
    print("   • Good test: take 2 photos of your desk, move something")
    
    print("\n" + "="*70)
    print("🏠 REAL-WORLD APPLICATIONS:")
    print("="*70)
    print("   📦 Warehouse: Detect missing inventory")
    print("   🏗️  Construction: Document progress")
    print("   🔒 Security: Spot intrusions")
    print("   🎮 Gaming: Spot-the-difference games")
    print("   📸 Photography: Before/after edits")
    print("   🏥 Medical: Compare X-rays/scans")
    
    print("\n" + "="*70)
    
    # Example: Uncomment to run
    create_test_images_from_photo('dubai-chocolate.jpg')
    compare_two_images('test_before.jpg', 'test_after.jpg', threshold=30)