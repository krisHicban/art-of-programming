# Instead of counting every slight pixel variation, we cluster similar colors to find the dominant colors that actually matter:
# Distribution Evenness (Entropy)
# Uses information theory to measure how evenly distributed colors are:

# All colors equal proportion → High score ✅
# One dominant color → Low score ❌

# Artifact: https://claude.ai/public/artifacts/b4221d87-a840-4af3-856c-9a45bf859794
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def analyze_meal_variety_smart(image_rgb, debug=False):
    """
    Intelligent meal variety analysis using:
    1. K-means clustering to find DOMINANT colors (not just unique values)
    2. Color distribution analysis (how much of each color?)
    3. Spatial analysis (are colors spread out or clustered?)
    4. Nutritional color mapping (green veggies, orange/red fruits, etc.)
    """
    
    # Step 1: Resize for faster processing (optional but smart)
    h, w = image_rgb.shape[:2]
    scale = 400 / max(h, w)
    resized = cv2.resize(image_rgb, (int(w*scale), int(h*scale)))
    
    # Step 2: Convert to HSV for better color analysis
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    
    # Step 3: K-means clustering to find DOMINANT colors
    pixels = resized.reshape(-1, 3)
    n_colors = 8  # Find top 8 dominant colors
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get color proportions
    labels = kmeans.labels_
    counts = np.bincount(labels)
    percentages = counts / counts.sum() * 100
    
    # Step 4: Filter out dominant colors (must be >5% of image to count)
    significant_threshold = 5.0
    significant_colors = percentages > significant_threshold
    n_significant = significant_colors.sum()
    
    # Step 5: Analyze color diversity in HSV space
    dominant_colors_hsv = []
    for center in kmeans.cluster_centers_[significant_colors]:
        hsv_center = cv2.cvtColor(
            np.uint8([[center]]), 
            cv2.COLOR_RGB2HSV
        )[0, 0]
        dominant_colors_hsv.append(hsv_center)
    
    # Step 6: Map to nutritional categories
    nutrition_categories = {
        'Greens': 0,      # Hue 40-80 (vegetables)
        'Reds/Oranges': 0, # Hue 0-20, 160-180 (fruits, proteins)
        'Yellows': 0,     # Hue 20-40 (grains, some fruits)
        'Browns': 0,      # Low saturation (whole grains, proteins)
        'Whites': 0       # High value, low saturation (dairy, grains)
    }
    
    for hsv_color, pct in zip(dominant_colors_hsv, percentages[significant_colors]):
        h, s, v = hsv_color
        
        if s < 30 and v > 200:  # White/light colors
            nutrition_categories['Whites'] += pct
        elif s < 50 and v < 120:  # Brown colors
            nutrition_categories['Browns'] += pct
        elif 40 <= h <= 80:  # Greens
            nutrition_categories['Greens'] += pct
        elif h <= 20 or h >= 160:  # Reds/Oranges
            nutrition_categories['Reds/Oranges'] += pct
        elif 20 < h < 40:  # Yellows
            nutrition_categories['Yellows'] += pct
    
    # Step 7: Calculate variety score using multiple factors
    # Factor 1: Number of significant colors (0-40 points)
    color_count_score = min(40, n_significant * 10)
    
    # Factor 2: Distribution evenness (0-30 points)
    # Use entropy - more even = better
    sig_pcts = percentages[significant_colors] / 100
    entropy = -np.sum(sig_pcts * np.log2(sig_pcts + 1e-10))
    max_entropy = np.log2(n_significant) if n_significant > 1 else 1
    evenness_score = (entropy / max_entropy) * 30 if max_entropy > 0 else 0
    
    # Factor 3: Nutritional category coverage (0-30 points)
    categories_present = sum(1 for v in nutrition_categories.values() if v > 3)
    category_score = min(30, categories_present * 10)
    
    variety_score = color_count_score + evenness_score + category_score
    
    # Generate insights
    insights = []
    if nutrition_categories['Greens'] < 10:
        insights.append("🥬 Add more green vegetables")
    if nutrition_categories['Reds/Oranges'] < 10:
        insights.append("🍅 Add colorful fruits or proteins")
    if n_significant < 3:
        insights.append("🌈 Meal is too monotone - diversify!")
    if nutrition_categories['Browns'] > 50:
        insights.append("⚠️ Too much brown - add fresh colors")
    if not insights:
        insights.append("✅ Excellent color variety!")
    
    results = {
        'variety_score': variety_score,
        'significant_colors': int(n_significant),
        'nutrition_breakdown': {k: f"{v:.1f}%" for k, v in nutrition_categories.items() if v > 2},
        'insights': insights,
        'color_distribution': percentages[significant_colors].tolist()
    }
    
    if debug:
        # Visualize dominant colors
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(resized)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Show color palette
        palette = kmeans.cluster_centers_[significant_colors].astype(int)
        color_bar = np.zeros((100, 500, 3), dtype=np.uint8)
        x_start = 0
        for i, (color, pct) in enumerate(zip(palette, percentages[significant_colors])):
            x_end = x_start + int(500 * pct / 100)
            color_bar[:, x_start:x_end] = color
            x_start = x_end
        
        axes[1].imshow(color_bar)
        axes[1].set_title(f"Dominant Colors (Score: {variety_score:.1f}/100)")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()
    
    return results


# Example usage
if __name__ == "__main__":
    # Test with your images
    img1 = cv2.cvtColor(cv2.imread('data-calculus-hero.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('dubai-chocolate.jpg'), cv2.COLOR_BGR2RGB)
    
    print("=" * 60)
    print("IMAGE 1 ANALYSIS:")
    print("=" * 60)
    result1 = analyze_meal_variety_smart(img1, debug=True)
    print(f"\n🎯 Variety Score: {result1['variety_score']:.1f}/100")
    print(f"🎨 Significant Colors: {result1['significant_colors']}")
    print(f"📊 Nutrition Breakdown: {result1['nutrition_breakdown']}")
    print(f"💡 Insights:")
    for insight in result1['insights']:
        print(f"   {insight}")
    
    print("\n" + "=" * 60)
    print("IMAGE 2 ANALYSIS:")
    print("=" * 60)
    result2 = analyze_meal_variety_smart(img2, debug=True)
    print(f"\n🎯 Variety Score: {result2['variety_score']:.1f}/100")
    print(f"🎨 Significant Colors: {result2['significant_colors']}")
    print(f"📊 Nutrition Breakdown: {result2['nutrition_breakdown']}")
    print(f"💡 Insights:")
    for insight in result2['insights']:
        print(f"   {insight}")