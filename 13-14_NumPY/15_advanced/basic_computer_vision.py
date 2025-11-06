"""
COMPUTER VISION - Cum "vede" un computer o imagine

Vom folosi o imagine reală și o vom procesa exact cum o face
un sistem de computer vision (recunoaștere facială, self-driving cars, etc.)

ÎNAINTE DE A RULA:
--------------------
1. Descarcă orice imagine simplă (preferabil cu margini clare)
2. Salvează-o ca 'test_image.jpg' în același folder cu acest script
3. Sau folosește comanda:
   curl -o test_image.jpg https://raw.githubusercontent.com/LaoWater/art-of-programming/refs/heads/main/28_ComputerVision_OpenCV/dubai-chocolate.jpg

Dacă nu ai imagine, scriptul va crea una simplificată pentru demo.
"""

import numpy as np

print("=" * 70)
print("COMPUTER VISION - Cum computerul 'vede' imagini")
print("=" * 70)
print()

# ============================================
# PASUL 1: ÎNCĂRCAREA IMAGINII
# ============================================
print("PASUL 1: Încărcare imagine")
print("-" * 70)

try:
    # Încercăm să încărcăm cu PIL (librăria standard pentru imagini)
    from PIL import Image

    try:
        # Încercăm să deschidem imaginea din folder
        img_pil = Image.open('test_image.jpg').convert('L')  # 'L' = grayscale
        imagine = np.array(img_pil)
        print(f"✓ Imagine încărcată: test_image.jpg")
        print(f"  Dimensiuni: {imagine.shape[0]} x {imagine.shape[1]} pixeli")
        print(f"  Tip: Imagine reală\n")
        using_real_image = True
    except FileNotFoundError:
        print("⚠ Nu am găsit 'test_image.jpg'")
        print("  Creez o imagine demo simplificată...\n")
        using_real_image = False

except ImportError:
    print("⚠ PIL/Pillow nu este instalat")
    print("  Pentru imagini reale, rulează: pip install Pillow")
    print("  Creez o imagine demo simplificată...\n")
    using_real_image = False

# ============================================
# Dacă nu avem imagine reală, creăm una demo
# ============================================
if not using_real_image:
    # Creăm o imagine 20x20 cu o formă simplă (un pătrat)
    # 0 = negru, 255 = alb
    imagine = np.zeros((20, 20), dtype=np.float32)

    # Desenăm un pătrat alb în mijloc
    imagine[5:15, 5:15] = 255

    # Adăugăm un cerc (aproximativ)
    center = 10
    radius = 3
    for i in range(20):
        for j in range(20):
            if (i - center)**2 + (j - 15)**2 < radius**2:
                imagine[i, j] = 200

    print(f"✓ Imagine demo creată: {imagine.shape[0]} x {imagine.shape[1]} pixeli")
    print(f"  (Pentru rezultate mai interesante, adaugă o imagine reală!)\n")

print("Ce este o imagine pentru computer?")
print("→ O MATRICE de numere! Fiecare număr = intensitatea unui pixel")
print(f"→ Valorile sunt între 0 (negru) și 255 (alb)\n")

# Afișăm un mic fragment pentru a ilustra
print("Exemplu - colțul stânga-sus al imaginii (primii 8x8 pixeli):")
fragment = imagine[:8, :8]
print(fragment.astype(int))
print()

# ============================================
# PASUL 2: FILTRELE (Cum vedem "patterns")
# ============================================
print("=" * 70)
print("PASUL 2: FILTRE - Detectarea pattern-urilor")
print("=" * 70)
print()

print("FILTRUL SOBEL pentru detectarea MARGINILOR VERTICALE:")
print("(Marginile = zone unde luminozitatea se schimbă brusc)\n")

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

print("Filtrul Sobel X (3x3):")
print(sobel_x.astype(int))
print()

print("Cum funcționează?")
print("  -1, -2, -1 (stânga)  → detectează pixeli întunecați")
print("   0,  0,  0 (mijloc)  → ignoră centrul")
print("  +1, +2, +1 (dreapta) → detectează pixeli luminoși")
print()
print("Dacă stânga e întunecată și dreapta luminoasă = MARGINE VERTICALĂ!\n")

# Creăm și filtru pentru margini orizontale
sobel_y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

print("Filtrul Sobel Y (3x3) - pentru margini ORIZONTALE:")
print(sobel_y.astype(int))
print()

# ============================================
# PASUL 3: CONVOLUȚIA (Aplicarea filtrului)
# ============================================
print("=" * 70)
print("PASUL 3: CONVOLUȚIA - Aplicarea filtrului pe imagine")
print("=" * 70)
print()

print("Ce este CONVOLUȚIA?")
print("  1. Plasăm filtrul 3x3 peste fiecare porțiune 3x3 din imagine")
print("  2. Înmulțim fiecare valoare din filtru cu pixelul corespunzător")
print("  3. Adunăm toate produsele → un singur număr")
print("  4. Repetăm pentru toată imaginea\n")

def aplica_convolutie(img, filtru):
    """
    Aplică un filtru pe imagine folosind convoluție.

    Aceasta este operația FUNDAMENTALĂ în computer vision!
    - CNN-urile (Convolutional Neural Networks) folosesc sute de astfel de filtre
    - Fiecare filtru învață să detecteze un pattern diferit
    - Filtrele simple detectează margini
    - Filtrele complexe detectează forme, texturi, obiecte
    """
    img_height, img_width = img.shape
    filtru_height, filtru_width = filtru.shape

    # Calculăm dimensiunea output-ului
    output_height = img_height - filtru_height + 1
    output_width = img_width - filtru_width + 1

    # Inițializăm output-ul
    output = np.zeros((output_height, output_width))

    # Aplicăm filtrul pe fiecare poziție
    for i in range(output_height):
        for j in range(output_width):
            # Extragem porțiunea 3x3 din imagine
            region = img[i:i+filtru_height, j:j+filtru_width]

            # Înmulțim element-cu-element și sumăm
            # Aceasta este operația de convoluție!
            output[i, j] = np.sum(region * filtru)

    return output

print("Aplicăm filtrul Sobel X (margini verticale)...")
margini_verticale = aplica_convolutie(imagine, sobel_x)

print("Aplicăm filtrul Sobel Y (margini orizontale)...")
margini_orizontale = aplica_convolutie(imagine, sobel_y)

# Combinăm ambele direcții pentru a obține toate marginile
magnitude_margini = np.sqrt(margini_verticale**2 + margini_orizontale**2)

print("✓ Convoluție completă!\n")

# ============================================
# PASUL 4: REZULTATELE
# ============================================
print("=" * 70)
print("PASUL 4: REZULTATE")
print("=" * 70)
print()

print(f"Imagine originală: {imagine.shape}")
print(f"  Min: {imagine.min():.1f}, Max: {imagine.max():.1f}")
print()

print(f"După detectarea marginilor: {magnitude_margini.shape}")
print(f"  Min: {magnitude_margini.min():.1f}, Max: {magnitude_margini.max():.1f}")
print()

# Afișăm statistici despre marginile detectate
threshold = magnitude_margini.max() * 0.3  # 30% din valoarea maximă
pixeli_margine = np.sum(magnitude_margini > threshold)
total_pixeli = magnitude_margini.size

print(f"Pixeli identificați ca MARGINI: {pixeli_margine}/{total_pixeli} " +
      f"({100*pixeli_margine/total_pixeli:.1f}%)")
print()

# Vizualizare ASCII simplă (dacă imaginea e mică)
if magnitude_margini.shape[0] <= 20 and magnitude_margini.shape[1] <= 20:
    print("Vizualizare ASCII a marginilor detectate:")
    print("(# = margine detectată, . = no edge)\n")

    for i in range(magnitude_margini.shape[0]):
        for j in range(magnitude_margini.shape[1]):
            if magnitude_margini[i, j] > threshold:
                print("#", end=" ")
            else:
                print(".", end=" ")
        print()
    print()

# ============================================
# SALVAREA REZULTATELOR (opțional)
# ============================================
if using_real_image:
    try:
        from PIL import Image

        # Normalizăm la 0-255
        margini_norm = (magnitude_margini - magnitude_margini.min())
        margini_norm = (margini_norm / margini_norm.max() * 255).astype(np.uint8)

        # Salvăm
        img_out = Image.fromarray(margini_norm)
        img_out.save('test_image_edges.jpg')

        print("✓ Margini salvate în: test_image_edges.jpg")
        print("  Compară imaginea originală cu cea procesată!\n")
    except:
        pass

# ============================================
# CONCLUZIE
# ============================================
print("=" * 70)
print("AȘADAR...")
print("=" * 70)
print()
print("Acest proces simplu (convoluție cu un filtru) este FUNDAMENTUL")
print("tuturor sistemelor moderne de computer vision:")
print()
print("  • Recunoașterea facială: sute de filtre învață să detecteze")
print("    ochi, nas, gură, apoi le combină")
print()
print("  • Self-driving cars: filtre detectează linii, mașini, pietoni")
print()
print("  • Diagnosticare medicală: filtre detectează anomalii în")
print("    raze X, MRI, etc.")
print()
print("  • ChatGPT/Claude cu imagini: folosesc CNN-uri pentru a")
print("    'înțelege' ce văd în poze")
print()
print("Diferența? Sistemele reale folosesc:")
print("  - Sute/mii de filtre (nu doar 2)")
print("  - Filtrele sunt ÎNVĂȚATE automat (nu hard-coded)")
print("  - Zeci de layer-uri (nu doar 1)")
print("  - Milioane de parametri antrenați pe milioane de imagini")
print("=" * 70)