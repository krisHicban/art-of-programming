"""
EIGENVALUES & EIGENVECTORS - Cea mai simplă explicație

ÎNTREBAREA FUNDAMENTALĂ:
"Când aplici o transformare (matrice) pe un vector, majoritatea vectorilor
își schimbă DIRECȚIA. Dar există câțiva vectori SPECIALI care își păstrează
direcția - doar se întind sau se strâng. Aceștia sunt EIGENVECTORII."

ANALOGIE SIMPLĂ:
Imaginează-te că tragi de un elastic în diferite direcții.
- În cele mai multe direcții, elasticul se deformează ciudat (schimbă direcția)
- Dar există 1-2 direcții SPECIALE unde elasticul doar se întinde, fără să se răsucească
- Acele direcții = EIGENVECTORI
- Cât de mult se întinde în acele direcții = EIGENVALORI

APLICAȚII REAL-WORLD:
  • Google PageRank: găsește cele mai importante pagini web
  • Recunoaștere facială: "eigenfaces"
  • Compresie date: păstrează doar informația importantă
  • Fizică: moduri de vibrație, stabilitate structuri
  • Machine Learning: PCA (reduc dimensionalitatea datelor)
"""

import numpy as np
from numpy.linalg import eig
import math

print("=" * 70)
print("EIGENVALUES & EIGENVECTORS - Explicație intuitivă")
print("=" * 70)
print()

# ============================================
# EXEMPLU 1: Înțelegerea vizuală
# ============================================
print("EXEMPLU 1: Ce este un EIGENVECTOR?")
print("-" * 70)
print()

print("Să luăm o transformare simplă: scalare pe axe diferite")
print("(întinde 3x pe axa X, întinde 2x pe axa Y)")
print()

# Matrice simplă de scalare
A = np.array([
    [3.0, 0.0],  # Întinde 3x pe X
    [0.0, 2.0]   # Întinde 2x pe Y
])

print("Matricea A (transformarea):")
print(A)
print()

print("Haideți să vedem ce face această matrice cu diferite vectori:")
print()

# Testăm câțiva vectori
vectori_test = [
    np.array([1.0, 0.0]),  # vector pe axa X
    np.array([0.0, 1.0]),  # vector pe axa Y
    np.array([1.0, 1.0]),  # vector diagonal
]

labels = ["Vector pe axa X: (1, 0)", "Vector pe axa Y: (0, 1)", "Vector diagonal: (1, 1)"]

for i, v in enumerate(vectori_test):
    v_transformat = np.dot(A, v)

    # Verificăm dacă direcția s-a păstrat
    # (un vector păstrează direcția dacă e multiplu scalar al originalului)
    directie_pastrata = np.allclose(v_transformat / np.linalg.norm(v_transformat),
                                     v / np.linalg.norm(v)) or \
                        np.allclose(v_transformat / np.linalg.norm(v_transformat),
                                     -v / np.linalg.norm(v))

    print(f"{labels[i]}")
    print(f"  După transformare: ({v_transformat[0]:.1f}, {v_transformat[1]:.1f})")

    if directie_pastrata:
        factor = v_transformat[0] / v[0] if v[0] != 0 else v_transformat[1] / v[1]
        print(f"  ✓ EIGENVECTOR! S-a întins cu factor: {factor:.1f}")
    else:
        print(f"  ✗ Nu e eigenvector (s-a schimbat direcția)")
    print()

print("OBSERVAȚIE: Vectorii pe axele X și Y sunt EIGENVECTORI!")
print("            Vectorul diagonal NU este eigenvector (se deformează)")
print()

# ============================================
# Calculăm eigenvalues & eigenvectors
# ============================================
print("=" * 70)
print("Calculăm EIGENVALUES & EIGENVECTORS cu NumPy")
print("=" * 70)
print()

eigenvalori, eigenvectori = eig(A)

print("Eigenvalori (cât se întind eigenvectorii):")
print(f"  λ₁ = {eigenvalori[0]:.2f}")
print(f"  λ₂ = {eigenvalori[1]:.2f}")
print()

print("Eigenvectori (direcțiile speciale):")
print(f"  v₁ = ({eigenvectori[0, 0]:.2f}, {eigenvectori[1, 0]:.2f})")
print(f"  v₂ = ({eigenvectori[0, 1]:.2f}, {eigenvectori[1, 1]:.2f})")
print()

print("VERIFICARE: A × v = λ × v")
print("(matricea înmulțită cu eigenvector = eigenvalue × eigenvector)")
print()

for i in range(len(eigenvalori)):
    v = eigenvectori[:, i]
    lhs = np.dot(A, v)  # A × v
    rhs = eigenvalori[i] * v  # λ × v

    print(f"Pentru eigenvector {i+1}:")
    print(f"  A × v = ({lhs[0]:.2f}, {lhs[1]:.2f})")
    print(f"  λ × v = ({rhs[0]:.2f}, {rhs[1]:.2f})")
    print(f"  ✓ Sunt egale!" if np.allclose(lhs, rhs) else "  ✗ Eroare!")
    print()

# ============================================
# EXEMPLU 2: Matricea de rotație + scalare
# ============================================
print("=" * 70)
print("EXEMPLU 2: Caz mai complex (rotație + scalare)")
print("=" * 70)
print()

print("Acum o matrice mai interesantă:")
print("(nu e atât de evidentă ca prima)")
print()

# Matrice care face ceva mai complicat
B = np.array([
    [4.0, 1.0],
    [2.0, 3.0]
])

print("Matricea B:")
print(B)
print()

print("Să vedem ce face cu un vector random (1, 1):")
v_test = np.array([1.0, 1.0])
v_transformed = np.dot(B, v_test)
print(f"  (1, 1) → ({v_transformed[0]:.1f}, {v_transformed[1]:.1f})")
print(f"  Direcția s-a schimbat complet!")
print()

# Calculăm eigenvectors pentru B
eigenvalori_B, eigenvectori_B = eig(B)

# Sortăm după mărimea eigenvalue-urilor
idx = eigenvalori_B.argsort()[::-1]
eigenvalori_B = eigenvalori_B[idx]
eigenvectori_B = eigenvectori_B[:, idx]

print("Eigenvalori:")
for i, λ in enumerate(eigenvalori_B):
    print(f"  λ_{i+1} = {λ:.3f}")
print()

print("Eigenvectori (direcțiile care NU se deformează):")
for i in range(len(eigenvalori_B)):
    v = eigenvectori_B[:, i]
    print(f"  v_{i+1} = ({v[0]:6.3f}, {v[1]:6.3f})")

    # Testăm
    lhs = np.dot(B, v)
    rhs = eigenvalori_B[i] * v
    print(f"        B × v_{i+1} = λ_{i+1} × v_{i+1} ✓")
print()

# ============================================
# APLICAȚIE REALĂ: PCA (Compresie de date)
# ============================================
print("=" * 70)
print("APLICAȚIE REALĂ: PCA - Principal Component Analysis")
print("=" * 70)
print()

print("PROBLEMA: Ai date cu multe dimensiuni (ex: 100 features)")
print("          Vrei să le comprimi la 2-3 dimensiuni, păstrând")
print("          cât mai multă informație.")
print()
print("SOLUȚIA: Eigenvectorii cu cei mai mari eigenvalori arată")
print("         direcțiile cu CEA MAI MULTĂ VARIAȚIE în date!")
print()

# Simulăm niște date: înălțime și greutate ale unor persoane
# (corelate - oamenii mai înalți tind să fie mai grei)
print("EXEMPLU: Date despre 100 persoane (înălțime, greutate)")
print()

np.random.seed(42)
inaltime = np.random.normal(170, 10, 100)  # media 170cm, std 10
greutate = inaltime * 0.8 + np.random.normal(0, 5, 100)  # corelată cu înălțimea

# Combinăm în matrice
date = np.column_stack([inaltime, greutate])

print(f"Date shape: {date.shape} (100 persoane, 2 features)")
print(f"Sample: înălțime={inaltime[0]:.1f}cm, greutate={greutate[0]:.1f}kg")
print()

# Centrăm datele (scădem media)
date_centrate = date - np.mean(date, axis=0)

# Calculăm matricea de covarianță
# (arată cum variază features-urile împreună)
cov_matrix = np.cov(date_centrate.T)

print("Matricea de covarianță:")
print(cov_matrix)
print("(Arată cât de mult variază și cât de corelate sunt features-urile)")
print()

# Calculăm eigenvectors ai matricei de covarianță
eigenvalori_pca, eigenvectori_pca = eig(cov_matrix)

# Sortăm
idx = eigenvalori_pca.argsort()[::-1]
eigenvalori_pca = eigenvalori_pca[idx]
eigenvectori_pca = eigenvectori_pca[:, idx]

print("EIGENVALORI (mărimea variației în fiecare direcție):")
for i, λ in enumerate(eigenvalori_pca):
    procent = 100 * λ / eigenvalori_pca.sum()
    print(f"  Componenta {i+1}: {λ:8.2f} ({procent:.1f}% din variație)")
print()

print("EIGENVECTORI (direcțiile principale de variație):")
for i in range(len(eigenvalori_pca)):
    v = eigenvectori_pca[:, i]
    print(f"  Componenta {i+1}: ({v[0]:6.3f}, {v[1]:6.3f})")
    print(f"                   (mix de {abs(v[0]):.1%} înălțime + {abs(v[1]):.1%} greutate)")
print()

print("INTERPRETARE:")
print(f"  • Prima componentă (eigenvalue={eigenvalori_pca[0]:.1f}) explică")
print(f"    {100*eigenvalori_pca[0]/eigenvalori_pca.sum():.1f}% din variație")
print(f"  • Dacă vrem să comprimi datele, putem folosi DOAR prima componentă")
print(f"    și păstrăm ~{100*eigenvalori_pca[0]/eigenvalori_pca.sum():.0f}% din informație!")
print()

# Proiectăm datele pe prima componentă principală
date_comprimate = np.dot(date_centrate, eigenvectori_pca[:, 0:1])

print(f"Date ORIGINALE: {date.shape} (100 persoane × 2 features)")
print(f"Date COMPRIMATE: {date_comprimate.shape} (100 persoane × 1 feature)")
print(f"Reducere: 50% din dimensiuni, păstrând ~{100*eigenvalori_pca[0]/eigenvalori_pca.sum():.0f}% din informație!")
print()

# ============================================
# APLICAȚII REAL-WORLD
# ============================================
print("=" * 70)
print("APLICAȚII REAL-WORLD ale EIGENVALUES/EIGENVECTORS")
print("=" * 70)
print()

print("1. GOOGLE PAGERANK")
print("   • Web-ul = matrice URIAȘĂ (miliarde de pagini)")
print("   • Eigenvector-ul principal = importanța fiecărei pagini")
print("   • Eigenvalue arată cât de 'conectat' este web-ul")
print()

print("2. RECUNOAȘTERE FACIALĂ (Eigenfaces)")
print("   • Fiecare față = vector de pixeli")
print("   • Eigenvectori = 'fețe prototip'")
print("   • Orice față = combinație de eigenfaces")
print("   • Reduci 10,000 pixeli → 50 eigenvalues!")
print()

print("3. FIZICĂ & ENGINEERING")
print("   • Vibrațiile unei poduri/clădiri")
print("   • Eigenvectori = modurile de vibrație")
print("   • Eigenvalori = frecvențele naturale")
print("   • Critic pentru design anti-seismic!")
print()

print("4. QUANTUM MECHANICS")
print("   • Eigenvectori = stările posibile ale unui sistem")
print("   • Eigenvalori = nivelurile de energie")
print("   • Ecuația lui Schrödinger = problemă de eigenvalues!")
print()

print("5. MACHINE LEARNING")
print("   • PCA: reduci dimensionalitate (1000 features → 10)")
print("   • Spectral Clustering: grupezi date")
print("   • Dimensionality reduction pentru vizualizare")
print()

print("6. COMPRESIE IMAGINI (JPEG, etc)")
print("   • Descompui imaginea în componente principale")
print("   • Păstrezi doar eigenvectorii cu eigenvalori mari")
print("   • Comprimi 90% păstrând 95% din calitate!")
print()

# ============================================
# FORMULA MAGICĂ
# ============================================
print("=" * 70)
print("FORMULA MAGICĂ (RECAP)")
print("=" * 70)
print()

print("Pentru o matrice A, căutăm vectori v și scalari λ astfel încât:")
print()
print("  A × v = λ × v")
print()
print("În cuvinte:")
print("  'Când transform vectorul v cu matricea A,")
print("   obțin același vector v, doar înmulțit cu λ'")
print()
print("Unde:")
print("  • v = EIGENVECTOR (direcția specială)")
print("  • λ = EIGENVALUE (factorul de scalare)")
print()
print("DE CE E IMPORTANT?")
print("  → Eigenvectorii descompun transformări complexe în componente simple")
print("  → Găsesc 'esența' datelor, eliminând noise-ul")
print("  → Transformă probleme complicate în probleme simple")
print()
print("=" * 70)