"""
Matrix Operations and Neural Networks
=====================================
Understanding the fundamental mathematics behind AI
"""

import numpy as np

print("=" * 60)
print("OPERAȚII FUNDAMENTALE CU MATRICI")
print("Fundamental Matrix Operations")
print("=" * 60)

# Definirea matricilor / Matrix Definition
A = np.array([[2, 1],
              [1, 3]])

B = np.array([[4, 0],
              [1, 2]])

print("\nMatricea A:")
print(A)
print("\nMatricea B:")
print(B)

# Înmulțire matriceală / Matrix Multiplication
rezultat = np.dot(A, B)  # sau A @ B în Python 3.5+

print("\n" + "=" * 60)
print("REZULTAT (A × B):")
print(rezultat)
print("=" * 60)

# =====================================================
# CUM FUNCȚIONEAZĂ ÎNMULȚIREA MATRICILOR
# How Matrix Multiplication Works
# =====================================================

print("\n" + "=" * 60)
print("CUM FUNCȚIONEAZĂ ÎNMULȚIREA MATRICILOR?")
print("=" * 60)

print("""
Pentru a înmulți două matrici A și B:
1. Numărul de COLOANE din A trebuie să fie egal cu numărul de RÂNDURI din B
2. Rezultatul va avea dimensiunea: (rânduri din A) × (coloane din B)

În cazul nostru:
- A are dimensiunea 2×2 (2 rânduri, 2 coloane)
- B are dimensiunea 2×2 (2 rânduri, 2 coloane)
- Rezultatul va fi 2×2
""")

print("CALCULUL PAS CU PAS:")
print("-" * 40)

# Calculăm manual fiecare element
print("\nPentru poziția [0,0] (rândul 1, coloana 1):")
print(f"  Luăm rândul 1 din A: {A[0]}")
print(f"  Luăm coloana 1 din B: {B[:, 0]}")
print(f"  Înmulțim element cu element și adunăm:")
print(f"  {A[0, 0]}×{B[0, 0]} + {A[0, 1]}×{B[1, 0]} = {A[0, 0] * B[0, 0]} + {A[0, 1] * B[1, 0]} = {rezultat[0, 0]}")

print("\nPentru poziția [0,1] (rândul 1, coloana 2):")
print(f"  Luăm rândul 1 din A: {A[0]}")
print(f"  Luăm coloana 2 din B: {B[:, 1]}")
print(f"  {A[0, 0]}×{B[0, 1]} + {A[0, 1]}×{B[1, 1]} = {A[0, 0] * B[0, 1]} + {A[0, 1] * B[1, 1]} = {rezultat[0, 1]}")

print("\nPentru poziția [1,0] (rândul 2, coloana 1):")
print(f"  Luăm rândul 2 din A: {A[1]}")
print(f"  Luăm coloana 1 din B: {B[:, 0]}")
print(f"  {A[1, 0]}×{B[0, 0]} + {A[1, 1]}×{B[1, 0]} = {A[1, 0] * B[0, 0]} + {A[1, 1] * B[1, 0]} = {rezultat[1, 0]}")

print("\nPentru poziția [1,1] (rândul 2, coloana 2):")
print(f"  Luăm rândul 2 din A: {A[1]}")
print(f"  Luăm coloana 2 din B: {B[:, 1]}")
print(f"  {A[1, 1]}×{B[0, 1]} + {A[1, 1]}×{B[1, 1]} = {A[1, 0] * B[0, 1]} + {A[1, 1] * B[1, 1]} = {rezultat[1, 1]}")

# =====================================================
# DE CE ESTE IMPORTANT ÎN REȚELELE NEURONALE?
# Why Is This Important in Neural Networks?
# =====================================================

print("\n" + "=" * 60)
print("DE CE ESTE CRUCIAL ÎN INTELIGENȚA ARTIFICIALĂ?")
print("=" * 60)

print("""
1. TRANSFORMAREA DATELOR (Data Transformation)
   ---------------------------------------------
   • Fiecare strat dintr-o rețea neurală aplică o transformare liniară
   • Input × Weights = Output transformat
   • Matricile de greutăți (weights) învață pattern-uri din date

2. PROPAGAREA ÎNAINTE (Forward Propagation)
   -----------------------------------------
   • Input Layer: x (vector de caracteristici)
   • Hidden Layer: h = W₁×x + b₁ (transformare liniară + bias)
   • Output Layer: y = W₂×h + b₂
   • Fiecare '×' este o înmulțire matriceală!

3. PROCESARE PARALELĂ (Parallel Processing)
   -----------------------------------------
   • În loc să procesăm un exemplu pe rând, procesăm batch-uri
   • Batch de 100 imagini × Matrice de greutăți = 100 rezultate simultan
   • GPU-urile excelează la înmulțiri matriceale paralele

4. COMPRESIA INFORMAȚIEI (Information Compression)
   ------------------------------------------------
   • O imagine 28×28 (784 pixeli) → vector 10 elemente (cifre 0-9)
   • Matricea de greutăți codifică ce caracteristici sunt importante
   • Reduce dimensionalitatea păstrând informația esențială
""")

# =====================================================
# EXEMPLU PRACTIC: MINI REȚEA NEURALĂ
# Practical Example: Mini Neural Network
# =====================================================

print("\n" + "=" * 60)
print("EXEMPLU PRACTIC: CLASIFICARE SIMPLĂ")
print("=" * 60)

# Date de intrare (2 caracteristici pentru 3 exemple)
X = np.array([[0.5, 0.8],  # Exemplul 1
              [0.2, 0.9],  # Exemplul 2
              [0.7, 0.3]])  # Exemplul 3

# Greutăți învățate de rețea
W = np.array([[0.3, -0.5],  # Neuroni pentru caracteristica 1
              [0.8, 0.2]])  # Neuroni pentru caracteristica 2

print("\nDate de intrare (3 exemple, 2 caracteristici fiecare):")
print(X)

print("\nMatrice de greutăți (transformare învățată):")
print(W)

# Calculăm output-ul
output = np.dot(X, W.T)  # Transpunem W pentru dimensiuni corecte

print("\nOutput după transformare:")
print(output)

print("""
INTERPRETARE:
• Fiecare rând din output reprezintă răspunsul pentru un exemplu
• Fiecare coloană reprezintă activarea unui neuron
• Rețeaua a învățat să transforme caracteristicile în reprezentări utile
""")

# =====================================================
# PROPRIETĂȚI IMPORTANTE
# Important Properties
# =====================================================

print("\n" + "=" * 60)
print("PROPRIETĂȚI MATEMATICE CHEIE")
print("=" * 60)

print("""
1. NON-COMUTATIVITATE: A×B ≠ B×A (în general)
   - Ordinea contează! Spre deosebire de înmulțirea numerelor

2. ASOCIATIVITATE: (A×B)×C = A×(B×C)
   - Putem grupa înmulțirile cum vrem

3. DISTRIBUTIVITATE: A×(B+C) = A×B + A×C
   - Putem distribui înmulțirea peste adunare

4. DIMENSIUNI: (m×n) × (n×p) = (m×p)
   - Coloanele primei matrici = Rândurile celei de-a doua
   - Rezultatul are m rânduri și p coloane
""")

# Demonstrație non-comutativitate
print("\nDEMONSTRAȚIE NON-COMUTATIVITATE:")
print("-" * 40)
AB = np.dot(A, B)
BA = np.dot(B, A)

print(f"\nA × B =")
print(AB)
print(f"\nB × A =")
print(BA)
print(f"\nSunt egale? {np.array_equal(AB, BA)}")

# =====================================================
# COMPLEXITATE COMPUTAȚIONALĂ
# Computational Complexity
# =====================================================

print("\n" + "=" * 60)
print("COMPLEXITATE ȘI PERFORMANȚĂ")
print("=" * 60)

print("""
Pentru matrici n×n:
• Operații necesare: O(n³)
• Pentru n=1000: ~1 miliard de înmulțiri!

De aceea:
• GPU-urile sunt esențiale pentru AI (mii de nuclee paralele)
• Librării optimizate (BLAS, cuBLAS) sunt critice
• Tehnici speciale pentru matrici sparse (multe zerouri)
""")

print("\n" + "=" * 60)
print("CONCLUZIE")
print("=" * 60)
print("""
Înmulțirea matricilor este FUNDAMENTUL matematicii din spatele AI:
• Transformă și combină informația
• Permite procesare paralelă masivă
• Codifică cunoștințele învățate în greutăți
• Face posibilă propagarea informației prin straturi

Fără înmulțirea matricilor eficientă, deep learning-ul modern
nu ar fi posibil!
""")