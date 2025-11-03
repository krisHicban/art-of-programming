"""
TRANSFORMĂRI GEOMETRICE - Cum se mișcă obiectele în jocuri și grafică

Imaginează-te că programezi un joc video și trebuie să rotești un personaj,
sau să deplasezi o dronă în spațiu, sau să animezi un robot.

Cum face computerul asta? Cu MATRICI de transformare!

Exemple real-world:
  - Jocuri: Unity, Unreal Engine, Godot
  - Roboți: mișcarea brațului robotic
  - Drone: navigare și orientare în spațiu
  - CGI în filme: animație 3D
  - AR/VR: tracking și poziționare
"""

import numpy as np
import math

print("=" * 70)
print("TRANSFORMĂRI GEOMETRICE - Matematica din spatele jocurilor")
print("=" * 70)
print()

# ============================================
# SCENARIUL: Un personaj în joc
# ============================================
print("SCENARIUL: Un personaj de joc (vârful de săgeată ↑)")
print("-" * 70)
print()

print("Personajul nostru este definit de 5 puncte în spațiu 2D:")
print("  - Vârful săgeții (în sus)")
print("  - 2 aripi laterale")
print("  - 2 puncte la bază")
print()

# Definim forma personajului (săgeată orientată în sus)
# Coordonate: (x, y)
#       (0, 2)        <- vârful
#      /      \
#   (-1, 0)  (1, 0)   <- aripi
#      \      /
#   (-0.5,-1) (0.5,-1) <- bază

personaj = np.array([
    [ 0.0,  2.0],  # vârf
    [-1.0,  0.0],  # aripa stângă
    [ 1.0,  0.0],  # aripa dreapta
    [-0.5, -1.0],  # bază stânga
    [ 0.5, -1.0]   # bază dreapta
])

print("Coordonatele personajului (poziția INIȚIALĂ):")
for i, (x, y) in enumerate(personaj):
    labels = ["vârf", "aripa stânga", "aripa dreapta", "bază stânga", "bază dreapta"]
    print(f"  {labels[i]:15s}: ({x:5.1f}, {y:5.1f})")
print()

def afiseaza_ascii(puncte, titlu="Vizualizare"):
    """Afișează o vizualizare ASCII simplă a punctelor."""
    print(f"{titlu}:")

    # Creăm un grid 15x15 centrat pe origine
    grid_size = 15
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    center = grid_size // 2

    # Desenăm axele
    for i in range(grid_size):
        grid[center][i] = '·'  # axa X
        grid[i][center] = '·'  # axa Y
    grid[center][center] = '+'  # origine

    # Plasăm punctele
    for i, (x, y) in enumerate(puncte):
        # Scalăm și convertim la poziție în grid
        grid_x = int(round(x * 2)) + center
        grid_y = center - int(round(y * 2))  # Y-ul e inversat în display

        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            grid[grid_y][grid_x] = str(i)  # marcăm cu indexul punctului

    # Afișăm grid-ul
    for row in grid:
        print("  " + "".join(row))
    print()

afiseaza_ascii(personaj, "Personajul ÎNAINTE de transformare")

# ============================================
# TRANSFORMARE 1: ROTAȚIE
# ============================================
print("=" * 70)
print("TRANSFORMARE 1: ROTAȚIE (playerul se întoarce)")
print("=" * 70)
print()

print("Să zicem că playerul apasă tasta 'A' → personajul se rotește 45°")
print()

# Unghiul de rotație
unghi_grade = 45
unghi_radiani = math.radians(unghi_grade)

print(f"Unghi: {unghi_grade}° = {unghi_radiani:.4f} radiani")
print()

# MATRICEA DE ROTAȚIE
# Aceasta este formula standard pentru rotație în 2D:
# R = [cos(θ)  -sin(θ)]
#     [sin(θ)   cos(θ)]

R = np.array([
    [math.cos(unghi_radiani), -math.sin(unghi_radiani)],
    [math.sin(unghi_radiani),  math.cos(unghi_radiani)]
])

print("MATRICEA DE ROTAȚIE R:")
print(R)
print()

print("Cum funcționează?")
print("  Pentru fiecare punct (x, y) din personaj:")
print("  - Înmulțim punctul cu matricea R")
print("  - Obținem noul punct (x', y') rotit")
print()
print("  [x']   [cos(θ)  -sin(θ)]   [x]")
print("  [y'] = [sin(θ)   cos(θ)] × [y]")
print()

# Aplicăm rotația
personaj_rotit = np.dot(personaj, R.T)  # .T = transpose

print("Coordonatele DUPĂ rotație:")
for i, (x, y) in enumerate(personaj_rotit):
    labels = ["vârf", "aripa stânga", "aripa dreapta", "bază stânga", "bază dreapta"]
    x_orig, y_orig = personaj[i]
    print(f"  {labels[i]:15s}: ({x_orig:5.1f}, {y_orig:5.1f}) → ({x:5.1f}, {y:5.1f})")
print()

afiseaza_ascii(personaj_rotit, "Personajul DUPĂ rotație cu 45°")

# ============================================
# TRANSFORMARE 2: TRANSLAȚIE (Mutare)
# ============================================
print("=" * 70)
print("TRANSFORMARE 2: TRANSLAȚIE (playerul se mișcă)")
print("=" * 70)
print()

print("Playerul apasă 'W' → personajul merge înainte (în sus)")
print()

# Vector de deplasare
deplasare = np.array([2.0, 3.0])  # +2 pe X, +3 pe Y

print(f"Deplasare: ({deplasare[0]:.1f}, {deplasare[1]:.1f})")
print("  (adică 2 unități la dreapta, 3 unități în sus)")
print()

# Aplicăm translația (e mai simplu - doar adunăm!)
personaj_mutat = personaj_rotit + deplasare

print("Coordonatele DUPĂ deplasare:")
for i, (x, y) in enumerate(personaj_mutat):
    x_prev, y_prev = personaj_rotit[i]
    labels = ["vârf", "aripa stânga", "aripa dreapta", "bază stânga", "bază dreapta"]
    print(f"  {labels[i]:15s}: ({x_prev:5.1f}, {y_prev:5.1f}) → ({x:5.1f}, {y:5.1f})")
print()

afiseaza_ascii(personaj_mutat, "Personajul DUPĂ rotație + deplasare")

# ============================================
# TRANSFORMARE 3: SCALARE (Mărire/Micșorare)
# ============================================
print("=" * 70)
print("TRANSFORMARE 3: SCALARE (power-up! personajul crește)")
print("=" * 70)
print()

print("Playerul ia un power-up → personajul devine de 1.5x mai mare")
print()

# Matricea de scalare
# S = [sx  0 ]
#     [0  sy]
# unde sx, sy sunt factorii de scalare

factor_scalare = 1.5
S = np.array([
    [factor_scalare, 0],
    [0, factor_scalare]
])

print("MATRICEA DE SCALARE S:")
print(S)
print()

# Aplicăm scalarea pe personajul original (pentru vizibilitate)
personaj_marit = np.dot(personaj, S.T)

print("Coordonatele DUPĂ scalare cu 1.5x:")
for i, (x, y) in enumerate(personaj_marit):
    x_orig, y_orig = personaj[i]
    labels = ["vârf", "aripa stânga", "aripa dreapta", "bază stânga", "bază dreapta"]
    print(f"  {labels[i]:15s}: ({x_orig:5.1f}, {y_orig:5.1f}) → ({x:5.1f}, {y:5.1f})")
print()

# ============================================
# COMBINAREA TRANSFORMĂRILOR
# ============================================
print("=" * 70)
print("COMBINAREA TRANSFORMĂRILOR (magia game engines)")
print("=" * 70)
print()

print("În jocuri reale, transformările se combină în FIECARE FRAME:")
print("  1. SCALARE (dimensiunea obiectului)")
print("  2. ROTAȚIE (orientarea obiectului)")
print("  3. TRANSLAȚIE (poziția în lume)")
print()

print("Exemplu: personaj de 0.8x, rotit cu 90°, la poziția (5, 5)")
print()

# Construim transformările
scale = 0.8
S_combined = np.array([[scale, 0], [0, scale]])

angle = math.radians(90)
R_combined = np.array([
    [math.cos(angle), -math.sin(angle)],
    [math.sin(angle),  math.cos(angle)]
])

translation = np.array([5.0, 5.0])

# Aplicăm în ordine: scalare → rotație → translație
step1 = np.dot(personaj, S_combined.T)
step2 = np.dot(step1, R_combined.T)
step3 = step2 + translation

print("Transformare finală (toate combinate):")
for i, (x, y) in enumerate(step3):
    labels = ["vârf", "aripa stânga", "aripa dreapta", "bază stânga", "bază dreapta"]
    print(f"  {labels[i]:15s}: ({x:5.1f}, {y:5.1f})")
print()

# ============================================
# EXEMPLU REAL-WORLD: Brațul robotic
# ============================================
print("=" * 70)
print("BONUS: BRAȚUL ROBOTIC (aplicație industrială)")
print("=" * 70)
print()

print("Un braț robotic trebuie să ajungă la un obiect.")
print("Brațul are 2 segmente care se pot roti:")
print()

# Segment 1: de la origine la articulație
lungime_segment1 = 3.0
unghi1 = math.radians(30)  # primul segment la 30°

# Segment 2: de la articulație la vârf
lungime_segment2 = 2.0
unghi2 = math.radians(45)  # al doilea segment la +45° relativ

# Calculăm poziția articulației
articulatie = np.array([
    lungime_segment1 * math.cos(unghi1),
    lungime_segment1 * math.sin(unghi1)
])

# Calculăm poziția vârfului (relativ la articulație)
varf_relativ = np.array([
    lungime_segment2 * math.cos(unghi1 + unghi2),
    lungime_segment2 * math.sin(unghi1 + unghi2)
])

varf_absolut = articulatie + varf_relativ

print(f"Segment 1 (lungime {lungime_segment1}, unghi {math.degrees(unghi1):.1f}°):")
print(f"  Articulație la: ({articulatie[0]:.2f}, {articulatie[1]:.2f})")
print()
print(f"Segment 2 (lungime {lungime_segment2}, unghi {math.degrees(unghi2):.1f}° relativ):")
print(f"  Vârf la: ({varf_absolut[0]:.2f}, {varf_absolut[1]:.2f})")
print()

# ============================================
# CONCLUZIE
# ============================================
print("=" * 70)
print("DE CE SUNT MATRICILE ATÂT DE IMPORTANTE?")
print("=" * 70)
print()
print("1. GAME ENGINES (Unity, Unreal, Godot):")
print("   - Fiecare obiect are o 'transform matrix'")
print("   - 60+ transformări pe secundă (60 FPS)")
print("   - Mii de obiecte simultan")
print()
print("2. ROBOTICĂ:")
print("   - Calculează poziția exactă a brațului/camerei")
print("   - Kinematics: de la unghiuri la poziții în spațiu")
print()
print("3. GRAFICĂ 3D & FILME:")
print("   - Pixar, Disney, ILM - toate folosesc transformări")
print("   - Animație: interpolează între matrici")
print()
print("4. AR/VR (Realitate augmentată/virtuală):")
print("   - Tracking în timp real")
print("   - Poziționare în spațiu 3D")
print()
print("5. DRONES & AUTO-PILOTING:")
print("   - Calculează orientarea în spațiu")
print("   - Rotații în 3D (quaternions → matrici de rotație)")
print()
print("Toate acestea folosesc ACEEAȘI matematică pe care tocmai")
print("am demonstrat-o! Doar că în 3D și cu milioane de calcule/sec.")
print("=" * 70)