"""
FORWARD PASS - Cum "gândește" un neuron artificial

Imaginează-te că creăm un "anger neuron" (neuron de furie) simplu -
o versiune extrem de simplificată a modului în care creierul tău procesează emoții.

Realitatea: creierul uman are ~86 miliarde neuroni cu trilioane de conexiuni.
Aici: demonstrăm principiul cu 3 neuroni simpli.
"""

import numpy as np

# ============================================
# PASUL 1: INPUT (Semnalele de intrare)
# ============================================
# Să zicem că monitorizăm starea ta emoțională:
# - hunger (foame): 0.8 (foarte înfometat, pe o scală 0-1)
# - fatigue (oboseală): 0.9 (extrem de obosit)
# - provocation (provocare): 0.6 (cineva ți-a zis ceva neplăcut)

inputs = np.array([0.8, 0.9, 0.6])
input_labels = ["hunger", "fatigue", "provocation"]

print("=" * 60)
print("STAREA TA CURENTĂ (inputs):")
print("=" * 60)
for label, value in zip(input_labels, inputs):
    print(f"  {label:15s}: {value:.1f}")
print()

# ============================================
# PASUL 2: WEIGHTS (Modelul tău mental/emotional unic)
# ============================================
# Fiecare persoană reactionează diferit la aceiași stimuli!
# Weights = "pattern-urile mentale și emocionale" personale
#
# Avem 3 neuroni în layer-ul nostru, fiecare învață
# o combinație diferită de factori:

weights = np.array([
    # Neuron 1: reacționează mai mult la FOAME + PROVOCARE
    [0.9, 0.2, 0.8],  # [hunger, fatigue, provocation]

    # Neuron 2: reacționează la OBOSEALĂ + PROVOCARE
    [0.1, 0.9, 0.7],

    # Neuron 3: balansat, ia în considerare tot
    [0.5, 0.5, 0.5]
])

print("=" * 60)
print("PATTERN-URILE TALE MENTALE (weights):")
print("=" * 60)
print("Fiecare neuron 'învățat' să răspundă diferit:\n")
neuron_names = ["Hungry neuron", "Tired-angry neuron", "Balanced neuron"]
for i, (name, w) in enumerate(zip(neuron_names, weights)):
    print(f"  {name}:")
    for label, weight in zip(input_labels, w):
        print(f"    - {label:15s}: {weight:.1f}")
    print()

# ============================================
# PASUL 3: MATRIX MULTIPLICATION (Procesare)
# ============================================
# Ce înseamnă np.dot(inputs, weights.T)?
#
# Pentru fiecare neuron, calculăm cât de "activat" este:
# - înmulțim fiecare input cu weight-ul corespunzător
# - adunăm toate produsele
#
# Exemplu pentru neuron 1:
# z1 = (0.8 × 0.9) + (0.9 × 0.2) + (0.6 × 0.8)
#    = 0.72 + 0.18 + 0.48 = 1.38

z = np.dot(inputs, weights.T)

print("=" * 60)
print("CALCULUL SEMNALULUI (matrix multiplication):")
print("=" * 60)
print("Pentru fiecare neuron, calculăm:")
print("signal = (hunger × weight₁) + (fatigue × weight₂) + (provocation × weight₃)\n")
for i, (name, signal) in enumerate(zip(neuron_names, z)):
    # Calculăm manual pentru claritate
    calculation = " + ".join([
        f"({inputs[j]:.1f} × {weights[i][j]:.1f})"
        for j in range(len(inputs))
    ])
    print(f"  {name}:")
    print(f"    {calculation} = {signal:.2f}")
    print()

# ============================================
# PASUL 4: ACTIVATION FUNCTION (Decizie binară)
# ============================================
# Neuronii reali sunt BINARI: fie "they Fire" (1), fie "they Don't Fire" (0)
# Dar învățarea e mai eficientă cu valori continue între -1 și 1
#
# tanh (tangenta hiperbolică):
# - Input mare pozitiv  → Output ~1  (FOARTE ACTIVAT)
# - Input ~0            → Output ~0  (NEUTRU)
# - Input mare negativ  → Output ~-1 (INHIBAT)
#
# E ca un "switch" moale în loc de unul dur.

activation = np.tanh(z)

print("=" * 60)
print("ACTIVAREA NEURONILOR (activation function):")
print("=" * 60)
print("tanh() transformă semnalul în răspuns neuronal:")
print("  -1.0 = complet inhibat")
print("   0.0 = neutru")
print("  +1.0 = complet activat\n")

for name, signal, active in zip(neuron_names, z, activation):
    print(f"  {name}:")
    print(f"    Signal: {signal:6.2f} → Activation: {active:5.2f}")
    print()

# ============================================
# PASUL 5: OUTPUT (Interpretare finală)
# ============================================
# În realitate, aceste valori ar merge în layer-uri ulterioare
# care le-ar transforma în "probabilități de reacție":
# - 80% șansă să răspunzi iritat
# - 15% șansă să ignori
# - 5% șansă să glumești

print("=" * 60)
print("OUTPUT FINAL:")
print("=" * 60)
print(f"Activarea layer-ului 'anger': {activation}")
print("\nÎn modelele reale (ChatGPT, etc.), acest output devine")
print("input pentru următorul layer, și tot așa prin miliarde")
print("de neuroni până la predicția finală!")
print()

# ============================================
# Funcția finală (același cod, dar explicat!)
# ============================================
def forward_pass(x, W):
    """
    Propagarea semnalului prin rețea.

    Args:
        x: inputs (semnale de intrare) - array 1D
        W: weights (pattern-uri învățate) - array 2D

    Returns:
        activation (răspunsul neuronilor) - array 1D
    """
    # 1. Matrix multiplication: calculăm semnalul fiecărui neuron
    z = np.dot(x, W.T)

    # 2. Activation: transformăm în răspuns neuronal (binară-ish)
    a = np.tanh(z)

    return a

print("=" * 60)
print("ChatGPT/Claude fac EXACT asta, dar cu:")
print("  - miliarde de neuroni")
print("  - sute de layer-uri")
print("  - trilioane de weights")
print("  - antrenament pe terabytes de text")
print("=" * 60)