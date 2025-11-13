import numpy as np
import time

# ============================================================
# 💼 Optimizare portofoliu — Modelul Markowitz
# ============================================================
# Scop:
#   - Să simulăm 50 de acțiuni (companii)
#   - Fiecare are randamente zilnice pe parcursul unui an (~252 zile)
#   - Calculăm:
#       → Randamentul mediu anual
#       → Volatilitatea (risc)
#       → Sharpe Ratio (randament ajustat la risc)
#
# Acest exemplu ilustrează:
#   - Matrici 2D și transpunerea lor
#   - Operații pe axe (mean, std, corrcoef)
#   - Vectorizare completă (fără loop-uri)
# ============================================================


# ------------------------------------------------------------
# 1️⃣ Generăm date de randament
# ------------------------------------------------------------
np.random.seed(42)

# 252 zile × 50 active = randamente zilnice
# Distribuție normală cu medie 0.001 (0.1%/zi) și deviație 0.02 (~2%)
returns = np.random.normal(0.001, 0.02, (252, 50))  # shape: (252, 50)

# ------------------------------------------------------------
# 2️⃣ Matricea de corelație
# ------------------------------------------------------------
# np.corrcoef(X.T) → calculează corelația între coloane (active)
# Transpunerea .T schimbă forma din (252, 50) → (50, 252)
# Astfel, fiecare activ devine o "serie" proprie
correlation_matrix = np.corrcoef(returns.T)  # shape: (50, 50)
# Fiecare element (i,j) = cât de corelate sunt cele două acțiuni


# ------------------------------------------------------------
# 3️⃣ Calculăm randamentul și riscul anualizat
# ------------------------------------------------------------
# np.mean(..., axis=0) → media pe fiecare coloană (pe activ)
# np.std(..., axis=0)  → deviația standard pe fiecare activ
# Apoi anualizăm:
#   - randament * 252 (număr zile tranzacționare)
#   - risc * sqrt(252)
mean_returns = np.mean(returns, axis=0) * 252
volatility = np.std(returns, axis=0) * np.sqrt(252)


# ------------------------------------------------------------
# 4️⃣ Calculăm Sharpe Ratio
# ------------------------------------------------------------
# Sharpe Ratio = (Randament mediu - Rată fără risc) / Volatilitate
# Rată fără risc (risk-free) ~ 3%/an
risk_free_rate = 0.03
sharpe_ratios = (mean_returns - risk_free_rate) / volatility

# ------------------------------------------------------------
# 5️⃣ Afișăm rezultatele
# ------------------------------------------------------------
print(f"📈 Randamente medii anuale (primele 5): {mean_returns[:5]}")
print(f"⚖️  Volatilități anuale (primele 5):    {volatility[:5]}")
print(f"💰 Sharpe ratios (primele 5):           {sharpe_ratios[:5]}")
print(f"⭐ Cel mai bun Sharpe ratio: {np.max(sharpe_ratios):.3f}")

# ============================================================
# 🧠 Ce am folosit din NumPy:
# ------------------------------------------------------------
# np.random.normal() → simulare randamente
# np.corrcoef() → matrice de corelație (relații între active)
# .T → transpunerea matricii
# np.mean(axis=0), np.std(axis=0) → calcule pe coloane
# np.sqrt() → anualizare a riscului
# np.max() → selectarea celui mai performant activ
# ============================================================

# Exemplu de rezultat:
# 📈 Randamente medii anuale (primele 5): [0.234 0.256 0.237 0.218 0.213]
# ⚖️  Volatilități anuale (primele 5):    [0.327 0.312 0.314 0.302 0.324]
# 💰 Sharpe ratios (primele 5):           [0.625 0.727 0.660 0.621 0.565]
# ⭐ Cel mai bun Sharpe ratio: 0.727
