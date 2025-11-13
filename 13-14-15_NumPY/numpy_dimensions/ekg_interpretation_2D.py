import numpy as np
import time

# ============================================================
# ❤️ Analiza unui semnal EKG cu FFT (Fast Fourier Transform)
# ============================================================
# Scop:
#   - Să simulăm un semnal cardiac (ECG simplificat)
#   - Să adăugăm zgomot (noise)
#   - Să descoperim automat frecvența dominantă → pulsul inimii
#
# FFT = "Fast Fourier Transform"
#      → transformă un semnal din domeniul timpului
#         (cum variază în timp)
#        în domeniul frecvenței
#         (ce frecvențe compun acel semnal)
#
# ============================================================


# ------------------------------------------------------------
# 1️⃣ Definim semnalul în timp
# ------------------------------------------------------------
# np.linspace(start, stop, num_points)
# → creează 1000 valori între 0 și 10 secunde
time = np.linspace(0, 10, 1000)

# Ritmul cardiac mediu în bătăi pe minut (BPM)
heart_rate = 75

# ------------------------------------------------------------
# 2️⃣ Creăm semnalul cardiac
# ------------------------------------------------------------
# Fiecare bătaie = o undă sinusoidală
# Formula: sin(2π * f * t)
#   unde f = frecvența în Hz (bătăi/secundă)
signal = np.sin(2 * np.pi * (heart_rate / 60) * time)

# Adăugăm zgomot (rușit – noise) pentru realism
# np.random.normal(mean, std_dev, size)
#   → distribuție normală (clopot)
noise = np.random.normal(0, 0.1, len(time))

# Semnalul final EKG (cu zgomot)
ekg = signal + noise


# ------------------------------------------------------------
# 3️⃣ Aplicăm Transformata Fourier (FFT)
# ------------------------------------------------------------
# np.fft.fft() transformă semnalul 1D din domeniul timpului
# în domeniul frecvenței → amplitudine pentru fiecare frecvență componentă
fft_result = np.fft.fft(ekg)

# np.fft.fftfreq(n, d)
#   → creează lista frecvențelor corespunzătoare fiecărei componente FFT
#   n = numărul de eșantioane
#   d = intervalul de timp între eșantioane (1 / frecvența de eșantionare)
frequencies = np.fft.fftfreq(len(ekg), 1 / 100)


# ------------------------------------------------------------
# 4️⃣ Identificăm frecvența dominantă
# ------------------------------------------------------------
# np.abs() → valoarea absolută (mărimea complexă a fiecărei componente)
# FFT returnează numere complexe (parte reală + imaginară)
#   → modulul (abs) arată "puterea" fiecărei frecvențe
# np.argmax() → indexul elementului cu valoarea maximă
#   → poziția frecvenței dominante
dominant_freq = frequencies[np.argmax(np.abs(fft_result))]

# Convertim în BPM (bătăi pe minut)
bpm = dominant_freq * 60

print(f"❤️ Frecvență cardiacă estimată: {bpm:.2f} BPM")


# ============================================================
# 🧠 Ce am folosit din NumPy:
# ------------------------------------------------------------
# np.linspace() → generare de intervale
# np.random.normal() → zgomot cu distribuție normală
# np.sin() → undă sinusoidală
# np.fft.fft() → transformare Fourier (timp → frecvență)
# np.fft.fftfreq() → vectorul de frecvențe
# np.abs() → modulul numerelor complexe
# np.argmax() → poziția celei mai mari valori
# ============================================================

# Exemplu de rezultat:
# ❤️ Frecvență cardiacă estimată: 74.99 BPM
