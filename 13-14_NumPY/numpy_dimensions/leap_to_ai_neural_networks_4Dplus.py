import numpy as np

# ============================================================
# 🤖 Rețele Neuronale cu NumPy — 4D Thinking
# ============================================================
# NumPy este baza matematică pentru toate bibliotecile AI moderne:
#   • TensorFlow
#   • PyTorch
#   • scikit-learn
#
# Toate acestea se bazează pe operații fundamentale de algebra liniară:
#   - înmulțiri de matrici (dot product)
#   - funcții de activare (tanh, relu, sigmoid)
#   - transformări multi-dimensionale (tensori)
#
# ------------------------------------------------------------
# 🧠 Model simplificat de rețea neuronală:
#   Input Layer: 784 neuroni  (imagine 28×28 pixeli)
#   Hidden Layer: 128 neuroni (caracteristici intermediare)
#   Output Layer: 10 neuroni  (probabilități pentru cifrele 0–9)
# ============================================================


# ------------------------------------------------------------
# 1️⃣ Definim o rețea neuronală simplă
# ------------------------------------------------------------
class SimpleNeuralNetwork:
    def __init__(self):
        # Matricile de greutăți (Weights)
        # np.random.randn() → distribuție normală (mean=0, std=1)
        # *0.1 → scalare mică pentru stabilitate numerică
        self.W1 = np.random.randn(784, 128) * 0.1  # input → hidden
        self.W2 = np.random.randn(128, 10) * 0.1   # hidden → output

    def forward(self, X):
        # --------------------------------------------------------
        # Propagarea înainte (Forward Pass)
        # --------------------------------------------------------
        # 1. Intrarea (X) are forma (batch_size, 784)
        #    → fiecare rând = o imagine aplatizată (28×28)
        #
        # 2. np.dot(X, W1) → înmulțire matriceală
        #    (batch_size × 784) · (784 × 128) = (batch_size × 128)
        self.z1 = np.dot(X, self.W1)

        # 3. Funcție de activare tanh (non-liniaritate)
        #    Se aplică element cu element
        self.a1 = np.tanh(self.z1)

        # 4. Al doilea strat (hidden → output)
        #    (batch_size × 128) · (128 × 10) = (batch_size × 10)
        self.z2 = np.dot(self.a1, self.W2)

        # 5. np.softmax() → transformă scorurile brute în probabilități
        return self.softmax(self.z2)

    def softmax(self, z):
        # --------------------------------------------------------
        # Softmax = e^(zi) / Σ(e^(zj))
        # Normalizează fiecare rând la sumă = 1 (probabilități)
        # --------------------------------------------------------
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))  # stabil numeric
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities


# ------------------------------------------------------------
# 2️⃣ Testăm rețeaua cu un "batch" de imagini
# ------------------------------------------------------------
# Să presupunem că avem 5 imagini de intrare (batch_size = 5)
# Fiecare imagine = vector de 784 valori normalizate (0–1)
X = np.random.rand(5, 784)

# Creăm rețeaua și rulăm propagarea
nn = SimpleNeuralNetwork()
output = nn.forward(X)

print("🔢 Forma ieșirii (batch_size × num_clase):", output.shape)
print("📊 Primele probabilități:\n", output[0])

# Exemplu de rezultat:
# 🔢 Forma ieșirii (batch_size × num_clase): (5, 10)
# 📊 Primele probabilități:
# [0.105 0.096 0.098 0.097 0.101 0.098 0.097 0.103 0.099 0.106]
