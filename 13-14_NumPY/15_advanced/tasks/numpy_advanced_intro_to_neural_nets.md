# Tema 10: Introducere în Rețele Neuronale cu NumPy

**Temă comună pentru înțelegerea conceptelor de bază ale rețelelor neuronale**

## I. Implementează un neuron simplu cu activare ReLU

Creează o clasă `NeuronSimplu` care să simuleze comportamentul unui neuron cu funcția de activare ReLU.

```python
import numpy as np

class NeuronSimplu:
    def __init__(self, numar_intrari):
        # Inițializează weights și bias aleatoriu
        self.weights = np.random.randn(numar_intrari)
        self.bias = np.random.randn()
    
    def relu(self, x):
        # Implementează funcția ReLU
        # ReLU(x) = max(0, x)
        pass
    
    def forward(self, intrari):
        # Calculează output = ReLU(weights · intrari + bias)
        pass

# Testare
neuron = NeuronSimplu(3)
intrare_test = np.array([1.0, 2.0, -0.5])
output = neuron.forward(intrare_test)
print(f"Output neuron: {output}")
```

**Cerințe:**
- Funcția `relu()` trebuie să returneze 0 pentru valori negative și valoarea însăși pentru valori pozitive
- Metoda `forward()` trebuie să calculeze suma ponderată și să aplice ReLU
- Afișează weights, bias și output-ul final

## II. Creează și aplică funcția sigmoid pe un dataset aleatoriu

Implementează funcția sigmoid și aplică-o pe 100 de valori generate aleatoriu.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    # Implementează funcția sigmoid: f(x) = 1 / (1 + e^(-x))
    pass

# Generează 100 de valori aleatorii între -10 și 10
date_aleatorii = np.random.uniform(-10, 10, 100)

# Aplică sigmoid pe toate valorile
rezultate = # aplică sigmoid aici

# Vizualizare (opțional dar recomandat)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(date_aleatorii, bins=20, alpha=0.7, color='blue')
plt.title('Distribuția valorilor inițiale')

plt.subplot(1, 2, 2)
plt.hist(rezultate, bins=20, alpha=0.7, color='green')
plt.title('Distribuția după aplicarea sigmoid')
plt.show()

# Statistici
print(f"Media inițială: {np.mean(date_aleatorii):.3f}")
print(f"Media după sigmoid: {np.mean(rezultate):.3f}")
print(f"Min/Max inițial: {np.min(date_aleatorii):.3f}/{np.max(date_aleatorii):.3f}")
print(f"Min/Max după sigmoid: {np.min(rezultate):.3f}/{np.max(rezultate):.3f}")
```

**Cerințe:**
- Funcția sigmoid trebuie să mapeze orice valoare reală în intervalul (0, 1)
- Calculează și afișează media, minimul și maximul înainte și după aplicarea sigmoid
- Observă cum sigmoid "comprimă" valorile în intervalul (0, 1)

## III. Simulează o rețea neuronală cu 2 straturi complet vectorizată

Implementează o rețea neuronală simplă cu 2 straturi dense (fully connected) folosind doar NumPy și operații vectorizate.

```python
import numpy as np

class ReteasNeuronala:
    def __init__(self, dim_intrare, dim_ascuns, dim_iesire):
        # Inițializare parametri pentru 2 straturi
        # Stratul 1: dim_intrare -> dim_ascuns
        self.W1 = np.random.randn(dim_intrare, dim_ascuns) * 0.1
        self.b1 = np.zeros((1, dim_ascuns))
        
        # Stratul 2: dim_ascuns -> dim_iesire
        self.W2 = np.random.randn(dim_ascuns, dim_iesire) * 0.1
        self.b2 = np.zeros((1, dim_iesire))
    
    def relu(self, Z):
        # Implementează ReLU vectorizat
        pass
    
    def sigmoid(self, Z):
        # Implementează sigmoid vectorizat
        pass
    
    def forward(self, X):
        """
        Forward pass prin rețea
        X: matrice de intrare (batch_size, dim_intrare)
        """
        # Stratul 1 cu activare ReLU
        Z1 = # calculează X · W1 + b1
        A1 = # aplică ReLU
        
        # Stratul 2 cu activare sigmoid
        Z2 = # calculează A1 · W2 + b2
        A2 = # aplică sigmoid
        
        return A2
    
    def prezice(self, X):
        """Returnează predicții binare (0 sau 1)"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Testare cu date simulate
np.random.seed(42)

# Creează o rețea: 4 intrări -> 5 neuroni ascunși -> 2 ieșiri
retea = ReteasNeuronala(dim_intrare=4, dim_ascuns=5, dim_iesire=2)

# Date de test: 10 exemple cu 4 features fiecare
X_test = np.random.randn(10, 4)

# Obține predicțiile
predictii = retea.forward(X_test)
clase_prezise = retea.prezice(X_test)

print("Forma intrării:", X_test.shape)
print("Forma ieșirii:", predictii.shape)
print("\nPrimele 3 probabilități de ieșire:")
print(predictii[:3])
print("\nPrimele 3 clase prezise:")
print(clase_prezise[:3])

# Verifică vectorizarea
print(f"\nDimensiuni parametri:")
print(f"W1: {retea.W1.shape}, b1: {retea.b1.shape}")
print(f"W2: {retea.W2.shape}, b2: {retea.b2.shape}")
```

**Cerințe:**
- Toate operațiile trebuie să fie **complet vectorizate** (fără bucle for explicite)
- Stratul 1 folosește activare ReLU, stratul 2 folosește sigmoid
- Rețeaua trebuie să poată procesa un batch de mai multe exemple simultan
- Afișează dimensiunile tuturor matricelor pentru a verifica corectitudinea
- Metoda `prezice()` trebuie să returneze clase binare (0 sau 1) bazate pe un prag de 0.5
[
## Criterii de evaluare

1. **Corectitudinea implementării** (40%)
   - Funcțiile matematice sunt implementate corect
   - Operațiile matriceale sunt făcute corect

2. **Vectorizare** (30%)
   - Codul folosește operații NumPy vectorizate
   - Nu există bucle for inutile

3. **Claritatea codului** (20%)
   - Cod bine comentat și organizat
   - Variabile cu nume sugestive

4. **Testing și validare** (10%)
   - Testarea funcțiilor cu date de exemplu]()
   - Afișarea rezultatelor într-un mod clar

## Note importante

- Folosiți `np.random.seed()` pentru reproducibilitate
- Atenție la dimensiunile matricelor în operațiile matriceale
- Broadcasting-ul NumPy vă poate ajuta să evitați bucle
- Pentru debugging, afișați formele (shapes) matricelor intermediare

## Resurse utile

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Funcții de activare](https://en.wikipedia.org/wiki/Activation_function)
- [Vectorizare în NumPy](https://numpy.org/doc/stable/user/quickstart.html#vectorization)

**Format livrare:** Jupyter Notebook sau script Python cu output-ul rulării