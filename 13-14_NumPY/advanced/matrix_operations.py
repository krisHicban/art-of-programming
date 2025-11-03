# Operații fundamentale cu matrici
import numpy as np

# Definirea matricilor
A = np.array([[2,1],[1,3]])
B = np.array([[4,0],[1,2]])

# Operații
rezultat = np.dot(A, B)  # Înmulțire matriceală

# În spatele fiecărei rețele neuronale!
print(f"Rezultat: \n{rezultat}")