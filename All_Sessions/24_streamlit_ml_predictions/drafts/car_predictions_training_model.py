import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import joblib

# === 1. Date ===
caracteristici = [
    [1, 20, 1.2],
    [2, 35, 1.4],
    [3, 60, 1.6],
    [4, 80, 1.6],
    [5, 120, 1.8],
    [6, 150, 2.0],
    [7, 180, 2.0],
]
preturi = [12000, 11000, 9500, 8500, 7000, 6000, 5000]

X = np.array(caracteristici)
y = np.array(preturi)

# === 2. Creare model ===
model = LinearRegression()
model.fit(X, y)

# === 3. Rezultate ===
print("Coeficienți:", model.coef_)
print("Intercept:", model.intercept_)
print("Scor R²:", model.score(X, y))

# === 4. Predicție pentru [4, 75, 1.6] ===
sample = np.array([[4, 75, 1.6]])
predicted_price = model.predict(sample)
print(f"Preț estimat pentru mașina [4 ani, 75k km, 1.6L]: {predicted_price[0]:.2f} EUR")

# === 5. Grafic 3D ===
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

x_age = X[:, 0]
y_km = X[:, 1]
z_price = y

ax.scatter(x_age, y_km, z_price, c='r', marker='o', s=50, label="Date reale")

ax.set_xlabel("Vechime (ani)")
ax.set_ylabel("Rulaj (mii km)")
ax.set_zlabel("Preț (€)")
ax.set_title("Regresie liniară multiplă - Prețuri mașini")

plt.legend()
plt.show()

# === 6. Salvare model ===
joblib.dump(model, "car_price_model.pkl")
print("Model salvat în car_price_model.pkl")

# === 7. Bonus: Input utilizator ===
try:
    age = int(input("Introduceți vechimea mașinii (ani): "))
    km = float(input("Introduceți rulajul (mii km): "))
    engine = float(input("Introduceți capacitatea motorului (litri): "))

    custom_sample = np.array([[age, km, engine]])
    custom_pred = model.predict(custom_sample)
    print(f"Preț estimat pentru mașina dvs.: {custom_pred[0]:.2f} EUR")
except Exception as e:
    print("Nu s-a putut procesa input-ul utilizatorului.", e)
