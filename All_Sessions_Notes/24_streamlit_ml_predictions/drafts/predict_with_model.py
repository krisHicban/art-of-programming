import numpy as np
import joblib

# Încărcare model salvat
model = joblib.load("car_price_model.pkl")

# Exemplu de predicție
sample = np.array([[4, 75, 1.6]])
predicted_price = model.predict(sample)
print(f"Preț estimat pentru [4 ani, 75k km, 1.6L]: {predicted_price[0]:.2f} EUR")

# Input utilizator
try:
    age = int(input("Introduceți vechimea mașinii (ani): "))
    km = float(input("Introduceți rulajul (mii km): "))
    engine = float(input("Introduceți capacitatea motorului (litri): "))

    custom_sample = np.array([[age, km, engine]])
    custom_pred = model.predict(custom_sample)
    print(f"Preț estimat pentru mașina dvs.: {custom_pred[0]:.2f} EUR")
except Exception as e:
    print("Nu s-a putut procesa input-ul utilizatorului.", e)
