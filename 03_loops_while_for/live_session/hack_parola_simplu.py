# --- Step 1: User sets password ---
parola_corecta = int(input("Setează parola (5 cifre): "))


# Complexitate O(n)
for guess in range(100000):  # de la 0 la 9999
    if guess == parola_corecta:
        print(f"Parola găsită:", guess)
        break
