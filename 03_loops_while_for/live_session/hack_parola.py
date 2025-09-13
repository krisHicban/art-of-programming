import string

# --- Step 1: User sets password ---
parola_corecta = input("Setează parola (exact 4 caractere, litere sau cifre): ")

# --- Step 2: Define possible characters ---
caractere = string.ascii_lowercase + string.digits  # "abcdefghijklmnopqrstuvwxyz0123456789"

# --- Step 3: Generate all combinations of length 4 ---
gasit = False
incercari = 0

# Brute-Force - 

for c1 in caractere:
    for c2 in caractere:
        for c3 in caractere:
            for c4 in caractere:
                incercari += 1
                test = c1 + c2 + c3 + c4
                if test == parola_corecta:
                    print(f"Parola găsită: {test} (după {incercari} încercări)")
                    gasit = True
                    break
            if gasit:
                break
        if gasit:
            break
    if gasit:
        break
