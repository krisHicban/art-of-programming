def calculeaza_bonus(salariu, experienta):
    """
    Calculează bonusul pe baza experienței
    Returnează: valoarea bonusului
    """
    if experienta >= 5:
        return salariu * 0.20  # 20% bonus
    elif experienta >= 3:
        return salariu * 0.15  # 15% bonus
    elif experienta >= 1:
        return salariu * 0.10  # 10% bonus
    else:
        return 0  # Fără bonus









# Definim lista de angajați
angajati = [
    {"nume": "Ana", "salariu": 4000, "experienta": 2},
    {"nume": "Mihai", "salariu": 5500, "experienta": 6},
    {"nume": "Ioana", "salariu": 4800, "experienta": 4},
    {"nume": "George", "salariu": 3000, "experienta": 1},
    {"nume": "Andreea", "salariu": 7000, "experienta": 8},
    {"nume": "Cristi", "salariu": 3500, "experienta": 0},
    {"nume": "Elena", "salariu": 6200, "experienta": 3},
    {"nume": "Vlad", "salariu": 4500, "experienta": 5},
    {"nume": "Radu", "salariu": 5100, "experienta": 2},
    {"nume": "Maria", "salariu": 3900, "experienta": 1},
]



# Calculăm bonusul fiecărui angajat
for angajat in angajati:
    # Before Functions
    # print(angajat["salariu"], angajat["experienta"])
    # calculeaza_bonus(7000, 8), calculeaza_bonus(3500, 0), etc..
    bonus = calculeaza_bonus(angajat["salariu"], angajat["experienta"])

    print(f"Salariul lui {angajat['nume']} este {angajat['salariu']} RON, "
          f"iar bonusul este {bonus} RON.")
