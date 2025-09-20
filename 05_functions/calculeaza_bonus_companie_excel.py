# First we manually populate an excel with the column [Angajat], [Salary], [Experience]
# Make it 1000 lines for real world  (use random to populate)

# import excel needed libraries to perform the task
# import os (i guess if we read and write in the computer we need a way to actually create new files/read/modify)

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

# Read From Excel (if Excel doesn't exist, create and write)
angajati = []

# Calculăm bonusul fiecărui angajat
for angajat in angajati:
    bonus = calculeaza_bonus(angajat["salariu"], angajat["experienta"])

    print(f"Salariul lui {angajat['nume']} este {angajat['salariu']} RON, "
          f"iar bonusul este {bonus} RON.")



# At the end, we write in the excel in a forth column the Bonus for each employee