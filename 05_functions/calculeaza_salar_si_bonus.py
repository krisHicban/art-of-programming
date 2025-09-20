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
        print("Am intrat in else-if-ul de experienta intre 1 an si 3 ani")
        return salariu * 0.10  # 10% bonus
    else:
        return 0  # Fără bonus

def salariu_total(salariu_baza, experienta):
    """
    Calculează salariul total (baza + bonus)
    """
    bonus = calculeaza_bonus(salariu_baza, experienta)
    return salariu_baza + bonus



# Exemple de utilizare:
total = salariu_total(5000, 2)
print(f"Total: {total} RON")






