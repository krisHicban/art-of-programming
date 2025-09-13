
# Crearea unui dicționar
studenti = {
    "Ana123": {
        "nume": "Ana Popescu",
        "clasa": "10A",
        "note": [9, 8, 10]
    }
    ,
    "Dana": {
        "nume": "DAna Popescu",
        "clasa": "10A",
        "note": [5, 8, 10]
    }
}


# # Accesarea datelor
# x = studenti["Ana123"]["clasa"]
# print(x)


date_student = studenti["Ana123"]
print(date_student)


# Parcurgerea dicționarului
for x, date in studenti.items():
    print(f"{x}: {date['note']}")



# Adăugarea unui student nou
studenti["Maria999"] = {
    "nume": "Maria Vasile",
    "clasa": "10C"
}


print("\nParcurgere dupa adaugare: ")
# Parcurgerea dicționarului
for id_student, date in studenti.items():
    print(f"{id_student}: {date['nume']}")