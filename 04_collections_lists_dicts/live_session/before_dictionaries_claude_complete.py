# Simularea unui dicționar de studenți folosind DOAR variabile
# FĂRĂ FUNCȚII - doar variabile și cod redundant manual!

# ========== CREAREA STUDENȚILOR (echivalent cu dicționarul inițial) ==========
# Student 1: Ana123
student_1_id = "Ana123"
student_1_nume = "Ana Popescu"
student_1_clasa = "10A"
student_1_nota_1 = 9
student_1_nota_2 = 8
student_1_nota_3 = 10

# Student 2: Dana
student_2_id = "Dana"
student_2_nume = "Dana Popescu"
student_2_clasa = "10A"
student_2_nota_1 = 5
student_2_nota_2 = 8
student_2_nota_3 = 10

total_studenti = 2

print("=== DATELE INIȚIALE ===")
print(f"Student 1: {student_1_id} - {student_1_nume} - {student_1_clasa}")
print(f"Student 2: {student_2_id} - {student_2_nume} - {student_2_clasa}")

# ========== ACCESAREA DATELOR (echivalent cu studenti["Ana123"]["clasa"]) ==========
# Vrem clasa lui Ana123 - trebuie să o căutăm manual!
id_cautat = "Ana123"
clasa_gasita = ""

# Verificăm manual fiecare student
if student_1_id == id_cautat:
    clasa_gasita = student_1_clasa
elif student_2_id == id_cautat:
    clasa_gasita = student_2_clasa

print(f"\nClasa lui {id_cautat}: {clasa_gasita}")

# ========== AFIȘAREA DATELOR UNUI STUDENT (echivalent cu date_student = studenti["Ana123"]) ==========
# Vrem datele complete ale lui Ana123
id_cautat = "Ana123"

# Căutăm manual și afișăm
if student_1_id == id_cautat:
    print(f"\nDatele lui {id_cautat}:")
    print(f"ID: {student_1_id}")
    print(f"Nume: {student_1_nume}")
    print(f"Clasa: {student_1_clasa}")
    print(f"Note: [{student_1_nota_1}, {student_1_nota_2}, {student_1_nota_3}]")
elif student_2_id == id_cautat:
    print(f"\nDatele lui {id_cautat}:")
    print(f"ID: {student_2_id}")
    print(f"Nume: {student_2_nume}")
    print(f"Clasa: {student_2_clasa}")
    print(f"Note: [{student_2_nota_1}, {student_2_nota_2}, {student_2_nota_3}]")

# ========== PARCURGEREA DICȚIONARULUI (echivalent cu for x, date in studenti.items()) ==========
print("\n=== PARCURGEREA STUDENȚILOR ===")

# Trebuie să afișăm manual fiecare student și notele lui
print(f"{student_1_id}: [{student_1_nota_1}, {student_1_nota_2}, {student_1_nota_3}]")
print(f"{student_2_id}: [{student_2_nota_1}, {student_2_nota_2}, {student_2_nota_3}]")

# ========== ADĂUGAREA UNUI STUDENT NOU ==========
print("\n=== ADĂUGAREA STUDENT NOU ===")

# Adăugăm Maria999 - trebuie variabile noi!
student_3_id = "Maria999"
student_3_nume = "Maria Vasile"
student_3_clasa = "10C"
# Maria nu are note încă
student_3_nota_1 = 0  # sau None
student_3_nota_2 = 0
student_3_nota_3 = 0

total_studenti = 3

print("Student Maria999 adăugat!")

# ========== PARCURGEREA DUPĂ ADĂUGARE ==========
print("\nParcurgere după adăugare:")

# Trebuie să rescriu tot codul pentru toți 3 studenții!
print(f"{student_1_id}: {student_1_nume}")
print(f"{student_2_id}: {student_2_nume}")
print(f"{student_3_id}: {student_3_nume}")

# ========== OPERAȚII SUPLIMENTARE - TOATE MANUAL! ==========

# Să calculez media lui Ana123
id_pentru_medie = "Ana123"
medie_calculata = 0

if student_1_id == id_pentru_medie:
    medie_calculata = (student_1_nota_1 + student_1_nota_2 + student_1_nota_3) / 3
elif student_2_id == id_pentru_medie:
    medie_calculata = (student_2_nota_1 + student_2_nota_2 + student_2_nota_3) / 3
elif student_3_id == id_pentru_medie:
    medie_calculata = (student_3_nota_1 + student_3_nota_2 + student_3_nota_3) / 3

print(f"\nMedia lui {id_pentru_medie}: {medie_calculata:.2f}")

# Să calculez media lui Dana
id_pentru_medie = "Dana"
medie_calculata = 0

if student_1_id == id_pentru_medie:
    medie_calculata = (student_1_nota_1 + student_1_nota_2 + student_1_nota_3) / 3
elif student_2_id == id_pentru_medie:
    medie_calculata = (student_2_nota_1 + student_2_nota_2 + student_2_nota_3) / 3
elif student_3_id == id_pentru_medie:
    medie_calculata = (student_3_nota_1 + student_3_nota_2 + student_3_nota_3) / 3

print(f"Media lui {id_pentru_medie}: {medie_calculata:.2f}")

# Să caut un student după nume
nume_cautat = "Maria Vasile"
id_gasit = ""

if student_1_nume == nume_cautat:
    id_gasit = student_1_id
elif student_2_nume == nume_cautat:
    id_gasit = student_2_id
elif student_3_nume == nume_cautat:
    id_gasit = student_3_id

print(f"\nID-ul pentru {nume_cautat}: {id_gasit}")

# Să număr câți studenți sunt în clasa 10A
clasa_cautata = "10A"
numar_studenti_clasa = 0

if student_1_clasa == clasa_cautata:
    numar_studenti_clasa = numar_studenti_clasa + 1
if student_2_clasa == clasa_cautata:
    numar_studenti_clasa = numar_studenti_clasa + 1
if student_3_clasa == clasa_cautata:
    numar_studenti_clasa = numar_studenti_clasa + 1

print(f"Numărul de studenți în clasa {clasa_cautata}: {numar_studenti_clasa}")

# Să afișez toți studenții cu notele lor - manual pentru fiecare!
print(f"\nToți studenții cu notele:")
print(f"Student: {student_1_nume}")
print(f"  Nota 1: {student_1_nota_1}")
print(f"  Nota 2: {student_1_nota_2}")
print(f"  Nota 3: {student_1_nota_3}")

print(f"Student: {student_2_nume}")
print(f"  Nota 1: {student_2_nota_1}")
print(f"  Nota 2: {student_2_nota_2}")
print(f"  Nota 3: {student_2_nota_3}")

print(f"Student: {student_3_nume}")
print(f"  Nota 1: {student_3_nota_1}")
print(f"  Nota 2: {student_3_nota_2}")
print(f"  Nota 3: {student_3_nota_3}")

# Să modificăm o notă - trebuie să știu exact care variabilă!
# Să schimb nota 2 a lui Dana din 8 în 9
id_de_modificat = "Dana"

if student_1_id == id_de_modificat:
    student_1_nota_2 = 9
elif student_2_id == id_de_modificat:
    student_2_nota_2 = 9
elif student_3_id == id_de_modificat:
    student_3_nota_2 = 9

print(f"\nNota modificată pentru {id_de_modificat}!")

# Să verific modificarea - din nou, manual!
if student_2_id == "Dana":
    print(f"Noile note ale lui Dana: [{student_2_nota_1}, {student_2_nota_2}, {student_2_nota_3}]")

# ========== ÎNCERCAREA DE A ADĂUGA AL 4-LEA STUDENT ==========
# Oh nu! Trebuie să creez din nou toate variabilele!

student_4_id = "Ion123"
student_4_nume = "Ion Dumitrescu"
student_4_clasa = "10B"
student_4_nota_1 = 7
student_4_nota_2 = 6
student_4_nota_3 = 8

total_studenti = 4

# Și acum trebuie să rescriu TOATE operațiile de mai sus pentru a include și student_4!
# De exemplu, parcurgerea:
print(f"\nToți cei 4 studenți:")
print(f"{student_1_id}: {student_1_nume}")
print(f"{student_2_id}: {student_2_nume}")
print(f"{student_3_id}: {student_3_nume}")
print(f"{student_4_id}: {student_4_nume}")

print("\n" + "="*60)
print("COȘMARUL ACESTEI ABORDĂRI:")
print("="*60)
print("• Pentru fiecare operație trebuie să scriu cod pentru FIECARE student")
print("• Dacă adaug un student nou, trebuie să modific TOATE operațiile")
print("• Cu 100 de studenți ar fi 1000+ linii de cod repetitiv")
print("• Orice modificare devine un coșmar de copy-paste")
print("• Extrem de predispos la erori")
print("• Imposibil de menținut sau scalat")
print("• Nu pot avea un număr dinamic de studenți")
print("\nFĂRĂ dicționare și liste, programarea ar fi IMPOSIBILĂ!")