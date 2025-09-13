# ===============================
# SIMULARE CLASĂ FĂRĂ LISTE/DICT
# ===============================

# Stocăm fiecare student separat
student_1_id = "Ana123"
student_1_nume = "Ana Popescu"
student_1_clasa = "10A"
student_1_nota_1 = 9
student_1_nota_2 = 8
student_1_nota_3 = 10

student_2_id = "Dana"
student_2_nume = "Dana Popescu"
student_2_clasa = "10A"
student_2_nota_1 = 5
student_2_nota_2 = 8
student_2_nota_3 = 10

student_3_id = "Maria999"
student_3_nume = "Maria Vasile"
student_3_clasa = "10C"
# Maria nu are note stocate
student_3_nota_1 = None
student_3_nota_2 = None
student_3_nota_3 = None


# ===============================
# ACCESAREA DATELOR
# ===============================
print("Datele lui Ana:")
print("Nume:", student_1_nume)
print("Clasa:", student_1_clasa)
print("Note:", student_1_nota_1, student_1_nota_2, student_1_nota_3)

print("\nDatele Danăi:")
print("Nume:", student_2_nume)
print("Clasa:", student_2_clasa)
print("Note:", student_2_nota_1, student_2_nota_2, student_2_nota_3)


# ===============================
# "PARCURGEREA" STUDENȚILOR
# ===============================
print("\nParcurgere toți studenții:")

# Student 1
print(student_1_id, "->", student_1_nume, "(", student_1_clasa, ") Note:",
      student_1_nota_1, student_1_nota_2, student_1_nota_3)

# Student 2
print(student_2_id, "->", student_2_nume, "(", student_2_clasa, ") Note:",
      student_2_nota_1, student_2_nota_2, student_2_nota_3)

# Student 3
print(student_3_id, "->", student_3_nume, "(", student_3_clasa, ") Note:",
      student_3_nota_1, student_3_nota_2, student_3_nota_3)


# ===============================
# ADAUGAREA UNUI NOU STUDENT
# ===============================
student_4_id = "George77"
student_4_nume = "George Ionescu"
student_4_clasa = "11B"
student_4_nota_1 = 7
student_4_nota_2 = 6
student_4_nota_3 = 9

print("\nParcurgere după adăugare:")

# Student 1
print(student_1_id, ":", student_1_nume)

# Student 2
print(student_2_id, ":", student_2_nume)

# Student 3
print(student_3_id, ":", student_3_nume)

# Student 4
print(student_4_id, ":", student_4_nume)
