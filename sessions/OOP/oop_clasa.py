class Student:
    """
    Clasa Student - șablonul pentru obiecte student
    """
    def __init__(self, nume, clasa):
        self.nume = nume          # Atribut
        self.clasa = clasa        # Atribut
        self.note = []            # Atribut (listă goală)
    
    def adauga_nota(self, nota):
        """Metodă pentru adăugarea unei note"""
        if 1 <= nota <= 10:
            self.note.append(nota)
            return True
        return False
    
    def calculeaza_media(self):
        """Metodă pentru calcularea mediei"""
        if len(self.note) == 0:
            return 0
        return sum(self.note) / len(self.note)
    
    def __str__(self):
        """Reprezentarea string a obiectului"""
        return f"Student: {'{'}self.nume{'}'} din clasa {'{'}self.clasa{'}'}"

# Crearea obiectelor (instanțiere)
student1 = Student("Ana Popescu", "10A")
student2 = Student("Ion Marinescu", "10B")

# Folosirea metodelor
student1.adauga_nota(9)
student1.adauga_nota(8)
student2.adauga_nota(7)

print(f"Media lui {student1.nume}: {student1.calculeaza_media():.2f}")
print(f"Media lui {student2.nume}: {student2.calculeaza_media():.2f}")
