# Clasele de animale (toate au aceeași interfață)
class Animal:
    def __init__(self, nume):
        self.nume = nume

    def face_sunet(self):
        pass


class Caine(Animal):
    def face_sunet(self):
        return f"{self.nume}: Ham ham! 🐕"


class Pisica(Animal):
    def face_sunet(self):
        return f"{self.nume}: Miau! 🐱"


class Vaca(Animal):
    def face_sunet(self):
        return f"{self.nume}: Muuu! 🐄"


class Oaie(Animal):
    def face_sunet(self):
        return f"{self.nume}: Beeee! 🐑"


class Rata(Animal):
    def face_sunet(self):
        return f"{self.nume}: Mac mac! 🦆"







# FACTORY PATTERN - creează animalele centralizat
class AnimalFactory:
    """
    Factory care știe să creeze orice tip de animal
    """
    @staticmethod
    def creeaza_animal(tip_animal, nume):
        """
        Metoda principală - primește tipul și returnează obiectul
        """
        if tip_animal.lower() == "caine":
            return Caine(nume)
        elif tip_animal.lower() == "pisica":
            return Pisica(nume)
        elif tip_animal.lower() == "vaca":
            return Vaca(nume)
        elif tip_animal.lower() == "oaie":
            return Oaie(nume)
        elif tip_animal.lower() == "rata":
            return Rata(nume)
        else:
            raise ValueError(f"Tipul '{tip_animal}' nu este suportat!")


# UTILIZARE - mult mai simplă!
def creeaza_ferma():
    factory = AnimalFactory()

    # În loc să scrii manual:
    # if tip == "caine": animal = Caine(nume)
    # elif tip == "pisica": animal = Pisica(nume)
    # ... (repetitiv și urat)

    animale = []



    # Folosești factory-ul:
    primul_animal = factory.creeaza_animal("caine", "Rex")
    animale.append(primul_animal)


    animale.append(factory.creeaza_animal("pisica", "Mimi"))
    animale.append(factory.creeaza_animal("vaca", "Maia"))

    # Concert de animale:
    for animal in animale:
        print(animal.face_sunet())


creeaza_ferma()

# 🎯 AVANTAJE Factory Pattern:
# ✅ Centralizează logica de creare
# ✅ Ușor de extins (adaugi noi animale)
# ✅ Codul client nu știe de clase specifice
# ✅ Respectă principiul "Open/Closed"