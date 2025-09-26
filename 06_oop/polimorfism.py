class Animal:
    """
    Clasa de bază pentru toate animalele
    """

    def __init__(self, nume):
        self.nume = nume

    def vorbeste(self):
        """Metodă abstractă - va fi suprascrisă"""
        pass

    def mananca(self):
        return f"{self.nume} mănâncă"


class Caine(Animal):
    def vorbeste(self):
        return f"{self.nume}: Ham ham! 🐕"


class Pisica(Animal):
    def vorbeste(self):
        return f"{self.nume}: Miau! 🐱"


class Papagal(Animal):
    def vorbeste(self):
        return f"{self.nume}: Polly wants a cracker! 🦜"


class Peste(Animal):
    def vorbeste(self):
        return f"{self.nume}: Blub blub... (nu face sunet) 🐟"


animal_1 = Caine("Dodo")

# Crearea unei liste cu animale(instante/obiecte) diferite care mostenesc clasa Animal:
animale = [
    Caine("Rex"),
    Pisica("Mimi"),
    Papagal("Tweety"),
    Peste("Goldy")
]


# POLIMORFISM în acțiune:
for animal in animale:
    # Aceeași metodă, comportamente diferite:
    print(animal.vorbeste())
