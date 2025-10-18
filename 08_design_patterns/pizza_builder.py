class Pizza:
    """Produsul final complex"""

    def __init__(self):
        self.marime = None
        self.aluat = None
        self.sos = None
        self.branza = False
        self.toppings = []
        self.pret = 0

    def __str__(self):
        toppings_text = ", ".join(self.toppings) if self.toppings else "Fără toppings"
        branza_text = "cu brânză" if self.branza else "fără brânză"
        return f"🍕 Pizza {self.marime} cu aluat {self.aluat}, sos {self.sos}, {branza_text}, toppings: {toppings_text} - {self.pret} RON"


class PizzaBuilder:
    """
    Builder Pattern - construire pas cu pas
    """

    def __init__(self):
        self.pizza = Pizza()

    def set_marime(self, marime):
        """Pas 1: Setează mărimea"""
        self.pizza.marime = marime
        print(f"🔧 Mărime setată: {marime}")
        return self  # Return self pentru method chaining!

    def set_aluat(self, aluat):
        """Pas 2: Setează alatul"""
        self.pizza.aluat = aluat
        print(f"🔧 Aluat setat: {aluat}")
        return self

    def set_sos(self, sos):
        """Pas 3: Setează sosul"""
        self.pizza.sos = sos
        print(f"🔧 Sos setat: {sos}")
        return self

    def add_branza(self):
        """Pas 4: Adaugă brânză"""
        self.pizza.branza = True
        print("🔧 Brânză adăugată")
        return self

    def add_topping(self, topping):
        """Pas 5: Adaugă topping"""
        self.pizza.toppings.append(topping)
        print(f"🔧 Topping adăugat: {topping}")
        return self

    def calculeaza_pret(self):
        """Calculează prețul final"""
        pret = {'Mică': 20, 'Medie': 30, 'Mare': 40}[self.pizza.marime]

        if self.pizza.aluat == 'Integrală': pret += 5
        if self.pizza.aluat == 'Fără gluten': pret += 8
        if self.pizza.branza: pret += 10
        pret += len(self.pizza.toppings) * 8

        self.pizza.pret = pret
        return self

    def build(self):
        """Finalizează și returnează pizza"""
        self.calculeaza_pret()
        pizza_finala = self.pizza
        self.pizza = Pizza()  # Reset pentru următoarea pizza
        print("✅ Pizza finalizată!")
        return pizza_finala


# UTILIZARE - Method Chaining elegant!
builder = PizzaBuilder()

pizza1 = (builder.set_marime("Mare")
          .set_aluat("Integrală")
          .set_sos("Roșii")
          .add_branza()
          .add_topping("Pepperoni")
          .add_topping("Ciuperci")
          .build())

print(pizza1)

# Alternativ - pas cu pas:
pizza2 = PizzaBuilder()
pizza2.set_marime("Mică")
pizza2.set_aluat("Tradițional")
pizza2.set_sos("Alb")
pizza2.add_topping("Șuncă")
result = pizza2.build()

print(result)
