class Utilizator:
    """
    Clasa de bază pentru utilizatori
    Demonstrează INCAPSULARE cu parolă privată
    """
    def __init__(self, nume, email, parola):
        self.nume = nume
        self.email = email
        self.__parola = parola  # PRIVAT - incapsulare!
        self.este_activ = True

    def login(self, parola_introdusa):
        """Metodă pentru login"""
        if self.__parola == parola_introdusa:
            return f"✅ {self.nume} s-a logat cu succes!"
        return "❌ Parolă incorectă!"

    def schimba_parola(self, parola_veche, parola_noua):
        """Metodă pentru schimbarea parolei"""
        if self.__parola == parola_veche:
            self.__parola = parola_noua
            return f"🔒 {self.nume} și-a schimbat parola!"
        return "❌ Parolă veche incorectă!"

    def afiseaza_profil(self):
        return f"👤 {self.nume} - {self.email}"


class Administrator(Utilizator):
    """
    Clasa Admin MOȘTENEȘTE din Utilizator
    Adaugă funcționalități specifice adminului
    """

    def __init__(self, nume, email, parola):
        super().__init__(nume, email, parola)  # Moștenire
        self.permisiuni = ['citire', 'scriere', 'stergere']

    def sterge_utilizator(self, utilizator):
        """Metodă NOUĂ - doar adminii pot șterge"""
        return f"🗑️ Admin {self.nume} a șters utilizatorul {utilizator.nume}"

    def afiseaza_profil(self):
        """Metodă SUPRASCRISĂ (override)"""
        return f"👑 Admin {self.nume} - {self.email}"


# POLIMORFISM în acțiune:
def procesează_login(lista_utilizatori, parola):
    """O funcție pentru toți utilizatorii"""
    for user in lista_utilizatori:
        print(user.login(parola))  # Comportament identic
        print(user.afiseaza_profil())  # Comportament diferit!


# Utilizare:
user1 = Utilizator("Ana", "ana@email.com", "parola123")
admin1 = Administrator("Ion Admin", "admin@site.com", "admin123")

print(user1.login("parola123"))  # ✅ Login user
print(admin1.login("admin123"))  # ✅ Login admin (moștenit)
print(admin1.sterge_utilizator(user1))  # 🗑️ Doar adminii pot!