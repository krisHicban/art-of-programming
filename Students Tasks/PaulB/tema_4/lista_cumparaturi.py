
# TEMA 2 LISTA CUMPĂRĂTURI:

lista_cumparaturi = []  # creăm o listă goală de cumpărături

print("Te rog introdu produsele ce trebuie cumpărate pe rând, scrie stop când ai terminat:")

while True:  # creăm bucla
    produs = input(
        "> ")  # cât timp bucla e adevărată ne va pune să adăugăm produse unul sub altul cu săgeata la început

    lista_cumparaturi.append(produs)  # adaugă fiecare produs pe care l-am introdus în lista de cumpărături

    if len(lista_cumparaturi) == 3:  # dacă lista are 3 produse introduse ne va printa mesajul de mai jos și va închide lista
        print("Am nevoie doar de 3 produse, mulțumesc!")
        break

print("Lista de cumpărături este:", lista_cumparaturi)  # ne afișează lista de cumpărături cu produsele introduse