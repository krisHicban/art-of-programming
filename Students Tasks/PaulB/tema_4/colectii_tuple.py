# TEMA 1 TUPLE MELODII:

melodii_favorite = ("golden", "soda pop", "your idol", "like my father")  # creăm tuplul cu melodiile

for index, song in enumerate(melodii_favorite, start=1):  # punem să ne indexeze fiecare melodie din tuplu în format 1. 2. ...
    print(f"{index}. {song}")


# TEMA 2 LISTA CUMPĂRĂTURI:

lista_cumparaturi = []  # creăm o listă goală de cumpărături

print("Te rog introdu produsele ce trebuie cumpărate pe rând, scrie stop când ai terminat:")

while True:  # creăm bucla
    produs = input("> ")  # cât timp bucla e adevărată ne va pune să adăugăm produse unul sub altul cu săgeata la început

    lista_cumparaturi.append(produs)  # adaugă fiecare produs pe care l-am introdus în lista de cumpărături

    if len(lista_cumparaturi) == 3:  # dacă lista are 3 produse introduse ne va printa mesajul de mai jos și va închide lista
        print("Am nevoie doar de 3 produse, mulțumesc!")
        break

print("Lista de cumpărături este:", lista_cumparaturi)  # ne afișează lista de cumpărături cu produsele introduse


# TEMA 3 ACTIVITY ASSISTANT

from typing import Any

prieteni = ["Mihai", "Marin", "Razvan", "Andrei", "Cristi"]  # creăm lista de prieteni

detalii_prieteni: dict[str, dict[str, Any]] = {   # creăm dicționarul cu detaliile prietenilor
    "Mihai": {
        "nume": "Mărginean Mihai",
        "telefon": "0749315226",
        "varsta": "36 de ani",
        "ultim contact": "cu 4 zile în urmă",
        "hobby-uri": "mașini CNC, seriale"
    },

    "Marin": {
        "nume": "Tolcă Marin",
        "telefon": "0752864913",
        "varsta": "32 de ani",
        "ultim contact": "acum o săptămână",
        "hobby-uri": "seriale, jocuri video, citit"
    },

    "Razvan": {
        "nume": "Szekey Răzvan",
        "telefon": "0742015386",
        "varsta": "24 de ani",
        "ultim contact": "1 an în urmă",
        "hobby-uri": "muzică, fotografie, mașini"
    },

    "Andrei": {
        "nume": "Moldovan Andrei",
        "telefon": "0755290318",
        "varsta": "28 de ani",
        "ultim contact": "acum 2 săptămâni",
        "hobby-uri": "fotbal, mașini"
    },

    "Cristi": {
        "nume": "Oltean Cristian",
        "telefon": "0740155294",
        "varsta": "43 de ani",
        "ultim contact": "ieri",
        "hobby-uri": "animale de companie, jocuri video"
    }
}

for nume in prieteni:   # printăm lista de prieteni
    print(nume)

detalii_prieteni["Mirela"] = {   # adăugăm prieten nou în listă
    "nume": "Cioșan Mirela",
    "telefon": "0752961238",
    "varsta": "19 ani",
    "ultim contact": "în urmă cu 3 ani",
    "hobby-uri": "muzică, sport, desen"
}

detalii_prieteni["Cristi"]["telefon"] = "0753911396"  # schimbăm nr de telefon al unui prieten

hobby_uri_unice = set()

for detalii in detalii_prieteni.values():
    hobbyuri = detalii["hobby-uri"].split(",")  # separăm hobby-urile prin virgulă
    hobby_uri_unice.update(hobbyuri)            # .update face hobby-urile să nu aibă dubluri

print("Hobby-uri unice:")
for hobby in hobby_uri_unice:                  # ne face o listă cu hobby-urile unice
    print("-", hobby)
