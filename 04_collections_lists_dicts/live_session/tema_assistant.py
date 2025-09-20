# Nevoie Client: AplicaÈ›ie simplÄƒ pentru managementul prietenilor È™i hobbyurilor â€“
# â€žRÄƒmÃ¢i conectat mereu cu oamenii apropiaÈ›iâ€.

# Scope:

# CreeazÄƒ o listÄƒ cu 5 prieteni (dicÈ›ionare cu: nume, telefon, vÃ¢rstÄƒ, ultim_contact, hobbyuri).
# AfiÈ™eazÄƒ lista.
# AdaugÄƒ un prieten nou.
# SchimbÄƒ numÄƒrul de telefon pentru un prieten.
# CreeazÄƒ un set cu toate hobbyurile unice È™i afiÈ™eazÄƒ-l.

#lista 5 prieteni cu dictionare
lista_prieteni = [
    {"nume": "Ana Popescu", "telefon": "0788393202", "varsta": 23, "ultim_contact": "3 februarie", "hobbyuri": ["karate"]},
    {"nume": "Corina Stela", "telefon": "0799393202", "varsta": 25, "ultim_contact": "2 aug", "hobbyuri": ["singing"]},
    {"nume": "Laurentiu Cioban", "telefon": "0700393202", "varsta": 24, "ultim_contact": "10 aug", "hobbyuri": ["dancing"]},
    {"nume": "Fidan Banciu", "telefon": "0722393202", "varsta": 25, "ultim_contact": "3 oct", "hobbyuri": ["karate"]},
    {"nume": "Daria Banciu", "telefon": "0733393202", "varsta": 24, "ultim_contact": "2 oct", "hobbyuri": ["teatru"]}
]



#aici am vrut sa fac lista de dictionare care au si cuvant cheie care descrie. nu s-a putut crea lista avand cuvinte cheie. Deci mai jos asta e doar un dictionar exemplu:
prietenii_mei = {
    "Ana": {
        "nume": "Ana Popescu",
        "telefon": "0788393202",
        "varsta": 23,
        "ultim_contact": "3 februarie",
        "hobbyuri": ["karate"]
    },
    "Corina": {
        "nume": "Corina Stela",
        "telefon": "0799393202",
        "varsta": 25,
        "ultim_contact": "2 aug",
        "hobbyuri": ["singing"]
    },
    "Laurentiu": {
        "nume": "Laurentiu Cioban",
        "telefon": "0700393202",
        "varsta": 24,
        "ultim_contact": "10 aug",
        "hobbyuri": ["dancing"]
    },
    "Fidan": {
        "nume": "Fidan Banciu",
        "telefon": "0722393202",
        "varsta": 25,
        "ultim_contact": "3 oct",
        "hobbyuri": ["karate"]
    },
    "Daria": {
        "nume": "Daria Banciu",
        "telefon": "0733393202",
        "varsta": 24,
        "ultim_contact": "2 oct",
        "hobbyuri": ["teatru"]
    }
}

#afiseaza lista
for prieten in lista_prieteni:  # fiecare element este un dicÈ›ionar
    print(prieten)               # afiÈ™eazÄƒ dicÈ›ionarul complet


#adauga prieten nou

lista_prieteni.append({
    "nume": 'Olga',
    "telefon": "074444444",
        "varsta": 24,
        "ultim_contact": "11 aug",
        "hobbyuri": ["eating"]
})

#schimba numarul unui prieten
for prieten in lista_prieteni:
    if prieten["nume"] == "Daria Banciu":
        prieten["telefon"] = "07111111111"

#afiseaza lista din nou dupa modificare
for prieten in lista_prieteni:  # fiecare element este un dicÈ›ionar
    print(prieten)               # afiÈ™eazÄƒ dicÈ›ionarul complet

#CreeazÄƒ un set cu toate hobbyurile unice È™i afiÈ™eazÄƒ
# CreeazÄƒ un set gol
hobbyuri_unice = set()

# Parcurge fiecare prieten È™i adaugÄƒ hobbyurile Ã®n set
for prieten in lista_prieteni:
    hobbyuri_unice.update(prieten["hobbyuri"])  # update() adaugÄƒ toate elementele din listÄƒ Ã®n set

# AfiÈ™eazÄƒ setul cu hobbyuri unice
print("Hobbyuri unice:", hobbyuri_unice)