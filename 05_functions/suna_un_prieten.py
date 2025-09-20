

def suna_un_prieten_random():
    # Definesc agenda mea telefonica
    agenda = {
        "Andrei": "0721123456",
        "Maria": "0734567890",
        "Alex": "0745678901",
        "Ana": "0756789012",
        "Radu": "0767890123",
        "Elena": "0778901234"
    }

    # Aleg un numar random de la len(agenda)
    import random
    nume_random = random.choice(list(agenda.keys()))

    # Sun prietenul(demonstrez printr-un print)
    print(f"Sun pe {nume_random} la numărul {agenda[nume_random]}")


def suna_un_prieten(nume):
    # Dictionar cu nume si numere
    agenda = {
        "Andrei": "0721123456",
        "Maria": "0734567890",
        "Alex": "0745678901",
        "Ana": "0756789012",
        "Radu": "0767890123",
        "Elena": "0778901234"
    }

    # Gasesc prietenul in dictionar si sun prietenul(demonstrez printr-un print)
    if nume in agenda:
        print(f"Sun pe {nume} la numărul {agenda[nume]}")
    # else:
    #     print(f"Nu am găsit pe {nume} în agendă")
















##################################
####### Program Principal ########
##################################

# Suna un Prieten Random
suna_un_prieten_random()

# Testez și funcția pentru un prieten specific
suna_un_prieten("Maria")


# Continui ce am de facut in task-ul asta