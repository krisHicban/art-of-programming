# TEMA 1 TUPLE MELODII:

melodii_favorite = ("golden", "soda pop", "your idol", "like my father")  # creăm tuplul cu melodiile

for index, song in enumerate(melodii_favorite,
                             start=1):  # punem să ne indexeze fiecare melodie din tuplu în format 1. 2. ...
    print(f"{index}. {song}")


