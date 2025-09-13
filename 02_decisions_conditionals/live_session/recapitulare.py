
# nume = input("Cum te cheamă? ")
# varsta = int(input("Câți ani ai? "))
# print(f"Salut, {nume}, la anul vei avea {varsta + 1} ani.")
#
#
# mesaj = input("Scrie un mesaj: ")
# print(mesaj + '\n' * 5)



#############################
###### Rezolvare Tema ######
#############################

print('\n' * 3)
confirmare = input("Continuăm cu tema? (da/nu): ").strip().lower()
if confirmare == "da":
    nume = input("Introdu numele: ")
    varsta = int(input("Introdu vârsta: "))
    localitate = input("Introdu localitatea: ")

    # Afișăm mesajul de salut
    print(f"Salut, {nume} din {localitate}!")

    # Calculăm vârsta peste 10 ani
    print(f"Peste 10 ani vei avea {varsta + 10} ani.")
