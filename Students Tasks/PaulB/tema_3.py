#TEMA 1 PIN-UL


pin = "6205"   #pin-ul correct

while True:    #setam o bucla pentru introducerea pin-ului
    user_input = input("Te rog introdu codul pin de 4 cifre:\n")
    if user_input == pin:    #Verific daca pin-ul e corect
        print("\n Cod pin acceptat \n") 
        break 
    else:   #Daca pin-ul e gresit imi va printa mesajul de mai jos
        print("\n Codul pin e gresit,incearca din nou. \n")


#TEMA 2 NUMARATOAREA


numere = []  #Creeaza o lista goala numita numere

for i in range(1,11):   # range(1, 11)ne da numerele de la 1 la 10,finalul adica nr 11 nu o sa il afiseze.
    numere.append(i)    #append(i) ne adauga fiecare numar in listafor number in numere:

for index, number in enumerate(numere, start=1):  #ne pune o numaratoare in fata fiecarui numar
    print(f"{index}. {number}") #formeaza numaratoare in stilul 1. 1


#TEMA 3 PIRAMIDA INVERSA


max_width = 11

for stars in range(max_width, 0, -2): #prima linie va avea 11 stele
                                      #Liniile urmatoare sunt reduse cu 2 stele                                                                                                              
    spaces = (max_width - stars) // 2 #spatiile cresc din partea stanga ca sa alinieze piramida inversa
    print(" " * spaces + "*" * stars)


#TEMA 4 ADUNAREA NR PARE 1-100


total = 0 #aici o sa tina cont de totalul sumei numerelor pare
for number in range(1,101): #aceasta bucla trece prin fiecare numar de la 1 la 100 (101 e exclus)
    if number % 2 == 0:  #verifica daca numarul care e la rand se imparte exact la 2(daca e par)
        total += number #aici adunam numarul par cu totalul
print("Suma tuturor numerelor pare este:", total)