import time

pin_input = "2303"
attempts = 0

while attempts < 3:
    pin = input("\nEnter a 4-digit PIN: ")

    if len(pin) == 4:
        if pin == pin_input:
            print("\nPIN accepted.")
            break

        else:
            attempts += 1
            if attempts == 1:
                print("\nIncorrect PIN. Waiting 15 seconds before next attempt...")
                time.sleep(15)
            elif attempts == 2:
                print("\nIncorrect PIN. Waiting 60 seconds before next attempt...")
                time.sleep(60)
            elif attempts == 3:
                print("\nToo many failed attempts. Access blocked.")
    else:
        print("\nInvalid input. Please enter exactly 4 digits.")
else:
    print("\nYour account is now blocked due to multiple failed attempts.")