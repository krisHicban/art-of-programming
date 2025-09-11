# Real-world example: ATM Machine Logic
max_attempts = 3
attempts = 0
correct_password = "1234"
balance = 10000
while attempts < max_attempts:
    password = input(f"\nEnter password (Attempt {attempts + 1}/{max_attempts}): ")
    if password == correct_password:
        print("\nAccess granted! Welcome!")
        break
    else:
        attempts += 1
        if attempts < max_attempts:
            print(f"\nWrong password. {max_attempts - attempts} attempts left.")
        else:
            print("\nAccount locked. Too many failed attempts.")


    


