# Real-world example: Student Grade System
score = int(input("Enter your exam score (0-100): "))

if score >= 90:
    grade = "A"
    message = "Excellent work!"
elif score >= 80:
    grade = "B"
    message = "Good job!"
elif score >= 70:
    grade = "C"
    message = "You passed!"
elif score >= 60:
    grade = "D"
    message = "You need to improve"
else:
    grade = "F"
    message = "You failed. Try again!"

print(f"Grade: {grade} - {message}")





# Real-world example: ATM Machine Logic
balance = 1000
pin = "1234"

entered_pin = input("Enter your PIN: ")
if entered_pin == pin:
    print(f"Welcome! Your balance is ${balance}")

    amount = float(input("How much would you like to withdraw? $"))
    if amount <= balance:
        balance -= amount
        print(f"Transaction successful! New balance: ${balance}")
    else:
        print("Insufficient funds!")
else:
    print("Invalid PIN. Access denied!")







# Nested conditions: Weather app
temperature = int(input("Enter temperature in Celsius: "))
is_raining = input("Is it raining? (yes/no): ").lower() == "yes"

if temperature > 25:
    if is_raining:
        clothing = "Light jacket and umbrella"
    else:
        clothing = "T-shirt and shorts"
elif temperature > 15:
    if is_raining:
        clothing = "Jacket and umbrella"
    else:
        clothing = "Light sweater"
else:
    if is_raining:
        clothing = "Heavy coat and umbrella"
    else:
        clothing = "Warm coat"

print(f"Recommended clothing: {clothing}")