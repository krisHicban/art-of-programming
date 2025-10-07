#homework 4 with "in, .lower ,len" operators to evaluate passwords
# password_input = input("\nEnter your password: ").lower()
# if len(password_input) <= 6:
#     print ("Password too weak!".upper())
# elif len(password_input) <= 10:
#     print("password is average!".upper())
# else:
#     print("strong password!".upper())

def password_strength(password):
    password = password.lower()
    if len(password) <= 6:
        return "Password too weak!".upper()
    elif len(password) <= 10:
        return "password is average!".upper()
    else:
        return "strong password!".upper()
password_input = input("\nEnter your password: ")
print(password_strength(password_input))