#homework 4 with "in, .lower ,len" operators to evaluate passwords
password_input = input("\nEnter your password: ").lower()
if len(password_input) <= 6:
    print ("Password too weak!".upper())
elif len(password_input) <= 10:
    print("password is average!".upper())
else:
    print("strong password!".upper())