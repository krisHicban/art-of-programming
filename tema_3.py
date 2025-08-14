#tema_3
username_correct = "krisHicban"
password_correct = "qwerty"
username = input("\nUsername:")
password = input("\nPassword:")
if username == username_correct and  password == password_correct:
    print ("\nAccess Granted!")
elif username != username_correct and password != password_correct:
    print ("\nAccess Denied!")
else:
    print ("\nUsername/Password Incorect!")

