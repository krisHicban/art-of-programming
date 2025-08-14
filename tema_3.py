#tema_3
username = input("\nUsername:")
password = input("\nPassword:")
if username == ("krisHicban") and  password == ("qwerty"):
    print ("\nAccess Granted!")
elif username != ("krisHicban") and  password ==("qwerty"):
    print ("\nAccess Denied!")
elif username == ("krisHicban") or not password != ("qwerty"):
    print ("\nUsername/Password Incorect!")
else:
    print ("\nSomething went wrong!")