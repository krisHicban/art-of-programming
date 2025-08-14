#tema_3
username = input("\nUsername:")
password = input("\nPassword:")
if username == ("krisHicban") and  password == ("qwerty"):
    print ("\nAccess Granted!")
elif username != ("krisHicban") and  password ==("qwerty"):
    print ("\nUsername/Password Incorrect!")
elif username == ("krisHicban") or not password != ("qwerty"):
    print ("\nUsername/Password Incorect!")
elif username != ("krisHicban") or not password != ("qwerty"):
    print ("\nAccess Denied")  
else:
    print ("\nSomething went wrong!")