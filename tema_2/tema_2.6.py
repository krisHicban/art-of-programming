
seconds=int(input("seconds "))
print("The total hour is:\n",seconds//(60*60), ":", (seconds%(60*60))//60 ,":", (((seconds%(60*60)))%3600)%60)