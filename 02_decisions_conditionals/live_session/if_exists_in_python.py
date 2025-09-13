# Truthy values
print("=== Truthy values ===")
for value in [True, "ploua", 1, -5, 3.14, [1], {"a": 1}]:
    if value:
        print(f"{repr(value)} este considerat ADEVARAT")

# Falsy values
print("\n=== Falsy values ===")
for value in [False, None, 0, "", [], {}]:
    if value:
        print(f"{repr(value)} este considerat ADEVARAT")
    else:
        print(f"{repr(value)} este considerat FALS")
