import datetime


friends = [
    {"name": "Andrew", "phone": "+40 721 123 456", "age": 26, "last_contact": "10-09-2025", "hobbies": ["hiking", "reading", "gaming"], "distance": 5, "connection": 4},
    {"name": "John", "phone": "+40 733 987 654", "age": 24, "last_contact": "05-09-2025", "hobbies": ["cooking", "reading", "painting"], "distance": 3, "connection": 5},
    {"name": "Michael", "phone": "+40 722 555 888", "age": 28, "last_contact": "01-09-2025", "hobbies": ["hiking", "cycling", "gaming"], "distance": 12, "connection": 3},
    {"name": "Elen", "phone": "+40 724 111 222", "age": 27, "last_contact": "15-08-2025", "hobbies": ["reading", "painting", "yoga"], "distance": 8, "connection": 4},
    {"name": "Kris", "phone": "+40 731 444 999", "age": 25, "last_contact": "20-08-2025", "hobbies": ["gaming", "cycling", "hiking"], "distance": 10, "connection": 3}
]

#Utility Functions
# def friend_list():
#     list = "name"
#     print (list)
def add_friend():
    name = input("Name: ")
    phone = input("Phone: ")
    age = int(input("Age: "))
    last_contact = input("Last contact (DD-MM-YYYY): ")
    hobbies = [h.strip() for h in input("Hobbies (comma separated): ").split(",")]
    distance = float(input("Distance (km): "))
    connection = int(input("Connection (1-5): "))
    friends.append({
        "name": name,
        "phone": phone,
        "age": age,
        "last_contact": last_contact,
        "hobbies": hobbies,
        "distance": distance,
        "connection": connection
    })
    print("Friend added!")

def delete_friend():
    name = input("Enter the name to delete: ")
    global friends
    friends = [f for f in friends if f['name'].lower() != name.lower()]
    print("Friend deleted if existed.")

def modify_friend():
    name = input("Name of friend to modify: ")
    for f in friends:
        if f['name'].lower() == name.lower():
            field = input("Which field to modify? (phone, age, last_contact, hobbies, distance, connection): ")
            if field == "hobbies":
                f['hobbies'] = [h.strip() for h in input("New hobbies (comma separated): ").split(",")]
            elif field == "age":
                f['age'] = int(input("New age: "))
            elif field == "distance":
                f['distance'] = float(input("New distance: "))
            elif field == "connection":
                f['connection'] = int(input("New connection (1-5): "))
            else:
                f[field] = input(f"New {field}: ")
            print("Friend updated!")
            return
    print("Friend not found.")

def display_unique_hobbies():
    hobbies = set()
    for f in friends:
        hobbies.update(f['hobbies'])
    print("Unique hobbies:", hobbies)

# --- Recommendation Algorithm ---
def recommend_friends_for_hobby():
    hobby = input("Enter a hobby: ").strip()
    recommendations = []

    for f in friends:
        score = 0
        if hobby in f['hobbies']:
            score += 10  # common hobby

# Recent contact (parse DD-MM-YYYY)
        last_contact_date = datetime.datetime.strptime(f['last_contact'], "%d-%m-%Y").date()
        days_since = (datetime.date.today() - last_contact_date).days
        if days_since < 14:
            score += 5

# Distance penalty
        score -= f['distance'] * 0.2

# Connection bonus
        score += f['connection']

        recommendations.append((f['name'], score, f))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    print("\nRecommended friends for hobby", hobby)
    for name, score, f in recommendations:
        print(f"{name} (Score: {score:.1f}, Distance: {f['distance']} km, Connection: {f['connection']})")

# Interaction simulation
    go_out = input("\nDo you want to go out with the top recommendation? (y/n): ")
    if go_out.lower() == 'y' and recommendations:
        top_friend = recommendations[0][2]
        top_friend['last_contact'] = datetime.date.today().strftime("%d-%m-%Y")
        top_friend['connection'] = min(top_friend['connection'] + 1, 5)
        print(f"Updated last_contact & connection for {top_friend['name']}.")

# --- Main Menu ---
while True:
    print("\nSmart Activity Assistant")
    print("1. Add friend")
    print("2. Delete friend")
    print("3. Modify friend")
    print("4. Display unique hobbies")
    print("5. Recommend friends for a hobby")
    print("0. Exit")

    choice = input("\nChoose an option: ")
    if choice == '1':
       add_friend()
    elif choice == '2':
        delete_friend()
    elif choice == '3':
        modify_friend()
    elif choice == '4':
        display_unique_hobbies()
    elif choice == '5':
        recommend_friends_for_hobby()
    elif choice == '0':
        print("\nGoodbye!")
        break
    else:
        print("\nInvalid choice.")
