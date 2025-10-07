
# item_list = [{"item": "beef", "quantity": 200, "unit": "grams"},
#                {"item": "onion", "quantity": 30, "unit": "grams"},
#                {"item": "carrot", "quantity": 50, "unit": "grams"},
#                {"item": "asparagus", "quantity": 50, "unit": "grams"},
#                {"item": "salt", "quantity": 2, "unit": "grams"},
#                {"item": "pepper", "quantity": 2, "unit": "grams"},
#                {"item": "olive oil", "quantity": 10, "unit": "ml"}
# ]

# def display_list(): #display the list
#     for k in item_list:
#         print(f"\n{k['quantity']} {k['unit']} of {k['item']}")
    
# def add_item(): #add some item to the list
#     item = input("Item name: ")
#     quantity = float(input("Quantity: "))
#     unit = input("Unit: ")
#     item_list.append({"item": item, 
#                       "quantity": quantity, 
#                       "unit": unit})
#     print(f"Added {item} to the list.")

# def remove_item(): #remove some item from the list
#     item = input("Item name to remove: ")
#     item_list[:]= [k for k in item_list 
#                    if k['item'].lower() != item.lower()]
#     print(f"Removed {item} from the list if it existed.")

# def modify_item(): #modify some item from the list
#     item = input("Item name to modify: ")
#     for k in item_list:
#         if k['item'].lower() == item.lower():
#             quantity = float(input("New Quantity: "))
#             unit = input("New Unit: ")
#             k['quantity'] = quantity
#             k['unit'] = unit
#             print(f"Modified {item}.")
#             return
#     print(f"{item} not found in the list.")

# list = [] #make a recipe
# def make_recipe():
#     print("\nEnter items for your recipe (type 'done' when finished):")
#     while True:
#         item = input("Item name: ")
#         if item.lower() == 'done':
#             break
#         quantity = float(input("Quantity: "))
#         unit = input("Unit: ")
#         list.append({"item": item, 
#                       "quantity": quantity, 
#                       "unit": unit})
#     print("\nYour recipe items:")
#     for k in list:
#         print(f"\n{k['quantity']} {k['unit']} of {k['item']}")
# # Sort items
# def sort_items(by):
#     if by == "item":
#         return sorted(item_list, key=lambda x: x['item'].lower())
#     elif by == "unit":
#         return sorted(item_list, key=lambda x: x['unit'].lower())
#     elif by == "quantity":
#         return sorted(item_list, key=lambda x: x['quantity'])
#     else:
#         return item_list

  
# def main_menu():
#     while True:#menu
#         print("\nRecipe Ingredient Manager")
#         print("1. Display Item")
#         print("2. Add Item")
#         print("3. Remove Item")
#         print("4. Modify Items")
#         print("5. Make your Recipe list!")
#         print("6. Sort Items")
#         print("0. Exit")

#         choice = input("Choose an option: ")
#         if choice == '1':
#             display_list()
#         elif choice == '2':
#             add_item()
#         elif choice == '3':
#             remove_item()
#         elif choice == '4':
#             modify_item()
#         elif choice == '5':
#             make_recipe()
#         elif choice == '6':
#             sort_by = input("Sort by (item/unit/quantity): ").strip().lower()
#             sorted_items = sort_items(sort_by)
#             print("\nSorted Items:")
#             for i in sorted_items:
#                 print(f"\n{i['quantity']} {i['unit']} of {i['item']}")
#         elif choice == '0':
#             print("\nGoodbye and Enjoy the Recipe!")
#             break
#         else:
#             print("\nInvalid choice. Please try again.")
# #this is where main script starts

#
#text analysis
def analyze_text(text):
    
    words = text.split()# Split text into words
    num_words = len(words) # Number of words
    longest_word_draft = max(words, key=len) if words else ""# Longest word (if text is empty, return empty string)
    longest_word = []
    for w in words:
        if len (w) == len(longest_word_draft):
            longest_word.append(w)
    contains_python = "python" in text.lower()# Check if "python" occurs (case-insensitive)
    return num_words, longest_word, contains_python

# main program
user_text = input("Enter your text: ")
num_words, longest_word, contains_python = analyze_text(user_text)

print("\nNumber of words:", num_words)
print("\nLongest word:", longest_word)
print("\nContains 'python':", contains_python)
