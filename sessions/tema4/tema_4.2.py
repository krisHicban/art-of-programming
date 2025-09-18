#Shopping and Budget Manager


products = [
    {"product": "rice", "price": 10},
    {"product": "chicken", "price": 9},
    {"product": "sugar", "price": 4},
    {"product": "garlic", "price": 5},
    {"product": "detergent", "price": 53},
    {"product": "toothpaste", "price": 15}
]

# Current basket
basket = []



def show_products():
    print("\n Available Products:")
    for idx, p in enumerate(products, start=1):
        print(f"{idx}. {p['product']} | {p['price']:.2f} lei")

def add_product():
    """Add product by name to basket"""
    global basket
    show_products()
    name = input("Enter product name to add to basket: ")
    for p in products:
        if p["product"].lower() == name.lower():
            basket = basket + [p] 
            print(f" '{name}' added to basket.")
            return
    print(" Product not found.")

def delete_product():
    """Remove product by name from basket"""
    global basket
    if not basket:
        print(" Basket is empty.")
        return
    print("\n Current Basket:")
    for item in basket:
        print(f"- {item['product']} | {item['price']:.2f} lei")
    name = input("Enter product name to delete from basket: ")
# build new basket without that product
    new_basket = [item for item in basket if item["product"].lower() != name.lower()]
    if len(new_basket) < len(basket):
        basket = new_basket
        print(f" '{name}' removed from basket.")
    else:
        print(" Product not found in basket.")

def show_total():
    total = sum(p["price"] for p in basket)
    print(f" Total basket price: {total:.2f} lei")
    return total

def sort_products():
    """Sort available products by price"""
    sorted_list = sorted(products, key=lambda x: x["price"])
    print("\n Products sorted by price:")
    for p in sorted_list:
        print(f"- {p['product']} | {p['price']:.2f} lei")

def filter_products():
    """Show products > 50 lei"""
    filtered = [p for p in products if p["price"] > 50]
    print("\n Products costing more than 50 lei:")
    for p in filtered:
        print(f"- {p['product']} | {p['price']:.2f} lei")
    if not filtered:
        print("No products above 50 lei.")

def show_basket():
    """Display basket contents"""
    if not basket:
        print(" Basket is empty.")
        return
    print("\n Your Basket:")
    for item in basket:
        print(f"- {item['product']} | {item['price']:.2f} lei")
    show_total()

#Budget Algorithm
def budget_check():
    """Check basket against user budget"""
    if not basket:
        print(" Basket is empty.")
        return
    budget = float(input("Enter your budget (lei): "))
    total = show_total()

    if total <= budget:
        print(f" You are within budget! (Budget: {budget:.2f} lei)")
        return

    print(f" Budget exceeded! Total: {total:.2f} lei, Budget: {budget:.2f} lei")
# Suggest removing most expensive items first
    sorted_basket = sorted(basket, key=lambda x: x["price"], reverse=True)
    print("\n Suggestions to reduce cost:")
    running_total = total
    suggestions = []
    for item in sorted_basket:
        running_total -= item["price"]
        suggestions.append(item)
        if running_total <= budget:
            break
    print("Consider removing:")
    for s in suggestions:
        print(f"- {s['product']} | {s['price']:.2f} lei")
# Suggest cheaper alternatives (if available)
    print("\n Cheaper alternatives available in products list:")
    for item in suggestions:
        cheaper = [p for p in products if p["price"] < item["price"]]
        if cheaper:
            cheapest = sorted(cheaper, key=lambda x: x["price"])[0]
            print(f"For {item['product']} ({item['price']} lei), consider {cheapest['product']} ({cheapest['price']} lei)")
        else:
            print(f"No cheaper alternative found for {item['product']}")

# Main Menu
while True:
    print("\nProduct & Budget Manager")
    print("1. Show available products")
    print("2. Add product to basket")
    print("3. Delete product from basket")
    print("4. Show basket & total")
    print("5. Sort products by price")
    print("6. Filter products > 50 lei")
    print("7. Check budget")
    print("0. Exit")

    choice = input("Choose an option (1-7 and 0): ")

    if choice == "1":
        show_products()
    elif choice == "2":
        add_product()
    elif choice == "3":
        delete_product()
    elif choice == "4":
        show_basket()
    elif choice == "5":
        sort_products()
    elif choice == "6":
        filter_products()
    elif choice == "7":
        budget_check()
    elif choice == "0":
        print("\nExiting Product & Budget Manager. Goodbye!")
        break
    else:
        print("\nInvalid choice, try again.")



