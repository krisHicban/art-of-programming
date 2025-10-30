# Bubble Sort - Algoritm educațional O(n²)
def bubble_sort(comenzi, key):
    comenzi = comenzi.copy()
    n = len(comenzi)
    for i in range(n):
        for j in range(0, n-i-1):
            if comenzi[j][key] > comenzi[j+1][key]:
                comenzi[j], comenzi[j+1] = comenzi[j+1], comenzi[j]
    return comenzi

# Exemplu real
orders = [
    {"id": 1, "timp": 25}, {"id": 2, "timp": 15},
    {"id": 3, "timp": 30}, {"id": 4, "timp": 12}
]
print("Nesortate:", [o["timp"] for o in orders])
# Output: Nesortate: [25, 15, 30, 12]

sorted_orders = bubble_sort(orders, "timp")
print("Sortate:", [o["timp"] for o in sorted_orders])
# Output: Sortate: [12, 15, 25, 30]..