def draw_circle  (radius=5):
    for y in range(-radius, radius + 1):
        for x in range(-2*radius, 2*radius + 1):
            if x**2 + (2*y)**2 <= radius**2 * 4:  # equation of a circle
                print("*", end="")
            else:
                print(" ", end="")
        print()


def draw_triangle(height=5):
    for i in range(1, height + 1):
        print(" " * (height - i) + "*" * (2 * i - 1))


def draw_hexagon(size=5):
    # top part
    for i in range(size):
        print("   " * (size - i) + " * " * (size + 2 * i))
    # bottom part
    for i in range(size - 2, -1, -1):
        print("   " * (size - i) + " * " * (size + 2 * i))


# Main program
print("Choose a shape to draw: circle, triangle, hexagon")
choice = input("Enter your choice: ").strip().lower()

if choice == "circle":
    draw_circle()
elif choice == "triangle":
    draw_triangle()
elif choice == "hexagon":
    draw_hexagon()
else:
    print("Invalid choice. Please enter circle, triangle, or hexagon.")
