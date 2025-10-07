class Backyard:
    def __init__(self, width, length):
        self.width = width
        self.length = length
                    #"-"is equivalent to 1m²
        self.grid = [["-" for _ in range(width)] for _ in range(length)]
        self.total_space = width * length
        self.used_space = 0
    
    # I want to upgrade further later on
    # class Feature:
         #def __init__(self, name, width, length):        
            # self.name = name
            # self.width = width
            # self.length = length
        

        #Display the backyard grid
    def display(self):
        print("\nBackyard Layout:")
        for row in self.grid:
            print(" ".join(row))
        print(f"Total area: {self.total_space} m²\n")

    def place_feature(self, name, symbol, start_row, start_col, w, l):
        
        #Place a feature of width w and length l at start_row, start_col.
        if start_row + l > self.length or start_col + w > self.width:
            print(f"{name} doesn't fit at the specified location.")
            return False
        
        # Check if space is free 
        for r in range(start_row, start_row + l):
            for c in range(start_col, start_col + w):
                if self.grid[r][c] != "-":
                    print(f"{name} overlaps with another feature.")
                    return False
        
        # Place feature
        for r in range(start_row, start_row + l):
            for c in range(start_col, start_col + w):
                self.grid[r][c] = symbol
        added_space = w * l
        self.used_space += added_space
        remaining_space = self.total_space - self.used_space
        print(f"\n{name} placed successfully.")
        print(f"Used space: {self.used_space} m², Remaining space: {remaining_space} m²")
        return True
        # Remove feature
    
    def remove_feature(self, symbol,name):
        for r in range(self.length):
            for c in range(self.width):
                if self.grid[r][c] == symbol:
                    self.grid[r][c] = "-"
        print(f"\n {name} removed.")

# Main program

def maximize_backyard():
    print ("\nWelcome to Backyard Manager!")
    width = int(input("\nEnter backyard width (m): "))
    length = int(input("Enter backyard height (m): "))

    backyard = Backyard(width, length)
    backyard.display()

    while True:
        
        print("\nOptions: (p)lace feature, (r)emove feature, (q)uit")
        choice = input("Choose action: ").lower()
        if choice == "p":
            name = input("Feature name (e.g., Terrace, Doghouse, etc.): ")
            symbol = input("Symbol to represent it (e.g., T, D, etc.): ")
            start_row = int(input("Start row: "))
            start_col = int(input("Start column: "))
            w = int(input("Feature width (m): "))
            l = int(input("Feature length (m): "))
            backyard.place_feature(name, symbol, start_row, start_col, w, l)
            backyard.display()
        
        elif choice == "r":
            symbol = input("Symbol of feature to remove: ")
            backyard.remove_feature(symbol,name)
            backyard.display()
            
                
        elif choice == "q":
            print("Exiting program.")
            break

        else:
            print("Invalid choice.")

maximize_backyard()
