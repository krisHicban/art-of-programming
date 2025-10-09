# Design Patterns in Python: Factory, Singleton, Builder

# 1. FACTORY PATTERN - Vehicle Manufacturing System
class Vehicle:
    """Base vehicle class"""
    def __init__(self, make, model):
        self.make = make
        self.model = model
    
    def start_engine(self):
        return f"{self.make} {self.model} engine started!"
    
    def get_info(self):
        return f"{self.__class__.__name__}: {self.make} {self.model}"

class Car(Vehicle):
    """Car implementation"""
    def __init__(self, make, model, doors=4):
        super().__init__(make, model)
        self.doors = doors
        self.vehicle_type = "Car"
    
    def open_trunk(self):
        return f"{self.make} {self.model} trunk opened"

class Motorcycle(Vehicle):
    """Motorcycle implementation"""
    def __init__(self, make, model, engine_size):
        super().__init__(make, model)
        self.engine_size = engine_size
        self.vehicle_type = "Motorcycle"
    
    def wheelie(self):
        return f"{self.make} {self.model} doing a wheelie!"

class Truck(Vehicle):
    """Truck implementation"""
    def __init__(self, make, model, payload_capacity):
        super().__init__(make, model)
        self.payload_capacity = payload_capacity
        self.vehicle_type = "Truck"
    
    def load_cargo(self, weight):
        if weight <= self.payload_capacity:
            return f"Loaded {weight}kg cargo into {self.make} {self.model}"
        return f"Cannot load {weight}kg - exceeds capacity of {self.payload_capacity}kg"

class VehicleFactory:
    """Factory class to create different types of vehicles"""
    
    @staticmethod
    def create_vehicle(vehicle_type, make, model, **kwargs):
        """Create vehicle based on type"""
        vehicle_types = {
            "car": Car,
            "motorcycle": Motorcycle,
            "truck": Truck
        }
        
        vehicle_class = vehicle_types.get(vehicle_type.lower())
        if vehicle_class:
            return vehicle_class(make, model, **kwargs)
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")

# Factory Pattern Demo
print("=== FACTORY PATTERN DEMO ===")

# Create different vehicles using factory
vehicles = [
    VehicleFactory.create_vehicle("car", "Toyota", "Camry", doors=4),
    VehicleFactory.create_vehicle("motorcycle", "Harley", "Davidson", engine_size=1200),
    VehicleFactory.create_vehicle("truck", "Ford", "F-150", payload_capacity=1000)
]

for vehicle in vehicles:
    print(vehicle.get_info())
    print(vehicle.start_engine())

# Test specific methods
print(vehicles[0].open_trunk())  # Car method
print(vehicles[1].wheelie())     # Motorcycle method
print(vehicles[2].load_cargo(800))  # Truck method

# 2. SINGLETON PATTERN - Database Connection Manager
class DatabaseConnection:
    """Singleton class for database connection management"""
    
    _instance = None
    _connection = None
    
    def __new__(cls):
        """Ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize connection if not already done"""
        if self._connection is None:
            self._connection = self._create_connection()
    
    def _create_connection(self):
        """Simulate database connection creation"""
        import random
        connection_id = f"DB_CONN_{random.randint(1000, 9999)}"
        print(f"Creating new database connection: {connection_id}")
        return {
            "connection_id": connection_id,
            "host": "localhost",
            "database": "company_db",
            "status": "connected"
        }
    
    def execute_query(self, query):
        """Execute a database query"""
        if self._connection["status"] == "connected":
            return f"Executing query on {self._connection['connection_id']}: {query}"
        return "Database not connected"
    
    def get_connection_info(self):
        """Get connection information"""
        return self._connection
    
    def close_connection(self):
        """Close database connection"""
        if self._connection:
            connection_id = self._connection["connection_id"]
            self._connection["status"] = "disconnected"
            return f"Connection {connection_id} closed"
        return "No connection to close"

# Singleton Pattern Demo
print("\n=== SINGLETON PATTERN DEMO ===")

# Create multiple "instances" - should all be the same object
db1 = DatabaseConnection()
db2 = DatabaseConnection()
db3 = DatabaseConnection()

print(f"db1 id: {id(db1)}")
print(f"db2 id: {id(db2)}")
print(f"db3 id: {id(db3)}")
print(f"All instances are the same object: {db1 is db2 is db3}")

# All instances share the same connection
print(db1.execute_query("SELECT * FROM users"))
print(db2.execute_query("SELECT * FROM products"))
print(f"Connection info: {db3.get_connection_info()}")

# 3. BUILDER PATTERN - Computer Configuration System
class Computer:
    """Computer class to be built"""
    def __init__(self):
        self.cpu = None
        self.ram = None
        self.storage = None
        self.gpu = None
        self.motherboard = None
        self.power_supply = None
        self.case = None
        self.price = 0
    
    def get_specs(self):
        """Get computer specifications"""
        specs = "\n=== Computer Specifications ===\n"
        specs += f"CPU: {self.cpu}\n"
        specs += f"RAM: {self.ram}\n"
        specs += f"Storage: {self.storage}\n"
        specs += f"GPU: {self.gpu}\n"
        specs += f"Motherboard: {self.motherboard}\n"
        specs += f"Power Supply: {self.power_supply}\n"
        specs += f"Case: {self.case}\n"
        specs += f"Total Price: ${self.price:,}\n"
        return specs

class ComputerBuilder:
    """Builder class for creating computers"""
    
    def __init__(self):
        self.computer = Computer()
    
    def set_cpu(self, cpu, price):
        """Set CPU component"""
        self.computer.cpu = cpu
        self.computer.price += price
        return self
    
    def set_ram(self, ram, price):
        """Set RAM component"""
        self.computer.ram = ram
        self.computer.price += price
        return self
    
    def set_storage(self, storage, price):
        """Set storage component"""
        self.computer.storage = storage
        self.computer.price += price
        return self
    
    def set_gpu(self, gpu, price):
        """Set GPU component"""
        self.computer.gpu = gpu
        self.computer.price += price
        return self
    
    def set_motherboard(self, motherboard, price):
        """Set motherboard component"""
        self.computer.motherboard = motherboard
        self.computer.price += price
        return self
    
    def set_power_supply(self, power_supply, price):
        """Set power supply component"""
        self.computer.power_supply = power_supply
        self.computer.price += price
        return self
    
    def set_case(self, case, price):
        """Set case component"""
        self.computer.case = case
        self.computer.price += price
        return self
    
    def build(self):
        """Build and return the computer"""
        # Validate that essential components are present
        essential_components = [self.computer.cpu, self.computer.ram, 
                              self.computer.storage, self.computer.motherboard]
        
        if not all(essential_components):
            raise ValueError("Missing essential components (CPU, RAM, Storage, Motherboard)")
        
        return self.computer

class ComputerDirector:
    """Director class with predefined computer configurations"""
    
    @staticmethod
    def build_gaming_pc():
        """Build a high-end gaming computer"""
        return (ComputerBuilder()
                .set_cpu("Intel i9-13900K", 600)
                .set_ram("32GB DDR5-5600", 400)
                .set_storage("1TB NVMe SSD", 150)
                .set_gpu("RTX 4080 16GB", 1200)
                .set_motherboard("ASUS ROG Z790", 300)
                .set_power_supply("850W 80+ Gold", 150)
                .set_case("NZXT H7 RGB", 200)
                .build())
    
    @staticmethod
    def build_office_pc():
        """Build a basic office computer"""
        return (ComputerBuilder()
                .set_cpu("Intel i5-13400", 200)
                .set_ram("16GB DDR4-3200", 100)
                .set_storage("512GB SSD", 80)
                .set_gpu("Integrated Graphics", 0)
                .set_motherboard("MSI B660M", 100)
                .set_power_supply("450W 80+ Bronze", 80)
                .set_case("Fractal Core 1000", 60)
                .build())
    
    @staticmethod
    def build_workstation():
        """Build a professional workstation"""
        return (ComputerBuilder()
                .set_cpu("AMD Ryzen 9 7950X", 700)
                .set_ram("64GB DDR5-5200", 800)
                .set_storage("2TB NVMe SSD", 300)
                .set_gpu("RTX A6000 48GB", 4500)
                .set_motherboard("ASUS Creator X670E", 500)
                .set_power_supply("1000W 80+ Platinum", 250)
                .set_case("Fractal Define 7 XL", 250)
                .build())

# Builder Pattern Demo
print("\n=== BUILDER PATTERN DEMO ===")

# Build predefined configurations
gaming_pc = ComputerDirector.build_gaming_pc()
office_pc = ComputerDirector.build_office_pc()
workstation = ComputerDirector.build_workstation()

print("GAMING PC:")
print(gaming_pc.get_specs())

print("OFFICE PC:")
print(office_pc.get_specs())

print("WORKSTATION:")
print(workstation.get_specs())

# Build custom configuration
print("CUSTOM BUILD:")
custom_pc = (ComputerBuilder()
             .set_cpu("AMD Ryzen 7 7700X", 400)
             .set_ram("32GB DDR5-5200", 350)
             .set_storage("1TB NVMe SSD", 150)
             .set_gpu("RTX 4070 12GB", 800)
             .set_motherboard("MSI X670E", 250)
             .set_power_supply("750W 80+ Gold", 120)
             .set_case("Lian Li O11 Dynamic", 180)
             .build())

print(custom_pc.get_specs())

# Demonstrate pattern benefits
print("\n=== DESIGN PATTERNS SUMMARY ===")
print("✓ Factory Pattern: Creates objects without specifying exact classes")
print("✓ Singleton Pattern: Ensures only one instance of a class exists")
print("✓ Builder Pattern: Constructs complex objects step by step")
print("\nThese patterns solve common programming problems and make code more:")
print("• Maintainable • Flexible • Reusable • Testable")