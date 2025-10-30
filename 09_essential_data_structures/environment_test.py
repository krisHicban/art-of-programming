import time
import random
import string

# Generate 1,000,000 random products
def random_name(length=6):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

products = {random_name(): random.uniform(0.5, 100.0) for _ in range(1_000_000)}

# Measure sorting time
start_time = time.time()

sorted_products = dict(sorted(products.items(), key=lambda x: x[1]))

end_time = time.time()
elapsed = end_time - start_time

print(f"✅ Sorted {len(products):,} products by price.")
print(f"⏱️ Time taken: {elapsed:.4f} seconds")

# Optional sanity check — show the first 5 results
for i, (name, price) in enumerate(sorted_products.items()):
    print(f"{name}: {price:.2f}")
    if i == 4:
        break
