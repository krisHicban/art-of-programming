# Fibonacci Naive - O(2^n) - FOARTE LENT!
import time

steps_count = 0

def fibonacci_naive(n):
    global steps_count
    steps_count += 1
    if n <= 1:
        return n
    # print(f"Hi from Reccursive Fibbonaci function with n = {n}")
    # Calculează din nou aceleași valori! ❌
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


# Test cu n=10 (rapid)
start = time.time()
result_10 = fibonacci_naive(10)
time_10 = (time.time() - start) * 1000
# print(f"fib(10) = {result_10}, Time: {time_10:.2f}ms")
# Output: fib(10) = 55, Time: 0.05ms
print("Numar pasi fibonnaci(10): ", steps_count)


steps_count = 0

# Test cu n=30 (lent!)
start = time.time()
result_30 = fibonacci_naive(30)
time_30 = (time.time() - start) * 1000
# print(f"fib(30) = {result_30}, Time: {time_30:.0f}ms")

print("Numar pasi fibonnaci(30): ", steps_count)



# Output: fib(30) = 832040, Time: 300ms
# Pentru n=40: ~5-10 SECUNDE! 🐌
# Crește exponențial - inutil pentru n > 35
