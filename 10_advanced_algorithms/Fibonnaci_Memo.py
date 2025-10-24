# Fibonacci Memoized - O(n) - RAPID!
import time
steps_count = 0


def fibonacci_memo(n, memo={}):
    global steps_count
    steps_count += 1
    # Dacă deja am calculat, returnez din cache ✅
    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    # Calculez o singură dată și salvez în cache
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


# Test cu n=30 (rapid!)
start = time.time()
result_30 = fibonacci_memo(30, {})
time_30 = (time.time() - start) * 1000
print(f"fib_memo(30) = {result_30}, Time: {time_30:.2f}ms")
print("Numar pasi fibonnaci(30): ", steps_count)




# Output: fib_memo(30) = 832040, Time: 0.05ms

# Test cu n=100 (INSTANT chiar și pentru valori mari!)
# start = time.time()
# result_100 = fibonacci_memo(100, {})
# time_100 = (time.time() - start) * 1000
# print(f"fib_memo(100) = {result_100}, Time: {time_100:.2f}ms")
# Output: fib_memo(100) = 354224848179261915075, Time: 0.15ms

# 300ms → 0.05ms = 6000x MAI RAPID! ⚡
