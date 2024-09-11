letters = ["a", "b", "c", "d", "e", "f"]
numbers = [0, 1, 2, 3, 4, 5]

for l, n in zip(letters, numbers):
    print(f"{l}, {n}")

l, n = zip(*sorted(zip(letters, numbers), key=lambda x: x[1], reverse=True))
print(l)
print(n)
