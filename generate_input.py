import sys
import random


if __name__ == '__main__':
    n, m = -1, -1
    if len(sys.argv) == 1:
        n = random.randint(1, 100)
        m = random.randint(1, 100)
    else:
        n = int(sys.argv[1])
        m = int(sys.argv[1])

    C = random.choices(range(-100, 101), k=m)
    A = []
    for _ in range(n):
        line = random.choices(range(-100, 101), k=m+1)
        A.append(line)

    print(n, m)
    print(*C)
    for line in A:
        print(*line)
