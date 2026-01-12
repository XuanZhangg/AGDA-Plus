p = 0.08

def c(k):
    return 1 if k<=7 else 2.6

t = [(1-p)**(k-1)*p for k in range(1,100)]
v = sum(t)
print(t)
print(f"v={v}")
