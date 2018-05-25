import sys
from math import log

file_1 = sys.argv[1]
file_2 = sys.argv[2]

f_1 = open(file_1, 'r+').read().splitlines()

r_1 = ""
r_2 = ""
n_1 = 0
n_2 = 0
result = []

for i in range(1, len(f_1)):
    m = f_1[i].split(',')
    result.append(m[-1])

for i in range(1, len(result)):
    if result[i] != result[0]:
        r_1 = result[0]
        r_2 = result[i]

for i in range(0, len(result)):
    if result[i] == r_1:
        n_1 += 1
    else:
        n_2 += 1


def log_eq(x):
    return x * log(x, 2)


def div(x, y):
    return float(x)/(x+y)


def entropy(a, b):
    c = div(a, b)
    d = 1 - c
    en = - log_eq(c) - log_eq(d)
    return en


def error(a, b):
    if a < b:
        return div(a, b)
    elif a > b:
        return 1 - div(a, b)
    else:
        return None


def result_write(n1, n2):
    result_line = "entropy: " + str(entropy(n1, n2)) + '\n' + "error: " + str(error(n1, n2))
    f_2 = open(file_2, 'w+')
    f_2.write(result_line)
    f_2.close()

result_write(n_1, n_2)
