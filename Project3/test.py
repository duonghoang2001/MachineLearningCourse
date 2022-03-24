import numpy as np

def main():
    m1 = np.matrix([[2, 1, 3], [3, 1, 4]])
    m2 = np.matrix([[1, 2, 1], [2, 1, 2]])
    #m3 = list(map(lambda x, y: x * y, m1, m2))
    #print(m3)
    for row in m1:
        print(row)

    for i in range(len(m1)):
        print(np.array(m1[i]))
main()