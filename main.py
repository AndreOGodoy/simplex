import numpy as np
from pl import *


INPUT_FILE = 'input.txt'


def read_input(input_file: str) -> PL:
    with open(input_file) as f:
        line_0 = f.readline()
        n, m = map(int, line_0.split())

        line_1 = f.readline()
        obj_func = np.array(line_1.split(), dtype=float)

        # Matriz de restrições
        restr = np.empty(shape=(n, m+1))
        for idx, line in enumerate(f.readlines()):
            restr[idx, :] = line.replace('\n', '').split()

        pl = PL(n, m, obj_func, restr)

    return pl


def main():
    pl = read_input(INPUT_FILE)
    print(pl)

    print(pl.get_tableaux())

    #pl_eq_form = pl.into_equality_form()
    #print(pl_eq_form)

if __name__ == '__main__':
    main()
