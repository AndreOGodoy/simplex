import sys

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
    print('Original: ', pl.tableaux(), sep='\n')

    pl_eq_form = pl.into_equality_form()
    print('FPI: ', pl_eq_form.tableaux(), sep='\n')

    aux_pl = pl_eq_form.get_aux_pl()
    print('Auxiliar: ', aux_pl.tableaux(), sep='\n')

    response = aux_pl.primal_simplex(is_aux_pl=True)
    print(response.pl_type, response.certificate, response.optimal_value, response.solution)

    if response.optimal_value != 0:
        print("A PL Original é inviável")
        sys.exit(0)

    base = np.where(response.solution > 0)[0]
    print('Base: ', base)

    canonical_pl = pl_eq_form.into_canonical(base=base)
    print('Original Canônica: ', canonical_pl.tableaux(), sep='\n')

    response_2 = canonical_pl.primal_simplex(base=base)
    print(response_2.pl_type, response_2.certificate, response_2.optimal_value, response_2.solution)


if __name__ == '__main__':
    main()
