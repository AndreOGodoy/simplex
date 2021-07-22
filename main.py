import sys

from pl import *


if len(sys.argv) == 1:
    print("Erro: especifique um arquivo de entrada")
    sys.exit(1)

INPUT_FILE = sys.argv[1]


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
    result = pl.solve()

    if result.pl_type is PLType.OPTIMAL:
        print('otima')
        print(-result.optimal_value)
        print(('{:g} '*result.solution.size).format(*result.solution))
        print(('{:g} '*result.certificate.size).format(*-result.certificate + 0))

    elif result.pl_type is PLType.INVIABLE:
        print('inviavel')
        print(('{:g} '*result.certificate.size).format(*-result.certificate + 0))

    elif result.pl_type is PLType.UNLIMITED:
        print('ilimitada')
        print(('{:g} '*result.solution.size).format(*result.solution))
        print(('{:g} '*result.certificate.size).format(*result.certificate))


if __name__ == '__main__':
    main()
