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
        print(round(-result.optimal_value, 7))
        print(('{:.7f} '*result.solution.size).format(*result.solution))
        print(('{:.7f} '*result.certificate.size).format(*np.abs(result.certificate)))

    elif result.pl_type is PLType.INVIABLE:
        print('inviavel')
        print(('{:.7f} '*result.certificate.size).format(*np.abs(result.certificate)))

    elif result.pl_type is PLType.UNLIMITED:
        print('ilimitada')
        print(('{:.7f} '*result.solution.size).format(*result.solution))
        print(('{:.7f} '*result.certificate.size).format(*np.abs(result.certificate)))


if __name__ == '__main__':
    main()
