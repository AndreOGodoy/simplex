from pl import *


def read_input() -> PL:
    n, m = map(int, input().split())
    obj_func = np.array(input().split(), dtype=float)

    restr = np.empty(shape=(n, m+1))
    for idx in range(n):
        restr[idx, :] = np.array(input().split(), dtype=float)

    pl = PL(n, m, obj_func, restr)

    return pl


def main():
    pl = read_input()
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
