from enum import Enum, auto
import numpy as np
from typing import Optional, Tuple


class PLType(Enum):
    OPTIMAL = auto(),
    INVIABLE = auto(),
    UNLIMITED = auto()


class SimplexReturn:
    def __init__(self, pl_type: PLType,
                 certificate: np.ndarray,
                 optimal_value: Optional[float] = None,
                 solution: Optional[np.ndarray] = None,
                 base: Optional[np.ndarray] = None):
        self.pl_type = pl_type
        self.certificate = certificate
        self.optimal_value = optimal_value
        self.solution = solution
        self.base = base


class RestrType(Enum):
    LESS_EQ = auto(),
    EQ = auto()


class ObjFuncType(Enum):
    MAX = auto(),
    MIN = auto(),


def idx_first(cond_arr: np.ndarray) -> Optional[int]:
    indexes = np.where(cond_arr)
    return indexes[0][0] if indexes[0].size > 0 else None


def get_simplex_primal_ratio(a: np.ndarray, b: np.ndarray):
    divided = np.array([a_v / b_v if b_v > 0 else np.inf for a_v, b_v in zip(a, b)])
    return divided


def get_basic_solution(restr: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    identity = np.identity(restr.shape[0])
    solution = np.zeros(restr.shape[1])

    indexes = []
    order = []

    for col_idx, col in enumerate(restr.T):
        result = np.where((identity == col).all(axis=1))[0]
        if result.size != 0:
            solution[col_idx] = b[result[0]]

            indexes.append(col_idx)
            order.append(result[0])
        else:
            solution[col_idx] = 0

    indexes = np.array(indexes)
    order = np.array(order)
    base = indexes[order]
    return solution, np.unique(base)


class PL:
    def __init__(self, n_rest: int, n_var: int,
                 obj_func: np.ndarray, restr: np.ndarray,
                 restr_type: RestrType = RestrType.LESS_EQ,
                 obj_func_type: ObjFuncType = ObjFuncType.MAX):
        self.n_var = n_var
        self.n_rest = n_rest

        self.obj_func = obj_func
        self.obj_func_type = obj_func_type

        self.restr = restr
        self.restr_type = restr_type

        self.op_reg = np.identity(n_rest, dtype=float)
        self.op_reg_c = np.zeros(n_rest, dtype=float)

        self.optimal_value = 0.0

    def __str__(self) -> str:
        obj_func_type_str = 'MAX' if self.obj_func_type is ObjFuncType.MAX else 'MIN'
        restr_type_str = '<=' if self.restr_type is RestrType.LESS_EQ else '='

        output_str = f'{obj_func_type_str} {self.obj_func}x o.v.: {self.optimal_value}\n'
        output_str += 's.t. \n'
        for line in self.restr:
            output_str += f'{line[:-1]} {restr_type_str} {line[-1]}\n'
        output_str += 'x >= 0\n'
        return output_str

    def tableaux(self) -> str:
        output_str = ''
        for idx, c in enumerate(np.hstack((self.op_reg_c, self.obj_func))):
            if idx == self.n_rest:
                output_str += ' | '

            c = c * -1 + 0
            output_str += f'{c:>5.2f} '

        output_str += f' | {str(self.optimal_value) : >5}\n'
        output_str += '----------' * self.n_var + '\n'

        for i in range(self.n_rest):
            for j in range(self.n_var + 1 + self.n_rest):
                if j == self.n_rest or j == self.n_var + self.n_rest:
                    output_str += ' | '
                if j >= self.n_rest:
                    output_str += f'{self.restr[i, j - self.n_rest]:>5.2f} '
                else:
                    output_str += f'{self.op_reg[i, j]:>5.2f} '

            output_str += '\n'

        return output_str

    def into_equality_form(self, inplace: bool = False) -> Optional['PL']:
        if self.restr_type is RestrType.EQ and inplace:
            return
        elif self.restr_type is RestrType.EQ and not inplace:
            return PL(self.n_rest, self.n_var, self.obj_func.copy(), self.restr.copy(),
                      self.restr_type, self.obj_func_type)

        new_obj_func = np.append(self.obj_func, [0] * self.n_rest)

        if self.obj_func_type is ObjFuncType.MIN:
            new_obj_func *= -1

        new_restr = np.hstack((self.restr, np.zeros((self.n_rest, self.n_rest))))
        new_restr[:, -1] = new_restr[:, self.n_var]
        new_restr[:, self.n_var: -1] = np.identity(self.n_rest)

        new_n_var = self.n_var + self.n_rest

        if inplace:
            self.restr = new_restr
            self.n_var = new_n_var
            self.obj_func = new_obj_func
            self.restr_type = RestrType.EQ
            self.obj_func_type = ObjFuncType.MAX
            return

        new_pl = PL(self.n_rest, new_n_var, new_obj_func, new_restr, RestrType.EQ, ObjFuncType.MAX)
        return new_pl

    def pivot_self_by(self, row_idx: int, col_idx: int):
        self.restr[row_idx, :] = self.restr[row_idx, :] / self.restr[row_idx, col_idx]
        pivot = self.restr[row_idx, col_idx]

        for idx, row in enumerate(self.restr):
            if idx == row_idx:
                continue

            ratio = row[col_idx] / pivot
            self.op_reg[idx] -= ratio * self.op_reg[row_idx, :]
            self.restr[idx] -= ratio * self.restr[row_idx, :]

        ratio = self.obj_func[col_idx] / pivot
        self.obj_func -= ratio * self.restr[row_idx, :-1]
        self.op_reg_c -= ratio * self.op_reg[row_idx]
        self.optimal_value -= ratio * self.restr[row_idx, -1]

        self.restr = self.restr + 0
        self.obj_func = self.obj_func + 0

    def primal_simplex(self, base: Optional[np.ndarray] = None, is_aux_pl: bool = False) -> SimplexReturn:
        canonical = self.into_canonical(base=base, is_aux_pl=is_aux_pl)

        while True:
            possible_columns = np.where(canonical.obj_func > 0)[0]
            if possible_columns.size == 0:
                solution, base = get_basic_solution(canonical.restr[:, :-1], canonical.restr[:, -1])
                return SimplexReturn(pl_type=PLType.OPTIMAL,
                                     certificate=canonical.op_reg_c,
                                     optimal_value=canonical.optimal_value,
                                     solution=solution,
                                     base=base)

            column = possible_columns[0]
            ratios = get_simplex_primal_ratio(canonical.restr[:, -1], canonical.restr[:, column])
            if np.all(ratios == np.inf):
                return SimplexReturn(pl_type=PLType.UNLIMITED,
                                     certificate=np.array([0, 0, 0]))

            min_ratio_idx = np.where(ratios == np.min(ratios))[0][0]
            line = min_ratio_idx

            canonical.pivot_self_by(line, column)

    def into_canonical(self, base: Optional[np.ndarray] = None, inplace: bool = False, is_aux_pl: bool = False):
        if base is not None and is_aux_pl:
            raise ValueError("Base inicial fornecida para PL Auxiliar")

        elif base is not None and base.ndim > 1:
            raise ValueError("'base' deve ser array unidimensional")

        if is_aux_pl:
            canonical = self
            if not inplace:
                canonical = PL(self.n_rest, self.n_var, self.obj_func.copy(),
                               self.restr.copy(), self.restr_type, self.obj_func_type)

            for i in range(self.n_rest):
                canonical.pivot_self_by(i, self.n_var - self.n_rest + i)

            if not inplace:
                return canonical
            return

        canonical = self
        if not inplace:
            canonical = PL(self.n_rest, self.n_var, self.obj_func.copy(), self.restr.copy(), self.restr_type,
                           self.obj_func_type)

        assert base is not None
        for line_idx, base_idx in enumerate(base):
            canonical.pivot_self_by(line_idx, base_idx)

        if not inplace:
            return canonical
        return

    def get_aux_pl(self) -> 'PL':
        if self.restr_type is not RestrType.EQ:
            raise ValueError("A PL deve estar em forma de igualdades antes de se obter sua auxiliar")

        n_ones = self.n_rest

        obj_func = np.zeros(self.obj_func.shape[0] + n_ones)
        obj_func[-n_ones:] = -1

        restr = self.restr

        for line_idx, b in enumerate(restr[:, -1]):
            if b < 0:
                restr[line_idx] *= -1

        restr = np.hstack((restr, np.zeros((n_ones, n_ones))))
        restr[:, -1] = restr[:, self.n_var]
        restr[:, self.n_var: -1] = np.identity(self.n_rest)

        restr = restr + 0
        obj_func = obj_func + 0

        return PL(self.n_rest, self.n_var + n_ones, obj_func, restr, RestrType.EQ)
