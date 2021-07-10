from enum import Enum, auto
import numpy as np
from typing import *

class PLType(Enum):
    OPTIMAL = auto(),
    INVIABLE = auto(),
    UNLIMITED = auto()


class RestrType(Enum):
    LESS_EQ = auto(),
    EQ = auto()


class ObjFuncType(Enum):
    MAX = auto(),
    MIN = auto(),


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

    def get_tableaux(self) -> str:
        output_str = ''
        for c in self.obj_func.astype(float):
            c = c * -1 + 0
            output_str += f'{c:>5.2f} '

        output_str += f' | {str(self.optimal_value) : >5}\n'
        output_str += '----------' * self.n_var + '\n'

        for i in range(self.n_rest):
            for j in range(self.n_var + 1):
                if j == self.n_var:
                    output_str += ' | '
                output_str += f'{self.restr[i, j]:>5.2f} '
            output_str += '\n'

        return output_str

    def into_equality_form(self) -> 'PL':
        if self.restr_type is RestrType.EQ:
            return self

        new_obj_func = np.append(self.obj_func, [0]*self.n_rest)
        if self.obj_func_type is ObjFuncType.MIN:
            new_obj_func *= -1

        new_restr = np.hstack((self.restr, np.zeros((self.n_rest, self.n_rest))))
        new_restr[:, -1] = new_restr[:, self.n_var]
        new_restr[:, self.n_var: -1] = np.identity(self.n_rest)

        new_n_var = self.n_var + self.n_rest

        new_pl = PL(self.n_rest, new_n_var, new_obj_func, new_restr, RestrType.EQ, ObjFuncType.MAX)
        return new_pl

    def pivot_self_by(self, row_idx: int, col_idx: int):
        self.restr[row_idx, :] = self.restr[row_idx, :] / self.restr[row_idx, col_idx]
        pivot = self.restr[row_idx, col_idx]

        for idx, row in enumerate(self.restr):
            if idx == row_idx:
                continue

            ratio = row[col_idx] / pivot
            self.restr[idx] -= ratio * self.restr[row_idx, :]

        ratio = self.obj_func[col_idx] / pivot
        self.obj_func -= ratio * self.restr[row_idx, :-1]
        self.optimal_value -= ratio * self.restr[row_idx, -1]

    def primal_simplex(self):
        raise NotImplementedError

    def to_canonical(self):
        raise NotImplementedError

    def get_aux_pl(self) -> 'PL':
        pl_eq_form = self.into_equality_form()
        n_ones = pl_eq_form.n_rest

        obj_func = np.zeros(pl_eq_form.obj_func.shape[0] + n_ones)
        obj_func[-n_ones:] = -1

        restr = np.hstack((pl_eq_form.restr, np.zeros((n_ones, n_ones))))
        restr[:, -1] = restr[:, pl_eq_form.n_var]
        restr[:, pl_eq_form.n_var: -1] = np.identity(pl_eq_form.n_rest)

        return PL(pl_eq_form.n_rest, pl_eq_form.n_var + n_ones, obj_func, restr, RestrType.EQ)
