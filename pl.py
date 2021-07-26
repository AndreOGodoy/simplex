from enum import Enum, auto
import numpy as np
from typing import Optional, Tuple


class PLType(Enum):
    OPTIMAL = auto(),
    INVIABLE = auto(),
    UNLIMITED = auto()


# Classe de retorno que contém informações sobre PL's resolvidas.
class SimplexReturn:
    def __init__(self, pl_type: PLType,
                 certificate: np.ndarray,
                 optimal_value: Optional[float] = None,
                 solution: Optional[np.ndarray] = None,
                 base: Optional[np.ndarray] = None,
                 obj_func: Optional[np.ndarray] = None):
        self.pl_type = pl_type
        self.certificate = certificate
        self.optimal_value = optimal_value
        self.solution = solution
        self.base = base
        self.obj_func = obj_func


class RestrType(Enum):
    LESS_EQ = auto(),
    EQ = auto()


class ObjFuncType(Enum):
    MAX = auto(),
    MIN = auto(),


def idx_first(cond_arr: np.ndarray) -> Optional[int]:
    indexes = np.where(cond_arr)
    return indexes[0][0] if indexes[0].size > 0 else None


class PL:
    def __init__(self, n_rest: int, n_var: int,
                 obj_func: np.ndarray, restr: np.ndarray,
                 restr_type: RestrType = RestrType.LESS_EQ,
                 obj_func_type: ObjFuncType = ObjFuncType.MAX,
                 op_reg: Optional[np.ndarray] = None):
        self.n_var = n_var
        self.n_rest = n_rest

        self.obj_func = obj_func
        self.obj_func_type = obj_func_type

        self.restr = restr
        self.restr_type = restr_type

        if op_reg is None:
            self.op_reg = np.identity(n_rest, dtype=float)
        else:
            self.op_reg = op_reg

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
            return None
        elif self.restr_type is RestrType.EQ and not inplace:
            return PL(self.n_rest, self.n_var, self.obj_func.copy(), self.restr.copy(),
                      self.restr_type, self.obj_func_type)

        new_obj_func = np.append(self.obj_func, [0] * self.n_rest)

        # Transforma uma PL com função objetivo MIN em uma PL com função objetivo MAX
        if self.obj_func_type is ObjFuncType.MIN:
            new_obj_func *= -1

        # Adiciona matriz identidade à direita das restrições originais para representar
        # as variáveis de folga adicionadas
        new_restr = np.hstack((self.restr, np.zeros((self.n_rest, self.n_rest))))
        new_restr[:, -1] = new_restr[:, self.n_var]
        new_restr[:, self.n_var: -1] = np.identity(self.n_rest)

        # Contabiliza as variáveis de folga adicionadas (1 para cada restrição original)
        new_n_var = self.n_var + self.n_rest

        if inplace:
            self.restr = new_restr
            self.n_var = new_n_var
            self.obj_func = new_obj_func
            self.restr_type = RestrType.EQ
            self.obj_func_type = ObjFuncType.MAX
            return None

        new_pl = PL(self.n_rest, new_n_var, new_obj_func, new_restr, RestrType.EQ, ObjFuncType.MAX)
        return new_pl

    def pivot_self_by(self, row_idx: int, col_idx: int):
        # Divide a linha do pivô para transformá-lo em 1, repete a operação na VERO
        self.op_reg[row_idx, :] = self.op_reg[row_idx, :] / self.restr[row_idx, col_idx]
        self.restr[row_idx, :] = self.restr[row_idx, :] / self.restr[row_idx, col_idx]

        # Opera sobre as demais linhas que não do pivô
        pivot = self.restr[row_idx, col_idx]
        for idx, row in enumerate(self.restr):
            if idx == row_idx:
                continue

            ratio = row[col_idx] / pivot

            self.restr[idx] -= ratio * self.restr[row_idx, :]

            # Espelha operações na VERO
            self.op_reg[idx] -= ratio * self.op_reg[row_idx, :]

        # Opera sobre demais valores, fora da matriz self.restr
        ratio = self.obj_func[col_idx] / pivot
        self.obj_func -= ratio * self.restr[row_idx, :-1]
        self.op_reg_c -= ratio * self.op_reg[row_idx]
        self.optimal_value -= ratio * self.restr[row_idx, -1]

        self.restr = self.restr
        self.obj_func = self.obj_func

    def primal_simplex(self, base: Optional[np.ndarray] = None, is_aux_pl: bool = False) -> SimplexReturn:
        canonical = self.into_canonical(base=base, is_aux_pl=is_aux_pl)

        # O algoritmo para apenas quando uma das duas condições de parada delimitadas pelos if
        # abaixo são alcançadas. São sempre alcançadas.
        while True:

            # Escolhe a coluna como sendo a primeira com entrada > 0
            possible_columns = np.where(np.round(canonical.obj_func, 7) > 0)[0]
            if possible_columns.size == 0:
                solution, base = canonical.get_basic_solution()
                return SimplexReturn(pl_type=PLType.OPTIMAL,
                                     certificate=canonical.op_reg_c,
                                     optimal_value=canonical.optimal_value,
                                     solution=solution,
                                     base=base,
                                     obj_func=canonical.obj_func)

            column = possible_columns[0]

            # Obtém as razões b / coluna. Se algum b_i da coluna for <= 0, então a razão vale np.inf
            ratios = canonical.__get_simplex_primal_ratio(column)
            if np.all(np.isclose(ratios, np.inf)):
                solution, _ = canonical.get_basic_solution()
                certificate = canonical.__unlimited_certificate()
                return SimplexReturn(pl_type=PLType.UNLIMITED,
                                     solution=solution,
                                     certificate=certificate)

            # Escolhe o pivô como sendo o primeiro elemento de menor razão
            min_ratio_idx = np.where(np.isclose(ratios, np.min(ratios)))[0][0]
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
                               self.restr.copy(), self.restr_type, self.obj_func_type, self.op_reg)

            # No caso de uma PL auxiliar, realiza-se um pivoteamento especial, onde as variáveis artificiais
            # são pivôs e utilizadas para zerar os valores do vetor objetivo em suas respectivas colunas
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

        # Adiciona N -1's no vetor C, onde N = número de restrições da PL
        n_ones = self.n_rest
        obj_func = np.zeros(self.obj_func.shape[0] + n_ones)
        obj_func[-n_ones:] = -1

        op_reg = self.op_reg
        restr = self.restr

        # Faz com que b >= 0. Reflete operações na matriz VERO
        for line_idx, b in enumerate(restr[:, -1]):
            if b < 0:
                restr[line_idx] *= -1
                op_reg[line_idx] *= -1

        # Adiciona uma matriz identidade lado da matriz de restrições original,
        # de tamanho N x N, onde N = número de -1's adicionadas anteriormente,
        # ou, equivalentemente, N = número de restrições da PL
        restr = np.hstack((restr, np.zeros((n_ones, n_ones))))
        restr[:, -1] = restr[:, self.n_var]
        restr[:, self.n_var: -1] = np.identity(self.n_rest)

        restr = restr
        obj_func = obj_func

        pl = PL(self.n_rest, self.n_var + n_ones, obj_func, restr, RestrType.EQ)
        pl.op_reg = op_reg
        return pl

    def solve(self, debug_inplace: bool = False) -> SimplexReturn:
        original_n_var = self.n_var
        original_n_rest = self.n_rest

        pl_eq: Optional['PL'] = self
        if debug_inplace:
            self.into_equality_form(inplace=True)
        else:
            pl_eq = self.into_equality_form()

        assert pl_eq is not None
        aux = pl_eq.get_aux_pl()
        response = aux.primal_simplex(is_aux_pl=True)

        if response.pl_type is PLType.OPTIMAL and not np.allclose(response.optimal_value, 0):
            return SimplexReturn(PLType.INVIABLE,
                                 response.certificate)

        assert response.base is not None
        # Trunca a base para que esteja de tamanho compatível com a PL original em forma de igualdades
        base = response.base[response.base <= pl_eq.n_var]

        # Caso a base não esteja completa, adiciona primeira coluna da PL original
        # que possua entrada < 0 e que não esteja contida na base
        if base.size < pl_eq.n_rest:
            columns = np.array(range(pl_eq.n_var))
            columns_not_in_base = np.setdiff1d(np.union1d(columns, base), np.intersect1d(columns, base))

            if debug_inplace:
                pl_eq = self

            first_possible_idx = idx_first(response.obj_func[columns_not_in_base] < 0)

            first_possible = columns_not_in_base[first_possible_idx]
            base = np.hstack((base, first_possible))

        if debug_inplace:
            response_2 = self.primal_simplex(base=base)
        else:
            response_2 = pl_eq.primal_simplex(base=base)

        solution = response_2.solution[:original_n_var] if response_2.solution is not None else None

        # Caso o certificado seja se otimalidade, o trunca em relação ao número de restrições da PL original.
        # Caso contrário, o mesmo será de ilimitabilidade, e deve ser truncado em relação ao número de variáveis
        # da PL original
        certificate = response_2.certificate[:original_n_rest] \
            if response_2.pl_type is PLType.OPTIMAL else response_2.certificate[:original_n_var]

        return SimplexReturn(response_2.pl_type,
                             certificate,
                             response_2.optimal_value,
                             solution,
                             base)

    def __unlimited_certificate(self) -> np.ndarray:
        certificate = np.zeros(self.n_var)
        identity = np.identity(self.n_rest)

        # Identifica a coluna da matriz de restrições que indica que a PL é ilimitada.
        # No caso, uma coluna cuja todas as entradas são não-positivas.
        target_column_idx = np.where((self.restr <= 0).all(axis=0))[0]
        if target_column_idx.size == 0:
            raise ValueError("Não é possível determinar coluna ilimitante")

        target_column = self.restr.T[target_column_idx[0]]

        # Monta o certificado seguindo a lógica:
        #   Passa por cada coluna de A fazendo o seguinte:
        #       Adiciona o valor -c ao certificado caso a coluna seja uma base,
        #           onde c = valor contido na mesma linha que o 1 da coluna-base na coluna-alvo
        #       Adiciona o valor 1 ao certificado caso a coluna seja a coluna-alvo
        #       Adiciona o valor 0 caso a coluna não esteja não seja básica nem alvo
        for idx, col in enumerate(self.restr[:, :-1].T):
            idx_in_base = np.where((identity == col).all(axis=1))[0]
            if idx_in_base.size != 0:
                pos = idx_in_base[0]
                certificate[idx] = -target_column[pos]
            elif np.any(col != target_column):
                certificate[idx] = 0
            elif np.all(col == target_column):
                certificate[idx] = 1

        return certificate

    def get_basic_solution(self) -> Tuple[np.ndarray, np.ndarray]:
        restr = self.restr[:, :-1]
        b = self.restr[:, -1]

        identity = np.identity(restr.shape[0])
        solution = np.zeros(restr.shape[1])

        indexes = []
        order = []

        # Constrói uma solução básica para a PL, verificando quais os valores de b
        # correspondem às colunas da base. Leva em consideração a ordem das colunas
        for col_idx, col in enumerate(restr.T):
            result = np.where(np.isclose(identity, col).all(axis=1))[0]
            if result.size != 0:
                solution[col_idx] = b[result[0]]

                indexes.append(col_idx)
                order.append(result[0])
            else:
                solution[col_idx] = 0

        np_indexes = np.array(indexes)
        np_order = np.array(order)
        base = np_indexes[np_order]

        # Retorna também uma base, composta pelo primeiro conjunto de colunas básicas encontrado.
        return solution, np.unique(base)

    def __get_simplex_primal_ratio(self, column: int) -> np.ndarray:
        a = self.restr[:, -1]
        b = self.restr[:, column]
        divided = np.array([a_v / b_v if b_v > 0 else np.inf for a_v, b_v in zip(a, b)])
        return divided
