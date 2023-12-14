
import numpy as np

"""
Generate projection basis and extract its irreps info.
This procedure is performed once at the beginning.
"""


class BasisInfo(object):

    def __init__(self, basis, symm=True):

        self.basis = basis
        self.symm = symm

        self.basis_ls, self.basis_nls, self.basis_mat_idx = self.__extract_basis_nl_info()
        self.basis_l3s, self.basis_nl3s, self.basis_irreps = self.__compute_basis_l3_info(symm)

    def __extract_basis_nl_info(self):

        basis = self.basis

        ls = [x[0] for x in basis]  # [0, 1, 2]
        nls = [len(x) - 1 for x in basis]

        mat_size = [nls[idx] * (2*ls[idx]+1) for idx in range(len(ls))]
        mat_idx = list(np.cumsum(mat_size))
        mat_idx = [0] + mat_idx

        return ls, nls, mat_idx

    def __compute_basis_l3_info(self, symm):

        l1s = self.basis_ls
        nl1s, nl2s = self.basis_nls, self.basis_nls

        max_l3 = 2*max(l1s)
        l3s = [a for a in range(max_l3 + 1) for b in range(2)]
        nl3s = [0] * len(l3s)
        for idx1, l1 in enumerate(l1s):
            nl1 = nl1s[idx1]
            l2s = self.basis_ls if not symm else self.basis_ls[:idx1+1]
            for idx2, l2 in enumerate(l2s):
                nl2 = nl2s[idx2]
                parity = int((-1)**(l1+l2) < 0)
                for l3 in range(abs(l1-l2), l1+l2+1):
                    l3_idx = 2 * l3 + parity
                    nl3s[l3_idx] += nl1 * nl2

        irrep_str = []
        for a in range(2 * (max_l3 + 1)):
            if nl3s[a] > 0:
                p = 'e' if a % 2 == 0 else 'o'
                irrep = str(nl3s[a]) + 'x' + str(l3s[a]) + p
                irrep_str.append(irrep)

        return l3s, nl3s, '+'.join(irrep_str)
