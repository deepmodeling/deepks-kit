
import torch

"""
Generate projection basis and extract its irreps info.
This procedure is performed once at the beginning.
"""


class BasisInfo(object):

    def __init__(self, basis):

        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0" if (torch.cuda.is_available() and args.device == "gpu") else "cpu")
        self.dtype = torch.float64
        self.basis = basis

        self.basis_ls, self.basis_nls, self.basis_mat_idx = self.__extract_basis_nl_info()
        self.basis_l3s, self.basis_nl3s, self.basis_irreps = self.__compute_basis_l3_info()

    def __extract_basis_nl_info(self) -> (torch.Tensor, torch.Tensor, torch.Tensor):

        basis = self.basis

        ls = torch.tensor([x[0] for x in basis], device=self.device, dtype=torch.int)  # [0, 1, 2]
        nls = torch.tensor([len(x) - 1 for x in basis], device=self.device, dtype=torch.int)

        mat_size = torch.tensor([nls[idx] * (2 * ls[idx] + 1) for idx in range(len(ls))])
        mat_idx = torch.cumsum(mat_size, dim=0)
        mat_idx = torch.cat((torch.tensor([0], device=self.device, dtype=torch.int), mat_idx))

        return ls, nls, mat_idx

    def __compute_basis_l3_info(self) -> (torch.Tensor, torch.Tensor, str):

        l1s, l2s = self.basis_ls, self.basis_ls
        nl1s, nl2s = self.basis_nls, self.basis_nls

        device = l1s.device

        max_l3 = torch.max(l1s) + torch.max(l2s)
        l3s = torch.tensor([a for a in range(max_l3 + 1) for b in range(2)], device=device, dtype=torch.int)
        nl3s = torch.tensor([0 for a in range(len(l3s))], device=device, dtype=torch.int)
        for idx1, l1 in enumerate(l1s):
            nl1 = nl1s[idx1]
            for idx2, l2 in enumerate(l2s):
                nl2 = nl2s[idx2]
                parity = int((-1) ** (l1 + l2) < 0)
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    l3_idx = 2 * l3 + parity
                    nl3s[l3_idx] += (nl1 * nl2).item()

        irrep_str = []
        for a in range(2 * (max_l3 + 1)):
            if nl3s[a] > 0:
                p = 'e' if a % 2 == 0 else 'o'
                irrep = str(nl3s[a].item()) + 'x' + str(l3s[a].item()) + p
                irrep_str.append(irrep)

        return l3s, nl3s, '+'.join(irrep_str)
