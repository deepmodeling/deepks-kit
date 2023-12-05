
import math

import torch
import torch.nn as nn

import e3nn.o3


class ClebschGordan(nn.Module):

    """
    Helper class that stores Clebsch-Gordan coefficients
    """

    def __init__(self, lmax=3, reorder_p=True, change_l3_basis=False,
                 device=torch.device('cpu'), dtype=torch.float64):
        super(ClebschGordan, self).__init__()

        self.lmax = lmax

        # this specifies the change of basis yzx -> xyz
        change_of_coord = torch.tensor([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=dtype, device=device)

        rot_mat = {0: torch.eye(1, dtype=dtype, device=device)}
        for l in range(1, 2*lmax+1):
            # apply change of basis to accommodate YXY Wigner D matrix convention in e3nn
            rot_mat[l] = torch.as_tensor(e3nn.o3.Irrep(l, 1).D_from_matrix(change_of_coord),
                                         dtype=dtype, device=device)

        for l1 in range(lmax):
            for l2 in range(lmax):
                for l3 in range(abs(l1-l2), l1+l2+1):
                    name = 'cg_{}_{}_{}'.format(l1, l2, l3)
                    if name not in dir(self):
                        # this is the CG coeffs that follow the convention of e3nn
                        cg = e3nn.o3.wigner_3j(l1, l2, l3, dtype=dtype, device=device) * math.sqrt(2*l3+1)

                        if change_l3_basis:
                            Q3 = rot_mat[l3]
                            cg = torch.einsum("iln,mn->ilm", cg, Q3)

                        # change the order of l=1 basis functions in pyscf xyz->yzx
                        if reorder_p:
                            if l1 == 1:
                                cg = torch.einsum("ji,jln->iln", change_of_coord.T.to(dtype=dtype), cg)
                            if l2 == 1:
                                cg = torch.einsum("iln,ml->imn", cg, change_of_coord.to(dtype=dtype))

                        self.register_buffer(name, cg)

    def forward(self, l1, l2, l3):

        return getattr(self, 'cg_{}_{}_{}'.format(l1, l2, l3))
