# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional
from utils.base_tools import *
from param import *


def arap_exact(vert_diff_t, vert_diff_0, neigh, n_vert):
    S_neigh = torch.bmm(vert_diff_t.unsqueeze(2), vert_diff_0.unsqueeze(1))

    S = my_zeros([n_vert, 3, 3])

    S = torch.index_add(S, 0, neigh[:, 0], S_neigh)
    S = torch.index_add(S, 0, neigh[:, 1], S_neigh)

    U, _, V = torch.svd(S.cpu(), compute_uv=True)

    U = U.to(device)
    V = V.to(device)

    R = torch.bmm(U, V.transpose(1, 2))

    Sigma = my_ones((R.shape[0], 1, 3))
    Sigma[:, :, 2] = torch.det(R).unsqueeze(1)

    R = torch.bmm(U * Sigma, V.transpose(1, 2))

    return R


def arap_energy_exact(vert_t, vert_0, neigh, lambda_reg_len=1e-6):
    n_vert = vert_t.shape[0]

    vert_diff_t = vert_t[neigh[:, 0], :] - vert_t[neigh[:, 1], :]
    vert_diff_0 = vert_0[neigh[:, 0], :] - vert_0[neigh[:, 1], :]

    R_t = arap_exact(vert_diff_t, vert_diff_0, neigh, n_vert)

    R_neigh_t = 0.5 * (
        torch.index_select(R_t, 0, neigh[:, 0])
        + torch.index_select(R_t, 0, neigh[:, 1])
    )

    vert_diff_0_rot = torch.bmm(R_neigh_t, vert_diff_0.unsqueeze(2)).squeeze()
    acc_t_neigh = vert_diff_t - vert_diff_0_rot

    E_arap = acc_t_neigh.norm() ** 2 + lambda_reg_len * (vert_t - vert_0).norm() ** 2

    return E_arap


if __name__ == "__main__":
    print("main of arap_potential.py")
