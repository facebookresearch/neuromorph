# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Marvin Eisenberger.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from scipy import sparse
import numpy as np
from torch_geometric.nn import fps, knn_graph
import matplotlib.pyplot as plt
from param import *
from utils.base_tools import *


def plot_curr_shape(vert, triv_x):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        vert[:, 0],
        vert[:, 1],
        vert[:, 2],
        triangles=triv_x,
        cmap="viridis",
        linewidths=0.2,
    )
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.4, 0.4)


class Shape:
    """Class for shapes. (Optional) attributes are:
    vert: Vertices in the format nx3
    triv: Triangles in the format mx3
    samples: Index list of active vertices
    neigh: List of 2-Tuples encoding the adjacency of vertices
    neigh_hessian: Hessian/Graph Laplacian of the shape based on 'neigh'
    mahal_cov_mat: The covariance matrix of our anisotropic arap energy"""

    def __init__(self, vert=None, triv=None):
        self.vert = vert
        self.triv = triv
        self.samples = list(range(vert.shape[0]))
        self.neigh = None
        self.neigh_hessian = None
        self.mahal_cov_mat = None
        self.normal = None
        self.D = None
        self.sub = None
        self.vert_full = None

        if not self.triv is None:
            self.triv = self.triv.to(dtype=torch.long)

    def subsample_fps(self, goal_vert):
        assert (
            goal_vert <= self.vert.shape[0]
        ), "you cannot subsample to more vertices than n"

        ratio = goal_vert / self.vert.shape[0]
        self.samples = fps(self.vert.detach().to(device_cpu), ratio=ratio).to(device)
        self._neigh_knn()

    def reset_sampling(self):
        self.gt_sampling(self.vert.shape[0])

    def gt_sampling(self, n):
        self.samples = list(range(n))
        self.neigh = None

    def scale(self, factor, shift=True):
        self.vert = self.vert * factor

        if shift:
            self.vert = self.vert + (1 - factor) / 2

    def get_bounding_box(self):
        max_x, _ = self.vert.max(dim=0)
        min_x, _ = self.vert.min(dim=0)

        return min_x, max_x

    def to_box(self, shape_y):

        min_x, max_x = self.get_bounding_box()
        min_y, max_y = shape_y.get_bounding_box()

        extent_x = max_x - min_x
        extent_y = max_y - min_y

        self.translate(-min_x)
        shape_y.translate(-min_y)

        scale_fac = torch.max(torch.cat((extent_x, extent_y), 0))
        scale_fac = 1.0 / scale_fac

        self.scale(scale_fac, shift=False)
        shape_y.scale(scale_fac, shift=False)

        extent_x = scale_fac * extent_x
        extent_y = scale_fac * extent_y

        self.translate(0.5 * (1 - extent_x))
        shape_y.translate(0.5 * (1 - extent_y))

    def translate(self, offset):
        self.vert = self.vert + offset.unsqueeze(0)

    def get_vert(self):
        return self.vert[self.samples, :]

    def get_vert_shape(self):
        return self.get_vert().shape

    def get_triv(self):
        return self.triv

    def get_triv_np(self):
        return self.triv.detach().cpu().numpy()

    def get_vert_np(self):
        return self.vert[self.samples, :].detach().cpu().numpy()

    def get_vert_full_np(self):
        return self.vert.detach().cpu().numpy()

    def get_neigh(self, num_knn=5):
        if self.neigh is None:
            self.compute_neigh(num_knn=num_knn)

        return self.neigh

    def compute_neigh(self, num_knn=5):
        if len(self.samples) == self.vert.shape[0]:
            self._triv_neigh()
        else:
            self._neigh_knn(num_knn=num_knn)

    def get_edge_index(self, num_knn=5):
        edge_index_one = self.get_neigh(num_knn).t()
        edge_index = torch.zeros(
            [2, edge_index_one.shape[1] * 2], dtype=torch.long, device=self.vert.device
        )
        edge_index[:, : edge_index_one.shape[1]] = edge_index_one
        edge_index[0, edge_index_one.shape[1] :] = edge_index_one[1, :]
        edge_index[1, edge_index_one.shape[1] :] = edge_index_one[0, :]
        return edge_index

    def _triv_neigh(self):
        self.neigh = torch.cat(
            (self.triv[:, [0, 1]], self.triv[:, [0, 2]], self.triv[:, [1, 2]]), 0
        )

    def _neigh_knn(self, num_knn=5):
        vert = self.get_vert().detach()
        print("Compute knn....")
        self.neigh = (
            knn_graph(vert.to(device_cpu), num_knn, loop=False)
            .transpose(0, 1)
            .to(device)
        )

    def get_neigh_hessian(self):
        if self.neigh_hessian is None:
            self.compute_neigh_hessian()

        return self.neigh_hessian

    def compute_neigh_hessian(self):

        neigh = self.get_neigh()

        n_vert = self.get_vert().shape[0]

        H = sparse.lil_matrix(1e-3 * sparse.identity(n_vert))

        I = np.array(neigh[:, 0].detach().cpu())
        J = np.array(neigh[:, 1].detach().cpu())
        V = np.ones([neigh.shape[0]])
        U = -V
        H = H + sparse.lil_matrix(
            sparse.coo_matrix((U, (I, J)), shape=(n_vert, n_vert))
        )
        H = H + sparse.lil_matrix(
            sparse.coo_matrix((U, (J, I)), shape=(n_vert, n_vert))
        )
        H = H + sparse.lil_matrix(
            sparse.coo_matrix((V, (I, I)), shape=(n_vert, n_vert))
        )
        H = H + sparse.lil_matrix(
            sparse.coo_matrix((V, (J, J)), shape=(n_vert, n_vert))
        )

        self.neigh_hessian = H

    def rotate(self, R):
        self.vert = torch.mm(self.vert, R.transpose(0, 1))

    def to(self, device):
        self.vert = self.vert.to(device)
        self.triv = self.triv.to(device)

    def detach_cpu(self):
        self.vert = self.vert.detach().cpu()
        self.triv = self.triv.detach().cpu()
        if self.normal is not None:
            self.normal = self.normal.detach().cpu()
        if self.neigh is not None:
            self.neigh = self.neigh.detach().cpu()
        if self.D is not None:
            self.D = self.D.detach().cpu()
        if self.vert_full is not None:
            self.vert_full = self.vert_full.detach().cpu()
        if self.samples is not None and torch.is_tensor(self.samples):
            self.samples = self.samples.detach().cpu()
        if self.sub is not None:
            for i_s in range(len(self.sub)):
                for i_p in range(len(self.sub[i_s])):
                    self.sub[i_s][i_p] = self.sub[i_s][i_p].detach().cpu()

    def compute_volume(self):
        return self.compute_volume_shifted(self.vert)

    def compute_volume_shifted(self, vert_t):
        vert_t = vert_t - vert_t.mean(dim=0, keepdim=True)
        vert_triv = vert_t[self.triv, :].to(device_cpu)

        vol_tetrahedra = (vert_triv.det() / 6).to(device)

        return vol_tetrahedra.sum()

    def get_normal(self):
        if self.normal is None:
            self._compute_outer_normal()
        return self.normal

    def _compute_outer_normal(self):
        edge_1 = torch.index_select(self.vert, 0, self.triv[:, 1]) - torch.index_select(
            self.vert, 0, self.triv[:, 0]
        )
        edge_2 = torch.index_select(self.vert, 0, self.triv[:, 2]) - torch.index_select(
            self.vert, 0, self.triv[:, 0]
        )

        face_norm = torch.cross(1e4 * edge_1, 1e4 * edge_2)

        normal = my_zeros(self.vert.shape)
        for d in range(3):
            normal = torch.index_add(normal, 0, self.triv[:, d], face_norm)
        self.normal = normal / (1e-5 + normal.norm(dim=1, keepdim=True))


if __name__ == "__main__":
    print("main of shape_utils.py")
