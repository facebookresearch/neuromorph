# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Marvin Eisenberger.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim
from torch.nn import Parameter
from utils.shape_utils import *
from param import device_cpu
from scipy.sparse.linalg import spsolve
from scipy.sparse import kron, spdiags, csr_matrix, lil_matrix, eye
from utils.base_tools import *


class ParamBase:
    """Base class for parameters"""

    def from_dict(self, d):
        for key in d:
            if hasattr(self, key):
                self.__setattr__(key, d[key])

    def print_self(self):
        print("parameters: ")
        p_d = self.__dict__
        for k in p_d:
            print(k, ": ", p_d[k], "  ", end="")
        print("")


class Param(ParamBase):
    """Base class for hyperparameters of interpolation methods"""

    def __init__(self):
        self.lr = 0.001
        self.num_it = 20
        self.scales = [200, 500, 1000, 2000, 12500]
        self.odd_multisteps = False
        self.num_timesteps = 3

        self.log = True


# -----------------------------------------------------------------------------------


class InterpolationEnergy:
    """Base class for local distortion interpolation potentials"""

    def __init__(self):
        super().__init__()

    def forward_single(self, vert_new, vert_ref, shape_i):
        raise NotImplementedError()


class InterpolationEnergyHessian(InterpolationEnergy):
    """Abstract for interpolation potentials that also allow for second order optimization"""

    def __init__(self):
        super().__init__()

    def get_hessian(self, shape_i):
        raise NotImplementedError()


# -----------------------------------------------------------------------------------


class InterpolationModuleBase(torch.nn.Module):
    """Base class for an interpolation method"""

    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError()


class InterpolationModuleMultiscaleBase(InterpolationModuleBase):
    """Base class for multi-scale interpolation methods """

    def __init__(self):
        super().__init__()

    def step_multiscale(self, i_scale):
        raise NotImplementedError()


class InterpolationModule(InterpolationModuleBase):
    """Base class for interpolation methods based on a local distortion potential"""

    def __init__(self, energy: InterpolationEnergy):
        super().__init__()
        self.energy = energy

    def forward(self):
        raise NotImplementedError()


class InterpolationModuleMultiscale(
    InterpolationModule, InterpolationModuleMultiscaleBase
):
    """Base class for interpolation methods based on a local distortion potential"""

    def __init__(self, energy: InterpolationEnergy):
        super().__init__(energy)


class InterpolationModuleHessian(InterpolationModule):
    """Base class for interpolation methods that allow for second order optimization"""

    def __init__(self, energy: InterpolationEnergyHessian):
        super().__init__(energy)

    def mul_with_inv_hessian(self):
        raise NotImplementedError()


class InterpolationModuleSingle(InterpolationModule):
    """Class for interpolation methods that compute shape interpolation
    as a weighted combination X^*=sum_i lambda_i*X_i, where
    lambda is an n-dim vector prescribing a discrete probability distribution"""

    def __init__(self, energy: InterpolationEnergy, shape_array, interpolation_coeff):
        super().__init__(energy)

        assert (
            all([coeff >= 0 for coeff in interpolation_coeff])
            and sum(interpolation_coeff) == 1
        ), "interpolation coeffs need to prescribe a discrete probability distribution"

        self.shape_array = shape_array
        self.interpolation_coeff = interpolation_coeff

        self.vert_new = Parameter(self.average_vertex(), requires_grad=True)

    def average_vertex(self):
        num_shapes = len(self.shape_array)
        vert_avg = my_zeros(self.shape_array[0].get_vert().shape)

        for i in range(num_shapes):
            vert_avg = (
                vert_avg + 1 / num_shapes * self.shape_array[i].get_vert().clone()
            )

        return vert_avg

    def reset_vert_new(self):
        self.vert_new.data = self.average_vertex()

    def set_vert_new(self, vert):
        self.vert_new.data = vert

    def forward(self):
        E_total = 0

        for i in range(len(self.shape_array)):
            E_curr = self.interpolation_coeff[i] * self.energy.forward_single(
                self.vert_new, self.shape_array[i].get_vert(), self.shape_array[i]
            )
            E_total = E_total + E_curr

        return E_total, [E_total]


class InterpolationModuleSingleHessian(
    InterpolationModuleSingle, InterpolationModuleHessian
):
    """Same as InterpolationModuleSingle, but with second order optimization"""

    def __init__(
        self, energy: InterpolationEnergyHessian, shape_array, interpolation_coeff
    ):
        super().__init__(energy, shape_array, interpolation_coeff)

    def mul_with_inv_hessian(self):
        hess = self.energy.get_hessian(self.shape_array[0])

        grad_hess = spsolve(hess, self.vert_new.grad.to(device_cpu).detach().cpu())
        self.vert_new.grad = torch.tensor(grad_hess, dtype=torch.float32, device=device)


class InterpolationModuleBvp(InterpolationModuleMultiscale):
    """Class that computes shape interpolations by optimizing for the geometry of
    an intermediate sequence of shapes. Two consecutive shapes are required to have
    a small local distortion between each other wrt. some energy 'InterpolationEnergy'.
    The scheme is initialized with a linear interpolation between shape_x and shape_y.
    Moreover, it allows for a hierarchical set of multi-scale shapes_x with an increasing
    resolution and it refines the temporal discretization over time."""

    def __init__(
        self,
        energy: InterpolationEnergy,
        shape_x: Shape,
        shape_y: Shape,
        param=Param(),
        vertices=None,
    ):
        super().__init__(energy)

        self.param = param
        self.shape_x = shape_x
        self.shape_y = shape_y

        if vertices is None:
            vertices = self.compute_vertices()
        self.vert_sequence = Parameter(vertices, requires_grad=True)

    def forward(self):
        num_t = self.param.num_timesteps

        E_x = self.energy.forward_single(
            self.vert_sequence[0, ...].detach(),
            self.vert_sequence[1, ...],
            self.shape_x,
        )
        E_y = self.energy.forward_single(
            self.vert_sequence[num_t - 2, ...],
            self.vert_sequence[num_t - 1, ...].detach(),
            self.shape_x,
        )

        E_total = E_x + E_y

        for i in range(1, num_t - 2):
            E_curr = self.energy.forward_single(
                self.vert_sequence[i, ...], self.vert_sequence[i + 1, ...], self.shape_x
            )
            E_total = E_total + E_curr

        E_total = E_total / (num_t - 1)

        return E_total, [E_total]

    def get_vert_sequence(self):
        return self.vert_sequence

    def compute_vertices(self):
        num_t = self.param.num_timesteps
        vertices = my_zeros([num_t, self.shape_x.get_vert_shape()[0], 3])
        for i in range(0, num_t):
            lambd = i / (num_t - 1)
            vertices[i, ...] = (
                1 - lambd
            ) * self.shape_x.get_vert().clone() + lambd * self.shape_y.get_vert().clone()

        return vertices

    def copy_self(self, vertices=None):
        return InterpolationModuleBvp(
            self.energy, self.shape_x, self.shape_y, self.param, vertices
        )

    def step_multiscale(self, i_scale):
        vertices = self.vert_sequence.data

        if i_scale % 2 == 0:
            vertices = self.insert_additional_vertices(vertices)
        else:
            vertices = self.upsample_resolution(vertices)

        return self.copy_self(vertices)

    def upsample_resolution(self, vertices):
        num_vert = self.shape_x.next_resolution()[0]
        num_t = vertices.shape[0]

        vertices_new = my_zeros([num_t, num_vert, 3])

        for t in range(1, num_t - 1):
            l = (t - 1) / (num_t - 2)
            vertices_new[t, ...] = (1 - l) * self.shape_x.apply_upsampling(
                vertices[t, ...]
            ) + l * self.shape_y.apply_upsampling(vertices[t, ...])

        self.shape_x.increase_scale_idx()
        self.shape_y.increase_scale_idx()

        vertices_new[0, ...] = self.shape_x.vert.clone()
        vertices_new[num_t - 1, ...] = self.shape_y.vert.clone()

        return vertices_new

    def insert_additional_vertices(self, vertices):
        num_t = self.param.num_timesteps
        num_vert = vertices.shape[1]
        self.param.num_timesteps = num_t * 2 - 1

        vertices = vertices.unsqueeze(1)

        vertices = vertices * torch.as_tensor(
            [1, 0], device=device, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        vertices = vertices.reshape([num_t * 2, num_vert, 3])
        vertices = vertices[0 : num_t * 2 - 1, ...]

        for i in range(num_t - 1):
            vertices[i * 2 + 1, ...] = (
                0.5 * (vertices[i * 2, ...] + vertices[i * 2 + 2, ...]).clone()
            )

        self.vert_sequence.data = vertices

        return vertices


class InterpolationModuleHesstrans(InterpolationModuleMultiscale):
    def __init__(
        self,
        energy: InterpolationEnergy,
        shape_x: Shape,
        shape_y: Shape,
        param=Param(),
        vertices=None,
    ):
        super().__init__(energy)

        self.param = param
        self.shape_x = shape_x
        self.shape_y = shape_y

        self.hess = torch.as_tensor(
            shape_x.get_neigh_hessian().todense(), dtype=torch.float, device=device
        )
        self.hess += 1e-3 * my_eye(self.hess.shape[0])
        self.hess_inv = self.hess.inverse()

        if vertices is None:
            vertices = self.compute_vertices()
        self.vert_sequence = Parameter(vertices, requires_grad=True)

    def forward(self):
        num_t = self.param.num_timesteps

        v_s = self.get_vert_sequence()

        E_x = self.energy.forward_single(
            v_s[0, ...].detach(), v_s[1, ...], self.shape_x
        )
        E_y = self.energy.forward_single(
            v_s[num_t - 2, ...], v_s[num_t - 1, ...].detach(), self.shape_x
        )

        E_total = E_x + E_y

        for i in range(1, num_t - 2):
            E_curr = self.energy.forward_single(
                v_s[i, ...], v_s[i + 1, ...], self.shape_x
            )
            E_total = E_total + E_curr

        E_total = E_total / (num_t - 1)

        return E_total, [E_total]

    def get_vert_sequence(self):
        v_s = my_zeros(self.vert_sequence.shape)
        for i in range(0, v_s.shape[0]):
            v_s[i] = torch.mm(self.hess_inv, self.vert_sequence[i])
        return v_s

    def compute_vertices(self):
        num_t = self.param.num_timesteps
        vertices = my_zeros([num_t, self.shape_x.get_vert_shape()[0], 3])
        for i in range(0, num_t):
            lambd = i / (num_t - 1)
            vertices[i, ...] = (
                1 - lambd
            ) * self.shape_x.get_vert().clone() + lambd * self.shape_y.get_vert().clone()
            vertices[i, ...] = torch.mm(self.hess, vertices[i, ...])

        return vertices

    def copy_self(self, vertices=None):
        return InterpolationModuleBvp(
            self.energy, self.shape_x, self.shape_y, self.param, vertices
        )

    def step_multiscale(self, i_scale):
        vertices = self.vert_sequence.data

        if i_scale % 2 == 0:
            vertices = self.insert_additional_vertices(vertices)
        else:
            vertices = self.upsample_resolution(vertices)

        return self.copy_self(vertices)

    def upsample_resolution(self, vertices):
        num_vert = self.shape_x.next_resolution()[0]
        num_t = vertices.shape[0]

        vertices_new = my_zeros([num_t, num_vert, 3])

        for t in range(1, num_t - 1):
            l = (t - 1) / (num_t - 2)
            vertices_new[t, ...] = (1 - l) * self.shape_x.apply_upsampling(
                torch.mm(self.hess_inv, vertices[t, ...])
            ) + l * self.shape_y.apply_upsampling(
                torch.mm(self.hess_inv, vertices[t, ...])
            )
            vertices_new[t, ...] = torch.mm(self.hess, vertices_new[t, ...])

        self.shape_x.increase_scale_idx()
        self.shape_y.increase_scale_idx()

        vertices_new[0, ...] = torch.mm(self.hess, self.shape_x.vert.clone())
        vertices_new[num_t - 1, ...] = torch.mm(self.hess, self.shape_y.vert.clone())

        return vertices_new

    def insert_additional_vertices(self, vertices):
        num_t = self.param.num_timesteps
        num_vert = vertices.shape[1]
        self.param.num_timesteps = num_t * 2 - 1

        vertices = vertices.unsqueeze(1)

        vertices = vertices * torch.as_tensor(
            [1, 0], device=device, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        vertices = vertices.reshape([num_t * 2, num_vert, 3])
        vertices = vertices[0 : num_t * 2 - 1, ...]

        for i in range(num_t - 1):
            vertices[i * 2 + 1, ...] = (
                0.5 * (vertices[i * 2, ...] + vertices[i * 2 + 2, ...]).clone()
            )

        self.vert_sequence.data = vertices

        return vertices


class InterpolationModuleBvpHessian(InterpolationModuleBvp, InterpolationModuleHessian):
    """Same as InterpolationModuleBvp, but with second order optimization"""

    def __init__(
        self,
        energy: InterpolationEnergy,
        shape_x: Shape,
        shape_y: Shape,
        param=Param(),
        vertices=None,
    ):
        super().__init__(energy, shape_x, shape_y, param, vertices=vertices)

    def copy_self(self, vertices=None):
        return InterpolationModuleBvpHessian(
            self.energy, self.shape_x, self.shape_y, self.param, vertices
        )

    def mul_with_inv_hessian(self):
        num_t = self.param.num_timesteps
        n_vert = self.shape_x.get_vert_shape()[0]

        hess_1d = self.energy.get_hessian(self.shape_x)

        central_diff_diags = -np.ones([3, num_t])
        central_diff_diags[1, :] = 2
        central_diff = lil_matrix(
            spdiags(central_diff_diags, np.array([-1, 0, 1]), num_t, num_t)
        )
        central_diff[[0, num_t - 1], :] = 0

        boundary_cond = lil_matrix((num_t, num_t))

        boundary_cond[0, 0] = 1
        boundary_cond[num_t - 1, num_t - 1] = 1

        hess = csr_matrix(
            kron(central_diff, hess_1d) + kron(boundary_cond, eye(n_vert))
        )

        grad_hess = spsolve(
            hess, self.vert_sequence.grad.view(-1, 3).to(device_cpu).detach().cpu()
        )
        self.vert_sequence.grad = torch.tensor(
            grad_hess, dtype=torch.float32, device=device
        ).view_as(self.vert_sequence)
        self.vert_sequence.grad = self.vert_sequence.grad.clone()


# -----------------------------------------------------------------------------------


class Interpolation:
    """Base class for interpolation optimizers for some
    interpolation method 'InterpolationModule'"""

    def __init__(self, interp_module: InterpolationModule, param=Param()):
        self.interp_module = interp_module
        self.param = param

        def interpolate(self, super_idx=-1):
            raise NotImplementedError()


class InterpolationNewton(Interpolation):
    """Shape interpolation with Newton optimization"""

    def __init__(self, interp_module: InterpolationModuleHessian, param=Param()):
        super().__init__(interp_module, param)

    def interpolate(self, super_idx=-1):

        lr = self.param.lr
        optimizer = torch.optim.Adam(self.interp_module.parameters(), lr=lr)

        self.interp_module.train()

        E = 0

        for it in range(self.param.num_it):
            optimizer.zero_grad()
            E, Elist = self.interp_module()
            E.backward()

            self.interp_module.mul_with_inv_hessian()

            optimizer.step()

            if self.param.log:
                if self.param.log:
                    if super_idx >= 0:
                        print(
                            "Super {:02d}, It {:03d}, E: {:.5f}".format(
                                super_idx, it, Elist[0]
                            )
                        )
                    else:
                        print("It {:03d}, E: {:.5f}".format(it, Elist[0]))

        self.energy = self.interp_module.eval()

        return E.detach()


class InterpolationLBFGS(Interpolation):
    """Shape interpolation with the quasi-Newton scheme LBFGS"""

    def __init__(self, interp_module: InterpolationModule, param=Param()):
        super().__init__(interp_module, param)

    def interpolate(self, super_idx=-1):

        lr = self.param.lr
        optimizer = torch.optim.LBFGS(
            self.interp_module.parameters(), lr=lr, line_search_fn="strong_wolfe"
        )

        self.interp_module.train()

        for it in range(self.param.num_it):

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                E, Elist = self.interp_module()
                if E.requires_grad:
                    E.backward()
                    if self.param.log:
                        if super_idx >= 0:
                            print(
                                "Super {:02d}, It {:03d}, E: {:.5f}".format(
                                    super_idx, it, Elist[0]
                                )
                            )
                        else:
                            print("It {:03d}, E: {:.5f}".format(it, Elist[0]))
                return E

            optimizer.step(closure)

        self.energy = self.interp_module.eval()


class InterpolationGD(Interpolation):
    """Shape interpolation with gradient descent 'Adam'"""

    def __init__(self, interp_module: InterpolationModuleBase, param=Param()):
        super().__init__(interp_module, param)

    def interpolate(self, super_idx=-1):

        lr = self.param.lr
        optimizer = torch.optim.Adam(self.interp_module.parameters(), lr=lr)

        self.interp_module.train()

        E = 0

        for it in range(self.param.num_it):
            optimizer.zero_grad()
            E, Elist = self.interp_module()
            E.backward()
            if self.param.log:
                if super_idx >= 0:
                    print(
                        "Super {:02d}, It {:03d}, E: {:.5f}".format(
                            super_idx, it, Elist[0]
                        )
                    )
                else:
                    print("It {:03d}, E: {:.5f}".format(it, Elist[0]))

            optimizer.step()

        self.energy = self.interp_module.eval()

        return E.detach()


class InterpolationMultiscale:
    """Shape interpolation with an additional multi-scale strategy"""

    def __init__(self, interp: Interpolation, param=Param(), num_it_super=None):
        self.interp = interp
        self.param = param
        self.E = None

        if num_it_super is None:
            if self.param.odd_multisteps:
                self.num_it_super = len(self.param.scales) * 2
            else:
                self.num_it_super = len(self.param.scales) * 2 - 1
        else:
            self.num_it_super = num_it_super

    def interpolate(self):

        for i in range(self.num_it_super - 1):
            self.interp.interpolate(i)
            self.interp.interp_module = self.interp.interp_module.step_multiscale(i)
        self.E = self.interp.interpolate(self.num_it_super - 1)

        return self.interp.interp_module


# -----------------------------------------------------------------------------------


if __name__ == "__main__":
    print("main of interpolation_base.py")
