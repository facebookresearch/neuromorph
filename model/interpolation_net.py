# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
from utils.arap_interpolation import *
from data.data import *
from model.layers import *


class NetParam(ParamBase):
    """Base class for hyperparameters of interpolation methods"""

    def __init__(self):
        super().__init__()
        self.lr = 1e-4
        self.num_it = 600
        self.batch_size = 16
        self.num_timesteps = 0
        self.hidden_dim = 128
        self.lambd = 1
        self.lambd_geo = 50

        self.log_freq = 10
        self.val_freq = 10

        self.log = True


class InterpolationModBase(torch.nn.Module):
    def __init__(self, interp_energy: InterpolationEnergy):
        super().__init__()
        self.interp_energy = interp_energy

    def get_pred(self, shape_x, shape_y):
        raise NotImplementedError()

    def compute_loss(self, shape_x, shape_y, point_pred_arr):
        raise NotImplementedError()

    def forward(self, shape_x, shape_y):
        point_pred_arr = self.get_pred(shape_x, shape_y)
        return self.compute_loss(shape_x, shape_y, point_pred_arr)


class InterpolationModGeoEC(InterpolationModBase):
    def __init__(self, interp_energy: InterpolationEnergy, param=NetParam()):
        super().__init__(interp_energy)
        self.param = param
        param.print_self()
        self.rn_ec = ResnetECPos(c_dim=3, dim=7, hidden_dim=param.hidden_dim)
        self.feat_module = ResnetECPos(
            c_dim=param.hidden_dim, dim=6, hidden_dim=param.hidden_dim
        )
        print("Uses module 'InterpolationModGeoEC'")
        self.Pi = None
        self.Pi_inv = None

    def get_pred(self, shape_x, shape_y, update_corr=True):
        if update_corr:
            self.match(shape_x, shape_y)

        step_size = 1 / (self.param.num_timesteps + 1)
        timesteps = step_size + torch.arange(0, 1, step_size, device=device).unsqueeze(
            1
        ).unsqueeze(
            2
        )  # [T, 1, 1]
        timesteps_up = timesteps * (
            torch.as_tensor([0, 0, 0, 0, 0, 0, 1], device=device, dtype=torch.float)
            .unsqueeze(0)
            .unsqueeze(1)
        )  # [T, 1, 7]

        points_in = torch.cat(
            (
                shape_x.vert,
                torch.mm(self.Pi, shape_y.vert) - shape_x.vert,
                my_zeros((shape_x.vert.shape[0], 1)),
            ),
            dim=1,
        ).unsqueeze(
            0
        )  # [1, n, 7]
        points_in = points_in + timesteps_up

        edge_index = shape_x.get_edge_index()

        displacement = my_zeros([points_in.shape[0], points_in.shape[1], 3])
        for i in range(points_in.shape[0]):
            displacement[i, :, :] = self.rn_ec(points_in[i, :, :], edge_index)
        # the previous three lines used to support batchwise processing in torch-geometric but are now deprecated:
        # displacement = self.rn_ec(points_in, edge_index)  # [T, n, 3]

        point_pred_arr = shape_x.vert.unsqueeze(0) + displacement * timesteps
        point_pred_arr = point_pred_arr.permute([1, 2, 0])
        return point_pred_arr

    def compute_loss(self, shape_x, shape_y, point_pred_arr, n_normalize=201.0):

        E_x_0 = self.interp_energy.forward_single(
            shape_x.vert, point_pred_arr[:, :, 0], shape_x
        ) + self.interp_energy.forward_single(
            point_pred_arr[:, :, 0], shape_x.vert, shape_x
        )

        lambda_align = n_normalize / shape_x.vert.shape[0]
        E_align = (
            lambda_align
            * self.param.lambd
            * (
                (torch.mm(self.Pi, shape_y.vert) - point_pred_arr[:, :, -1]).norm() ** 2
                + (
                    shape_y.vert - torch.mm(self.Pi_inv, point_pred_arr[:, :, -1])
                ).norm()
                ** 2
            )
        )

        if shape_x.D is None:
            E_geo = my_tensor(0)
        elif self.param.lambd_geo == 0:
            E_geo = my_tensor(0)
        else:
            E_geo = (
                self.param.lambd_geo
                * (
                    (
                        torch.mm(torch.mm(self.Pi, shape_y.D), self.Pi.transpose(0, 1))
                        - shape_x.D
                    )
                    ** 2
                ).mean()
            )

        E = E_x_0 + E_align + E_geo

        for i in range(self.param.num_timesteps):
            E_x = self.interp_energy.forward_single(
                point_pred_arr[:, :, i], point_pred_arr[:, :, i + 1], shape_x
            )
            E_y = self.interp_energy.forward_single(
                point_pred_arr[:, :, i + 1], point_pred_arr[:, :, i], shape_x
            )

            E = E + E_x + E_y

        return E, [E - E_align - E_geo, E_align, E_geo]

    def match(self, shape_x, shape_y):
        feat_x = torch.cat((shape_x.vert, shape_x.get_normal()), dim=1)
        feat_y = torch.cat((shape_y.vert, shape_y.get_normal()), dim=1)

        feat_x = self.feat_module(feat_x, shape_x.get_edge_index())
        feat_y = self.feat_module(feat_y, shape_y.get_edge_index())

        feat_x = feat_x / feat_x.norm(dim=1, keepdim=True)
        feat_y = feat_y / feat_y.norm(dim=1, keepdim=True)

        D = torch.mm(feat_x, feat_y.transpose(0, 1))

        sigma = 1e2
        self.Pi = F.softmax(D * sigma, dim=1)
        self.Pi_inv = F.softmax(D * sigma, dim=0).transpose(0, 1)

        return self.Pi


################################################################################################


class InterpolNet:
    def __init__(
        self,
        interp_module: InterpolationModBase,
        dataset,
        dataset_val=None,
        time_stamp=None,
        preproc_mods=[],
        settings_module=None,
    ):
        super().__init__()
        self.time_stamp = time_stamp
        self.interp_module = interp_module
        self.settings_module = settings_module
        self.preproc_mods = preproc_mods
        self.dataset = dataset
        if dataset is not None:
            self.train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=True
            )
        self.dataset_val = dataset_val
        self.i_epoch = 0
        self.optimizer = torch.optim.Adam(
            self.interp_module.parameters(), lr=self.interp_module.param.lr
        )

    def train(self):
        print("start training ...")

        self.interp_module.train()

        while self.i_epoch < self.interp_module.param.num_it:
            tot_loss = 0
            tot_loss_comp = None

            self.update_settings()

            for i, data in enumerate(self.train_loader):
                shape_x = batch_to_shape(data["X"])
                shape_y = batch_to_shape(data["Y"])

                shape_x, shape_y = self.preprocess(shape_x, shape_y)

                loss, loss_comp = self.interp_module(shape_x, shape_y)

                loss.backward()

                if (i + 1) % self.interp_module.param.batch_size == 0 and i < len(
                    self.train_loader
                ) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if tot_loss_comp is None:
                    tot_loss_comp = [
                        loss_comp[i].detach() / self.dataset.__len__()
                        for i in range(len(loss_comp))
                    ]
                else:
                    tot_loss_comp = [
                        tot_loss_comp[i]
                        + loss_comp[i].detach() / self.dataset.__len__()
                        for i in range(len(loss_comp))
                    ]

                tot_loss += loss.detach() / self.dataset.__len__()

            self.optimizer.step()
            self.optimizer.zero_grad()

            print(
                "epoch {:04d}, loss = {:.5f} (arap: {:.5f}, reg: {:.5f}, geo: {:.5f}), reserved memory={}MB".format(
                    self.i_epoch,
                    tot_loss,
                    tot_loss_comp[0],
                    tot_loss_comp[1],
                    tot_loss_comp[2],
                    torch.cuda.memory_reserved(0) // (1024 ** 2),
                )
            )

            if self.time_stamp is not None:
                if (self.i_epoch + 1) % self.interp_module.param.log_freq == 0:
                    self.save_self()
                if (self.i_epoch + 1) % self.interp_module.param.val_freq == 0:
                    self.test(self.dataset_val)

            self.i_epoch += 1

    def test(self, dataset, compute_val_loss=True):
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        shape_x_out = []
        shape_y_out = []
        points_out = []

        tot_loss_val = 0

        for i, data in enumerate(test_loader):
            shape_x = batch_to_shape(data["X"])
            shape_y = batch_to_shape(data["Y"])

            shape_x, shape_y = self.preprocess(shape_x, shape_y)

            point_pred = self.interp_module.get_pred(shape_x, shape_y)

            if compute_val_loss:
                loss, _ = self.interp_module.compute_loss(shape_x, shape_y, point_pred)
                tot_loss_val += loss.detach() / len(dataset)

            shape_x.detach_cpu()
            shape_y.detach_cpu()
            point_pred = point_pred.detach().cpu()

            points_out.append(point_pred)
            shape_x_out.append(shape_x)
            shape_y_out.append(shape_y)

        if compute_val_loss:
            print("Validation loss = ", tot_loss_val)

        return shape_x_out, shape_y_out, points_out

    def preprocess(self, shape_x, shape_y):
        for pre in self.preproc_mods:
            shape_x, shape_y = pre.preprocess(shape_x, shape_y)
        return shape_x, shape_y

    def update_settings(self):
        if self.settings_module is not None:
            self.settings_module.update(self.interp_module, self.i_epoch)

    def save_self(self):
        folder_path = save_path(self.time_stamp)

        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        ckpt_last_name = "ckpt_last.pth"
        ckpt_last_path = os.path.join(folder_path, ckpt_last_name)

        ckpt_name = "ckpt_ep{}.pth".format(self.i_epoch)
        ckpt_path = os.path.join(folder_path, ckpt_name)

        self.save_chkpt(ckpt_path)
        self.save_chkpt(ckpt_last_path)

    def save_chkpt(self, ckpt_path):
        ckpt = {
            "i_epoch": self.i_epoch,
            "interp_module": self.interp_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "par": self.interp_module.param.__dict__,
        }

        torch.save(ckpt, ckpt_path)

    def load_self(self, folder_path, num_epoch=None):
        if num_epoch is None:
            ckpt_name = "ckpt_last.pth"
            ckpt_path = os.path.join(folder_path, ckpt_name)
        else:
            ckpt_name = "ckpt_ep{}.pth".format(num_epoch)
            ckpt_path = os.path.join(folder_path, ckpt_name)

        self.load_chkpt(ckpt_path)

        if num_epoch is None:
            print("Loaded model from ", folder_path, " with the latest weights")
        else:
            print(
                "Loaded model from ",
                folder_path,
                " with the weights from epoch ",
                num_epoch,
            )

    def load_chkpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)

        self.i_epoch = ckpt["i_epoch"]
        self.interp_module.load_state_dict(ckpt["interp_module"])

        if "par" in ckpt:
            self.interp_module.param.from_dict(ckpt["par"])
            self.interp_module.param.print_self()

        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.interp_module.train()


class SettingsBase:
    def update(self, interp_module, i_epoch):
        raise NotImplementedError()


class SettingsFaust(SettingsBase):
    def __init__(self, increase_thresh):
        super().__init__()
        self.increase_thresh = increase_thresh
        print("Uses settings module 'SettingsFaust'")

    def update(self, interp_module, i_epoch):
        if i_epoch < self.increase_thresh:  # 0 - 300
            return
        elif i_epoch < self.increase_thresh * 1.5:  # 300 - 450
            num_t = 1
        elif i_epoch < self.increase_thresh * 1.75:  # 450 - 525
            num_t = 3
        else:  # > 525
            num_t = 7

        interp_module.param.num_timesteps = num_t
        print("Set the # of timesteps to ", num_t)

        interp_module.param.lambd_geo = 0
        print("Deactivated the geodesic loss")


class PreprocessBase:
    def preprocess(self, shape_x, shape_y):
        raise NotImplementedError()


class PreprocessRotateBase(PreprocessBase):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def _create_rot_matrix(self, alpha):
        return create_rotation_matrix(alpha, self.axis)

    def _rand_rot(self):
        alpha = torch.rand(1) * 360
        return self._create_rot_matrix(alpha)

    def rot_sub(self, shape, r):
        if shape.sub is not None:
            for i_p in range(len(shape.sub[0])):
                shape.sub[0][i_p][0, :, :] = torch.mm(shape.sub[0][i_p][0, :, :], r)

        if shape.vert_full is not None:
            shape.vert_full = torch.mm(shape.vert_full, r)

        return shape

    def preprocess(self, shape_x, shape_y):
        raise NotImplementedError()


class PreprocessRotate(PreprocessRotateBase):
    def __init__(self, axis=1):
        super().__init__(axis)
        print("Uses preprocessing module 'PreprocessRotate'")

    def preprocess(self, shape_x, shape_y):
        r_x = self._rand_rot()
        r_y = self._rand_rot()
        shape_x.vert = torch.mm(shape_x.vert, r_x)
        shape_y.vert = torch.mm(shape_y.vert, r_y)
        shape_x = self.rot_sub(shape_x, r_x)
        shape_y = self.rot_sub(shape_y, r_y)
        return shape_x, shape_y


class PreprocessRotateSame(PreprocessRotateBase):
    def __init__(self, axis=1):
        super().__init__(axis)
        print("Uses preprocessing module 'PreprocessRotateSame'")

    def preprocess(self, shape_x, shape_y):
        r = self._rand_rot()
        shape_x.vert = torch.mm(shape_x.vert, r)
        shape_y.vert = torch.mm(shape_y.vert, r)

        shape_x = self.rot_sub(shape_x, r)
        shape_y = self.rot_sub(shape_y, r)
        return shape_x, shape_y


class PreprocessRotateAugment(PreprocessRotateBase):
    def __init__(self, axis=1, sigma=0.3):
        super().__init__(axis)
        self.sigma = sigma
        print(
            "Uses preprocessing module 'PreprocessRotateAugment' with sigma =",
            self.sigma,
        )

    def preprocess(self, shape_x, shape_y):
        r_x = self._rand_rot_augment()
        r_y = self._rand_rot_augment()
        shape_x.vert = torch.mm(shape_x.vert, r_x)
        shape_y.vert = torch.mm(shape_y.vert, r_y)

        shape_x = self.rot_sub(shape_x, r_x)
        shape_y = self.rot_sub(shape_y, r_y)
        return shape_x, shape_y

    # computes a pair of approximately similar rotation matrices
    def _rand_rot_augment(self):
        rot = torch.randn(
            [3, 3], dtype=torch.float, device=device
        ) * self.sigma + my_eye(3)

        U, _, V = torch.svd(rot, compute_uv=True)

        rot = torch.mm(U, V.transpose(0, 1))

        return rot


if __name__ == "__main__":
    print("main of interpolation_net.py")
