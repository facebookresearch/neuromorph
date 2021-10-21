# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from model.interpolation_net import *
from utils.arap_interpolation import *
from data.data import *


class HypParam(ParamBase):
    def __init__(self):
        self.increase_thresh = 300

        self.method = "arap"
        self.in_mod = get_in_mod()

        self.load_dist_mat = True
        self.load_sub = True


def get_in_mod():
    in_mod = InterpolationModGeoEC

    return in_mod


def create_interpol(
    dataset,
    dataset_val=None,
    folder_weights_load=None,
    time_stamp=None,
    param=None,
    hyp_param=None,
):

    if time_stamp is None:
        time_stamp = get_timestr()

    if param is None:
        param = NetParam()

    if hyp_param is None:
        hyp_param = HypParam()

    hyp_param.print_self()

    interpol_energy = ArapInterpolationEnergy()

    interpol_module = hyp_param.in_mod(interpol_energy, param).to(device)

    preproc_mods = []

    settings_module = SettingsFaust(increase_thresh=hyp_param.increase_thresh)

    preproc_mods.append(PreprocessRotateSame(dataset.axis))

    interpol = InterpolNet(
        interpol_module,
        dataset,
        dataset_val=dataset_val,
        time_stamp=time_stamp,
        preproc_mods=preproc_mods,
        settings_module=settings_module,
    )

    if folder_weights_load is not None:
        interpol.load_self(save_path(folder_str=folder_weights_load))

    interpol.i_epoch = 0

    return interpol


def remesh_individual(dataset):
    return ShapeDatasetCombineRemesh(dataset)


def create_dataset(
    dataset_cls,
    resolution,
    num_shapes=None,
    load_dist_mat=True,
    remeshing_fct=None,
    load_sub=False,
):
    if num_shapes is None:
        dataset = dataset_cls(
            resolution, load_dist_mat=load_dist_mat, load_sub=load_sub
        )
    else:
        dataset = dataset_cls(
            resolution, num_shapes, load_dist_mat=load_dist_mat, load_sub=load_sub
        )

    if remeshing_fct is not None:
        dataset = remeshing_fct(dataset)

    return dataset


def start_train(dataset, dataset_val=None, folder_weights_load=None):
    interpol = create_interpol(
        dataset, dataset_val=dataset_val, folder_weights_load=folder_weights_load
    )

    interpol.train()

    return interpol


def train_main():
    hyp_param = HypParam()

    # FAUST_remeshed:
    dataset = create_dataset(
        Faust_remeshed_train,
        2000,
        None,
        hyp_param.load_dist_mat,
        remesh_individual,
        hyp_param.load_sub,
    )
    dataset_val = create_dataset(
        Faust_remeshed_test,
        2000,
        None,
        hyp_param.load_dist_mat,
        remesh_individual,
        hyp_param.load_sub,
    )

    start_train(dataset, dataset_val)


if __name__ == "__main__":
    train_main()
