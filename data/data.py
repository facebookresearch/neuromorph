# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch.utils.data
import scipy.io
from utils.shape_utils import *


def input_to_batch(mat_dict):
    dict_out = dict()

    for attr in ["vert", "triv"]:
        if mat_dict[attr][0].dtype.kind in np.typecodes["AllInteger"]:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.int32)
        else:
            dict_out[attr] = np.asarray(mat_dict[attr][0], dtype=np.float32)

    return dict_out


def batch_to_shape(batch):
    shape = Shape(batch["vert"].squeeze().to(device), batch["triv"].squeeze().to(device, torch.long) - 1)

    if "D" in batch:
        shape.D = batch["D"].squeeze().to(device)

    if "sub" in batch:
        shape.sub = batch["sub"]
        for i_s in range(len(shape.sub)):
            for i_p in range(len(shape.sub[i_s])):
                shape.sub[i_s][i_p] = shape.sub[i_s][i_p].to(device)

    if "idx" in batch:
        shape.samples = batch["idx"].squeeze().to(device, torch.long)

    if "vert_full" in batch:
        shape.vert_full = batch["vert_full"].squeeze().to(device)

    return shape


class ShapeDatasetBase(torch.utils.data.Dataset):
    def __init__(self, axis=1):
        self.axis = axis
        self.num_shapes = None

    def dataset_name_str(self):
        raise NotImplementedError()

    def _get_file_from_folder(self, i, folder_path):
        shape_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        shape_files.sort()
        return os.path.join(folder_path, shape_files[i])


class ShapeDatasetInMemory(ShapeDatasetBase):
    def __init__(self, folder_path, num_shapes, axis=1, load_dist_mat=False, load_sub=False):
        super().__init__(axis)
        self.folder_path = folder_path
        self.num_shapes = num_shapes
        self.axis = axis
        self.load_dist_mat = load_dist_mat
        self.load_sub = load_sub

        self.data = []

        self._init_data()

    def _init_data(self):
        for i in range(self.num_shapes):
            file_name = self._get_file(self._get_index(i))
            load_data = scipy.io.loadmat(file_name)

            data_curr = input_to_batch(load_data["X"][0])

            print("Loaded file ", file_name, "")

            if self.load_dist_mat:
                file_name = self._get_file_from_folder(self._get_index(i),
                                                       os.path.join(self.folder_path, "distance_matrix"))
                load_dist = scipy.io.loadmat(file_name)
                load_dist["D"][load_dist["D"] > 1e2] = 2
                data_curr["D"] = np.asarray(load_dist["D"], dtype=np.float32)
                print("Loaded file ", file_name, "")

            self.data.append(data_curr)

    def _get_file(self, i):
        return self._get_file_from_folder(i, self.folder_path)

    def _get_index(self, i):
        return i

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def dataset_name_str(self):
        raise NotImplementedError()


class ShapeDatasetCombine(ShapeDatasetInMemory):
    def __init__(self, folder_path, num_shapes, axis=1, load_dist_mat=False, load_sub=False):
        super().__init__(folder_path, num_shapes, axis, load_dist_mat, load_sub)
        self.num_pairs = num_shapes ** 2
        print("loaded", self.dataset_name_str(), "with", self.num_pairs, "pairs")

    def __getitem__(self, index):
        i1 = int(index / self.num_shapes)
        i2 = int(index % self.num_shapes)
        data_curr = dict()
        data_curr["X"] = self.data[i1]
        data_curr["Y"] = self.data[i2]
        data_curr["axis"] = self.axis
        return data_curr

    def __len__(self):
        return self.num_pairs

    def dataset_name_str(self):
        raise NotImplementedError()


class ShapeDatasetCombineRemesh(ShapeDatasetBase):
    def __init__(self, dataset: ShapeDatasetCombine, remeshing_folder="remeshing_idx"):
        super().__init__(dataset.axis)
        self.dataset = dataset
        self.data = self.dataset.data
        self.num_shapes = self.dataset.num_shapes
        self.idx_arr_arr = None
        self.triv_arr_arr = None
        self.remeshing_folder = remeshing_folder
        self._init_mesh_info()
        print("Using the precomputed remeshings of", self.dataset_name_str())

    def _init_mesh_info(self):
        self.idx_arr_arr = []
        self.triv_arr_arr = []

        for i in range(self.dataset.num_shapes):
            remesh_file = self._get_file_from_folder(self.dataset._get_index(i), os.path.join(self.dataset.folder_path, self.remeshing_folder))
            mesh_info = scipy.io.loadmat(remesh_file)
            idx_arr = mesh_info["idx_arr"]
            triv_arr = mesh_info["triv_arr"]

            print("Loaded file ", remesh_file, "")

            self.idx_arr_arr.append(idx_arr)
            self.triv_arr_arr.append(triv_arr)

    def __getitem__(self, index):
        data_curr = self.dataset[index]

        i1 = int(index / self.dataset.num_shapes)
        i2 = int(index % self.dataset.num_shapes)

        idx_arr_x = self.idx_arr_arr[i1]
        idx_arr_y = self.idx_arr_arr[i2]

        triv_arr_x = self.triv_arr_arr[i1]
        triv_arr_y = self.triv_arr_arr[i2]

        i_mesh_x = random.randint(0, idx_arr_x.shape[0] - 1)
        i_mesh_y = random.randint(0, idx_arr_y.shape[0] - 1)

        data_new = dict()
        data_new["X"] = dict()
        data_new["Y"] = dict()

        idx_x = idx_arr_x[i_mesh_x][0].astype(np.long) - 1
        idx_y = idx_arr_y[i_mesh_y][0].astype(np.long) - 1

        data_new["X"]["vert_full"] = data_curr["X"]["vert"]
        data_new["Y"]["vert_full"] = data_curr["Y"]["vert"]
        data_new["X"]["idx"] = idx_x
        data_new["Y"]["idx"] = idx_y

        data_new["X"]["vert"] = data_curr["X"]["vert"][idx_x, :]
        data_new["Y"]["vert"] = data_curr["Y"]["vert"][idx_y, :]
        data_new["X"]["triv"] = triv_arr_x[i_mesh_x][0].astype(np.long)
        data_new["Y"]["triv"] = triv_arr_y[i_mesh_y][0].astype(np.long)

        if "D" in data_curr["X"]:
            idx_x = idx_x.squeeze()
            idx_y = idx_y.squeeze()
            data_new["X"]["D"] = data_curr["X"]["D"][:, idx_x][idx_x, :]
            data_new["Y"]["D"] = data_curr["Y"]["D"][:, idx_y][idx_y, :]

        if "sub" in data_curr["X"]:
            data_new["X"]["sub"] = data_curr["X"]["sub"]
            data_new["Y"]["sub"] = data_curr["Y"]["sub"]
            data_new["X"]["idx"] = idx_x
            data_new["Y"]["idx"] = idx_y

        data_new["axis"] = self.axis

        return data_new

    def __len__(self):
        return len(self.dataset)

    def dataset_name_str(self):
        return self.dataset.dataset_name_str()


def get_faust_remeshed_folder(resolution):
    if resolution is None:
        folder_path = os.path.join(data_folder_faust_remeshed, "full")
    else:
        folder_path = os.path.join(data_folder_faust_remeshed, "sub_" + str(resolution))
    return folder_path


def get_shrec20_folder(resolution):
    if resolution is None:
        folder_path = os.path.join(data_folder_shrec20, "full")
    else:
        folder_path = os.path.join(data_folder_shrec20, "sub_" + str(resolution))
    return folder_path


class Faust_remeshed_train(ShapeDatasetCombine):
    def __init__(self, resolution, num_shapes=80, load_dist_mat=False, load_sub=False):
        self.resolution = resolution
        super().__init__(get_faust_remeshed_folder(resolution), num_shapes, load_dist_mat=load_dist_mat, load_sub=load_sub)

    def dataset_name_str(self):
        return "FAUST_remeshed_" + str(self.resolution) + "_train"


class Faust_remeshed_test(ShapeDatasetCombine):
    def __init__(self, resolution, num_shapes=20, load_dist_mat=False, load_sub=False):
        self.resolution = resolution
        super().__init__(get_faust_remeshed_folder(resolution), num_shapes, load_dist_mat=load_dist_mat, load_sub=load_sub)

    def _get_index(self, i):
        return i+80

    def dataset_name_str(self):
        return "FAUST_remeshed_" + str(self.resolution) + "_test"


class Mano_train(ShapeDatasetCombine):
    def __init__(self, resolution=None, num_shapes=100, load_dist_mat=False, load_sub=False):
        super().__init__(data_folder_mano_right, num_shapes, axis=2, load_dist_mat=load_dist_mat, load_sub=load_sub)

    def dataset_name_str(self):
        return "Mano_train"


class Mano_test(ShapeDatasetCombine):
    def __init__(self, resolution=None, num_shapes=20, load_dist_mat=False, load_sub=False):
        super().__init__(data_folder_mano_test, num_shapes, axis=2, load_dist_mat=load_dist_mat, load_sub=load_sub)

    def dataset_name_str(self):
        return "Mano_test"


class Shrec20_full(ShapeDatasetCombine):
    def __init__(self, resolution, num_shapes=14, load_dist_mat=False, load_sub=False):
        self.resolution = resolution
        super().__init__(get_shrec20_folder(resolution), num_shapes, load_dist_mat=load_dist_mat, load_sub=load_sub)

    def dataset_name_str(self):
        return "Shrec20_" + str(self.resolution) + "_train"



if __name__ == "__main__":
    print("main of data.py")
