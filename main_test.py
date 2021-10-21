# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from main_train import *
import sys
import os


def save_sequence(
    folder_name, file_name, vert_sequence, shape_x, shape_y, time_elapsed=0
):
    """Saves an interpolation sequence to a .mat file"""

    if not os.path.isdir(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    vert_x = shape_x.vert.detach().cpu().numpy()
    vert_y = shape_y.vert.detach().cpu().numpy()
    triv_x = shape_x.triv.detach().cpu().numpy() + 1
    triv_y = shape_y.triv.detach().cpu().numpy() + 1

    if type(shape_x.samples) is list:
        samples = np.array(shape_x.samples, dtype=np.float32)
    else:
        samples = shape_x.samples.detach().cpu().numpy()

    vert_sequence = vert_sequence.detach().cpu().numpy()

    if shape_x.mahal_cov_mat is None:
        mat_dict = {
            "vert_x": vert_x,
            "vert_y": vert_y,
            "triv_x": triv_x,
            "triv_y": triv_y,
            "vert_sequence": vert_sequence,
            "time_elapsed": time_elapsed,
            "samples": samples,
        }
    else:
        shape_x.mahal_cov_mat = shape_x.mahal_cov_mat.detach().cpu().numpy()
        mat_dict = {
            "vert_x": vert_x,
            "vert_y": vert_y,
            "triv_x": triv_x,
            "triv_y": triv_y,
            "vert_sequence": vert_sequence,
            "time_elapsed": time_elapsed,
            "samples": samples,
            "mahal_cov_mat": shape_x.mahal_cov_mat,
        }

    scipy.io.savemat(os.path.join(folder_name, file_name), mat_dict)


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


def save_seq_collection_hard_correspondences(
    interp_module, shape_x_out, shape_y_out, points_out, res_name
):
    """Save test correspondences on a shape"""

    if not os.path.isdir(os.path.join(data_folder_out, res_name)):
        os.makedirs(os.path.join(data_folder_out, res_name), exist_ok=True)

    if not os.path.isdir(os.path.join(data_folder_out, res_name, "corrs")):
        os.makedirs(os.path.join(data_folder_out, res_name, "corrs"), exist_ok=True)

    print("Saving", len(points_out), "sequences in", os.path.join(data_folder_out, res_name), "...")
    for i in range(len(points_out)):
        vert_x = shape_x_out[i].vert.detach().cpu().numpy()
        vert_y = shape_y_out[i].vert.detach().cpu().numpy()
        triv_x = shape_x_out[i].triv.detach().cpu().numpy()
        triv_y = shape_y_out[i].triv.detach().cpu().numpy()

        plot_curr_shape(vert_x, triv_x)
        plt.savefig(
            os.path.join(
                data_folder_out,
                res_name,
                "seq_" + str(i).zfill(3) + "_" + str(0).zfill(3) + "_x.png",
            )
        )
        plt.clf()

        for j in range(points_out[i].shape[2]):
            vert = points_out[i][:, :, j].detach().cpu().numpy()
            plot_curr_shape(vert, triv_x)
            plt.savefig(
                os.path.join(
                    data_folder_out,
                    res_name,
                    "seq_" + str(i).zfill(3) + "_" + str(j + 1).zfill(3) + ".png",
                )
            )
            plt.clf()

        plot_curr_shape(vert_y, triv_y)
        plt.savefig(
            os.path.join(
                data_folder_out,
                res_name,
                "seq_"
                + str(i).zfill(3)
                + "_"
                + str(points_out[i].shape[2] + 1).zfill(3)
                + "_y.png",
            )
        )
        plt.clf()

        file_name_mat = "seq_" + str(i).zfill(3) + ".mat"
        save_sequence(
            os.path.join(data_folder_out, res_name),
            file_name_mat,
            points_out[i],
            shape_x_out[i],
            shape_y_out[i],
        )

        corr_out = interp_module.match(shape_x_out[i], shape_y_out[i])
        assignment = corr_out.argmax(dim=1).detach().cpu().numpy()
        assignmentinv = corr_out.argmax(dim=0).detach().cpu().numpy()
        file_name_mat_corr = os.path.join(
            data_folder_out, res_name, "corrs", "corrs_" + str(i).zfill(3) + ".mat"
        )
        scipy.io.savemat(
            file_name_mat_corr,
            {
                "assignment": assignment + 1,
                "assignmentinv": assignmentinv + 1,
                "X": {"vert": vert_x, "triv": triv_x + 1},
                "Y": {"vert": vert_y, "triv": triv_y + 1},
            },
        )


def run_test(time_stamp_chkpt=None):
    time_stamp_arr = [time_stamp_chkpt]

    module_arr = None

    hyp_param = HypParam()

    dataset_val = Faust_remeshed_test(2000)

    hyp_param.rot_mod = 0

    for i_time, time_stamp in enumerate(time_stamp_arr):

        if module_arr is not None:
            hyp_param.in_mod = module_arr[i_time]

        print(
            "Evaluating time_stamp",
            time_stamp,
            "with the dataset",
            dataset_val.dataset_name_str(),
        )

        interpol = create_interpol(
            dataset=dataset_val,
            dataset_val=dataset_val,
            time_stamp=time_stamp,
            hyp_param=hyp_param,
        )

        interpol.load_self(save_path(folder_str=time_stamp))

        interpol.interp_module.param.num_timesteps = 1
        shape_x_out, shape_y_out, points_out = interpol.test(dataset_val)
        interpol.interp_module = interpol.interp_module.to(device_cpu)
        save_seq_collection_hard_correspondences(
            interpol.interp_module,
            shape_x_out,
            shape_y_out,
            points_out,
            time_stamp
            + "__"
            + dataset_val.dataset_name_str()
            + "__epoch"
            + str(interpol.i_epoch + 1)
            + "_steps"
            + str(interpol.interp_module.param.num_timesteps),
        )


if __name__ == "__main__":
    run_test(sys.argv[1])
