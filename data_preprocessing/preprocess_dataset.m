function preprocess_dataset(dataset_path, shape_file_extension, resolution_sub)

    % Copyright (c) Facebook, Inc. and its affiliates.
    %
    % This source code is licensed under the MIT license found in the
    % LICENSE file in the root directory of this source tree.

    addpath(genpath(fileparts(mfilename('fullpath'))));

    if ~exist("shape_file_extension", "var")
        shape_file_extension = ".obj";
    end

    if ~exist("resolution_sub", "var")
        resolution_sub = 2000;
    end

    disp("Converting the shape files to .mat files...")
    dataset_path_mat = convert_dataset(dataset_path, shape_file_extension);

    fprintf("Subsampling the shapes to a resolution of %d vertices...\n", resolution_sub)
    dataset_path_sub = reduce_folder_individual(dataset_path_mat, resolution_sub);

    disp("Creating the remeshed version of individual shapes...")
    create_remeshed_dataset(dataset_path_sub);

    disp("Calculating the geodesic distance matrices...")
    compute_dist_matrices(dataset_path_sub);
end
