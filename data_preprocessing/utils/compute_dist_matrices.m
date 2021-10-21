function compute_dist_matrices(shapes_dir)

    % Copyright (c) Facebook, Inc. and its affiliates.
    %
    % This source code is licensed under the MIT license found in the
    % LICENSE file in the root directory of this source tree.

    files = dir(fullfile(shapes_dir, "*.mat"));
    matrices_dir = fullfile(shapes_dir, "distance_matrix");
    if ~isfolder(matrices_dir); mkdir(matrices_dir); end

    for i = 1:numel(files)
        fprintf(" Processing %d of %d\n", i, numel(files))
        if exist(fullfile(matrices_dir, files(i).name)), continue; end

        S = load(fullfile(shapes_dir, files(i).name));
        D = compute_dist_matrix(S);
        D = single(D);

        save(fullfile(matrices_dir, files(i).name), 'D');
    end

end
