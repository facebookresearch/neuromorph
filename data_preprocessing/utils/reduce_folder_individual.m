function folder_out = reduce_folder_individual(shapes_dir, num_vert_reduced)

    % Copyright (c) Facebook, Inc. and its affiliates.
    %
    % This source code is licensed under the MIT license found in the
    % LICENSE file in the root directory of this source tree.

    files = dir(fullfile(shapes_dir, "*.mat"));
    folder_out = fullfile(shapes_dir, "sub_" + string(num_vert_reduced));
    if ~exist(folder_out, 'file'); mkdir(folder_out); end

    for i = 1:length(files)
        fprintf(" Processing %d of %d\n", i, length(files));

        file_curr = fullfile(shapes_dir, files(i).name);
        [~, name, ~] = fileparts(file_curr);

        red_file = fullfile(folder_out, string(name) + ".mat");
        if exist(red_file, 'file'); continue; end

        S = load(file_curr);

        refarea = 0.44;
        S.X.vert = S.X.vert - mean(S.X.vert, 1);
        S.X.vert = S.X.vert ./ sqrt(sum(compute_triangle_areas(S.X))) .* sqrt(refarea);

        [samples, faces] = subsample_shape(S.X, num_vert_reduced);

        X = struct;
        X.triv = faces;
        X.vert = S.X.vert(samples, :);

        save(red_file, 'X')
    end

end

function [samples, faces] = subsample_shape(X, num_vert_reduced)
    X_p.vertices = X.vert;
    X_p.faces = X.triv;

    ratio = num_vert_reduced / size(X.vert, 1);

    if ratio < 1
        X_p = reducepatch(X_p, ratio);
    end

    samples = knnsearch(X.vert, X_p.vertices);
    faces = X_p.faces;
end
