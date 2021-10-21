function folder_out = convert_dataset(folder_in, shape_file_ext)

    % Copyright (c) Facebook, Inc. and its affiliates.
    %
    % This source code is licensed under the MIT license found in the
    % LICENSE file in the root directory of this source tree.

    files = dir(fullfile(folder_in, "*" + shape_file_ext));
    folder_out = fullfile(folder_in, "mat/");
    if ~isfolder(folder_out); mkdir(folder_out); end

    for i = 1:length(files)

        fprintf(" Processing %d of %d\n", i, length(files));

        file_in = fullfile(folder_in, files(i).name);
        [~, ~, file_in_ext] = fileparts(file_in);

        file_out = fullfile(folder_out, "shape_" + string(num2str(i - 1, '%03d')) + ".mat");
        if exist(file_out); continue; end

        switch file_in_ext
            case '.off'
                [X.vert, X.triv] = read_off(file_in);
                X.vert = X.vert';
                X.triv = X.triv';
            case '.obj'
                [X.vert, X.triv] = read_obj(file_in);
            case '.ply'
                [X.vert, X.triv] = read_ply(file_in);
            case '.mat'
                load(file_in, 'X');
        end

        X.n = size(X.vert, 1);
        X.m = size(X.triv, 1);

        X.vert = X.vert - mean(X.vert, 1);

        refarea = 0.44;
        X.vert = X.vert ./ sqrt(sum(compute_triangle_areas(X))) .* sqrt(refarea);

        save(file_out, 'X');

    end

end
