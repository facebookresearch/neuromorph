function create_remeshed_dataset(shapes_dir)

    % Copyright (c) Facebook, Inc. and its affiliates.
    %
    % This source code is licensed under the MIT license found in the
    % LICENSE file in the root directory of this source tree.

    out_dir = fullfile(shapes_dir, "remeshing_idx");
    if ~isfolder(out_dir); mkdir(out_dir); end

    files = dir(fullfile(shapes_dir, "*.mat"));

    for i = 1:length(files)
        fprintf(" Processing %d of %d\n", i, length(files));
        file_in = fullfile(shapes_dir, files(i).name);
        file_out = fullfile(out_dir, files(i).name);
        if exist(file_out); continue; end

        [idx_arr, triv_arr] = create_remeshed_collection(file_in);

        save(file_out, "idx_arr", "triv_arr");
    end

end
