function [idx_arr, triv_arr] = create_remeshed_collection(file_in, res_array)

    % Copyright (c) Facebook, Inc. and its affiliates.
    %
    % This source code is licensed under the MIT license found in the
    % LICENSE file in the root directory of this source tree.

    S = load(file_in);

    if ~exist("res_array", "var")
        res_array = 200:2000;
    end

    idx_arr = cell(length(res_array), 1);
    triv_arr = cell(length(res_array), 1);

    for i_res = 1:length(res_array)
        num_vert_reduced = res_array(i_res);

        X_p.vertices = S.X.vert;
        X_p.faces = S.X.triv;

        ratio = num_vert_reduced / size(S.X.vert, 1);

        if ratio < 1
            X_p = reducepatch(X_p, ratio);
        end

        idx_arr{i_res} = knnsearch(S.X.vert, X_p.vertices);
        triv_arr{i_res} = X_p.faces;

        X_rec.vert = S.X.vert(idx_arr{i_res}, :);
        X_rec.triv = triv_arr{i_res};
    end

end
